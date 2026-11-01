#!/usr/bin/env bash
set -euo pipefail

ROOT="training/outputs/phase12_headless"
OUT="${OUT:-$ROOT/round528_policy_population_matrix_128x1_20261031}"
SELECTOR="${SELECTOR:-$ROOT/round402_selector_currentdata_8ep_20261030/selector_candidate.pt}"
SHOWDOWN_PATH="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
PAIRS="${PAIRS:-128}"
REPEATS="${REPEATS:-1}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-2026115288}"

names=(round430 round472 round473 round476 round485 round524 round102)
checkpoints=(
  "$ROOT/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt"
  "$ROOT/round472_fresh_robust_mixed_ppo_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round473_fresh_hard_mixed_ppo_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round485_maxdamage_exposure_ppo_20k_20261031/checkpoints/latest.pt"
  "$ROOT/round524_opening5_teacher_low_lr_20k_20261031/checkpoints/latest.pt"
  "$ROOT/direct_policy_round102_round101_class_balanced_round94_value_gae_ppo_20k_20261024/checkpoints/latest.pt"
)
for file in "${checkpoints[@]}" "$SELECTOR"; do
  [[ -s "$file" ]] || { echo "Missing required checkpoint: $file" >&2; exit 1; }
done
[[ -d "$SHOWDOWN_PATH/dist/sim" ]] || { echo "Missing local Showdown simulator: $SHOWDOWN_PATH" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing matrix run: $OUT" >&2; exit 1; }
[[ "$PAIRS" =~ ^[1-9][0-9]*$ && "$REPEATS" =~ ^[1-9][0-9]*$ && "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "PAIRS, REPEATS, and WORKERS must be positive integers" >&2; exit 2;
}
mkdir -p "$OUT"
export POKEMON_SHOWDOWN_PATH="$SHOWDOWN_PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python3 - "$OUT/policies.json" "${names[@]}" <<'PY'
import json,sys
from pathlib import Path
names=sys.argv[2:]
root=Path('training/outputs/phase12_headless')
paths={
'round430':root/'round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt',
'round472':root/'round472_fresh_robust_mixed_ppo_20k_20261031/checkpoints/latest.pt',
'round473':root/'round473_fresh_hard_mixed_ppo_20k_20261031/checkpoints/latest.pt',
'round476':root/'round476_refreshed_population_ppo_20k_20261031/checkpoints/latest.pt',
'round485':root/'round485_maxdamage_exposure_ppo_20k_20261031/checkpoints/latest.pt',
'round524':root/'round524_opening5_teacher_low_lr_20k_20261031/checkpoints/latest.pt',
'round102':root/'direct_policy_round102_round101_class_balanced_round94_value_gae_ppo_20k_20261024/checkpoints/latest.pt',
}
Path(sys.argv[1]).write_text(json.dumps({'names':names,'checkpoints':{n:str(paths[n]) for n in names}},indent=2)+'\n')
PY

for ((i=0; i<${#names[@]}; i++)); do
  for ((j=i+1; j<${#names[@]}; j++)); do
    matchup="${names[$i]}_vs_${names[$j]}"
    echo "Evaluating $matchup ($PAIRS team clusters)" >&2
    venv/bin/python scripts/evaluate_native_paired_parallel.py \
      --checkpoint "${checkpoints[$i]}" --opponent policy \
      --opponent-checkpoint "${checkpoints[$j]}" \
      --team-selection-checkpoint "$SELECTOR" \
      --opponent-team-selection-checkpoint "$SELECTOR" \
      --pairs "$PAIRS" --outcome-repeats "$REPEATS" \
      --workers "$WORKERS" --seed "$SEED" --device cpu \
      --battle-format gen9championsdoubles2026mcclosed \
      --output-dir "$OUT/$matchup"
  done
done

venv/bin/python - "$OUT" "$PAIRS" "$REPEATS" "$WORKERS" "$SEED" <<'PY'
import csv,json,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
root=Path(sys.argv[1]); requested=int(sys.argv[2]); repeats=int(sys.argv[3]); workers=int(sys.argv[4]); seed=int(sys.argv[5])
manifest=json.loads((root/'policies.json').read_text()); names=manifest['names']; n=len(names)
cluster_scores={}; summaries={}; throughput={};errors={};adjustments={};total_games=0;total_seconds=0.0
for i,a in enumerate(names):
 for j in range(i+1,n):
  b=names[j]; folder=root/f'{a}_vs_{b}'; d=json.loads((folder/'paired_summary.json').read_text()); summaries[(i,j)]=d
  rows={}
  for row in d['results']:
   if row.get('complete'):
    rows.setdefault(int(row['team_seed']),[]).append(float(row['checkpoint_points'])/2.0)
  cluster_scores[(i,j)]={k:float(np.mean(v)) for k,v in rows.items()}
  throughput[f'{a}_vs_{b}']=float(d['aggregate_games_per_second'])
  errors[f'{a}_vs_{b}']=int(d['timeouts_or_errors']);adjustments[f'{a}_vs_{b}']=int(d['action_adjustments'])
  total_games+=int(d['games_completed']);total_seconds+=float(d['parallel_wall_seconds'])
common=sorted(set.intersection(*(set(x) for x in cluster_scores.values())))
if len(common)<max(32,requested//2): raise SystemExit(f'only {len(common)} common completed team clusters')
M=np.full((n,n),0.5,dtype=np.float64);ci={}
for (i,j),scores in cluster_scores.items():
 vals=np.asarray([scores[k] for k in common],dtype=np.float64);M[i,j]=float(vals.mean());M[j,i]=1.0-M[i,j]
 rng=np.random.default_rng(seed+i*100+j)
 ix=rng.integers(0,len(vals),(10000,len(vals))); boot=vals[ix].mean(axis=1)
 ci[f'{names[i]}_vs_{names[j]}']=[float(x) for x in np.quantile(boot,[.025,.975])]
# Robust maximin mixed strategy for this finite checkpoint population.
def solve(A):
 c=np.r_[np.zeros(A.shape[0]),-1.0]
 aub=np.c_[-A.T,np.ones(A.shape[1])]
 res=linprog(c,A_ub=aub,b_ub=np.zeros(A.shape[1]),A_eq=np.array([[*np.ones(A.shape[0]),0.0]]),b_eq=np.array([1.0]),bounds=[(0,None)]*A.shape[0]+[(None,None)],method='highs')
 if not res.success: raise RuntimeError('minimax LP failed: '+res.message)
 p=np.maximum(res.x[:-1],0);p/=p.sum();return p,float(res.x[-1])
point_mix,point_value=solve(M)
rng=np.random.default_rng(seed+528);B=2000;mixes=np.zeros((B,n));values=np.zeros(B)
common_arr=np.asarray(common,dtype=np.int64)
seed_ix=rng.integers(0,len(common_arr),(B,len(common_arr)))
for k in range(B):
 sample=common_arr[seed_ix[k]]
 A=np.full((n,n),.5,dtype=np.float64)
 for (i,j),scores in cluster_scores.items():
  vals=np.asarray([scores[x] for x in common],dtype=np.float64)
  A[i,j]=float(vals[sample].mean());A[j,i]=1.0-A[i,j]
 mixes[k],values[k]=solve(A)
result={
 'status':'Complete; restricted policy-population payoff screen',
 'method':'Unordered head-to-head games among policy checkpoints, both player seats, common preview selector, common generated teams across pairings, paired cluster bootstrap. Randomness inside Showdown is server-side. The minimax mix is only for this finite checkpoint population.',
 'policies':names,'checkpoints':manifest['checkpoints'],'pairs_requested_per_matchup':requested,
 'repeats_per_team_cluster':repeats,'workers':workers,'seed':seed,
 'common_team_clusters':len(common),'payoff_matrix_row_score_rate':M.tolist(),
 'payoff_matrix_ci95_by_pair':ci,
 'restricted_maximin_mix':{name:float(point_mix[i]) for i,name in enumerate(names)},
 'restricted_maximin_value':point_value,
 'bootstrap_replicates':B,
 'bootstrap_maximin_value_ci95':[float(x) for x in np.quantile(values,[.025,.975])],
 'bootstrap_mean_strategy':{name:float(mixes[:,i].mean()) for i,name in enumerate(names)},
 'strategy_weight_ci95':{name:[float(x) for x in np.quantile(mixes[:,i],[.025,.975])] for i,name in enumerate(names)},
 'strategy_support_frequency_gt_5pct':{name:float((mixes[:,i]>.05).mean()) for i,name in enumerate(names)},
 'timeouts_or_errors_by_matchup':errors,'action_adjustments_by_matchup':adjustments,
 'aggregate_games_completed':total_games,'sum_matchup_worker_wall_seconds':total_seconds,
 'sequential_end_to_end_games_per_second':total_games/max(total_seconds,1e-9),
 'matchup_games_per_second':throughput,
 'interpretation':'A restricted empirical matrix is useful only if the bootstrap shows a stable exploitable cycle/mix. It is not a Nash equilibrium of the full hidden-information VGC game and does not establish tournament strength.',
 'reproduction':'scripts/evaluate_round528_policy_population_matrix.sh'
}
(root/'payoff_matrix.csv').write_text('policy,'+','.join(names)+'\n'+'\n'.join(nm+','+','.join(f'{x:.6f}' for x in M[i]) for i,nm in enumerate(names))+'\n')
(root/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['common_team_clusters','restricted_maximin_mix','restricted_maximin_value','bootstrap_maximin_value_ci95','bootstrap_mean_strategy','strategy_support_frequency_gt_5pct','aggregate_games_completed','sequential_end_to_end_games_per_second']},indent=2))
PY
