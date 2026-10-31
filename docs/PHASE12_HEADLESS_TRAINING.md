# Headless training and paired evaluation update

New run data is stored under `outputs/phase12_headless/`; the previous
training iteration and its large artifacts remain under
`archive/current-training/`.

## Training path

`scripts/train_headless_ppo.py` collects rollouts through the persistent local
Showdown `Battle` API, without routing every action through the websocket
server. It trains the existing native PPO policy with terminal win/loss reward
against a bounded pool of older policies and a random-action mixture. Rollout
jobs run in separate processes with CPU math thread counts capped to avoid
oversubscription. The normal websocket server remains the evaluation and LAN
play environment.

Each round writes JSONL rollouts and a versioned checkpoint. The runner checks
that a candidate contains the model configuration and weights, then atomically
replaces both its run-local `checkpoints/latest.pt` and the optional
`--publish-to` path. The LAN bot reads that path at team preview and keeps the
loaded model for the rest of the match. The starting checkpoint is copied to
an immutable file under `inputs/` so publishing cannot change the run's source
policy or an opponent-pool entry.

Example:

```bash
POKEMON_SHOWDOWN_PATH=/path/to/pokemon-showdown PYTHONPATH=. \
  venv/bin/python scripts/train_headless_ppo.py \
  --rounds 5 --episodes-per-round 24 --random-fraction 0.25 \
  --opponent-checkpoint outputs/phase5/pool_run_fixed/checkpoints/round_0003/update_000001.pt \
  --opponent-pool-size 4 \
  --initial-checkpoint outputs/phase10/live_training/initial.pt \
  --output-root outputs/phase12_headless/robust_v1 \
  --publish-to outputs/phase10/live_training/checkpoints/latest.pt \
  --seed 20261022 --epochs 4 --minibatch-size 128 --rollout-workers 4
```

The run completed five rounds and 120 headless matches. PPO trained on 1,534
player-one decisions over the five updates. Round wall times including PPO were
17.5–22.3 seconds, or 1.14 matches/second across the complete run. Individual
rollout jobs measured 0.48–0.61 matches/second; the Showdown engine accounted
for roughly 0.06–0.13 seconds per match, while CPU inference and process startup
were larger costs. This is faster than the earlier 0.49–0.75 matches/second
websocket rollout profiles, but still far below the raw simulator microbench
because neural inference and data handling dominate. No CUDA simulator work is
justified by this profile.

Two default-target translations were recorded across the 120 completed
matches. They
cover targetable moves for which poke-env exposed a no-location order; the
adapter now chooses a valid deterministic location based on the move's target
type. The reproduced Solar Beam and Milk Drink cases complete in the local
Showdown engine. The existing closed-sheet observation parity checks remain
the information-boundary reference.

The new `damage_ko_friendly_fire` action-feature variant adds an explicit
ally-target flag and suppresses the opponent-damage estimate for those actions.
It can be bootstrapped without changing the round-five actor weights:

```bash
PYTHONPATH=. venv/bin/python scripts/upgrade_native_action_features.py \
  --input outputs/phase12_headless/robust_v1/checkpoints/round_0005/update_000001.pt \
  --output outputs/phase12_headless/target_aware/initial.pt \
  --feature-variant damage_ko_friendly_fire
```

Train it from fresh target-aware rollouts with `scripts/train_headless_ppo.py`
and keep it in a separate output root until paired results justify promotion.

The zero-update ablation completed three paired comparisons against round five:
the target-aware checkpoint won 2 of 6 completed legs, lost 4, and one pair
timed out in both orientations. This does not show an improvement yet. The
feature is being carried into fresh PPO training to let outcomes tune the
initial ally-target penalty; the ablation checkpoint is not published to the
LAN service.

## Evaluation of round five

The side-swapped local-server evaluator reused each team and preview selection
for both legs:

| Opponent | Completed | Wins | Score rate | Paired outcomes | Timeouts / action adjustments |
|---|---:|---:|---:|---|---:|
| Max-damage | 8 | 0 | 0.00 | 4 pair losses | 0 / 0 |
| Preserved initial policy | 8 | 4 | 0.50 | 2 pair sweeps, 2 pair losses | 0 / 0 |

The initial-policy comparison uses the immutable
`outputs/phase10/live_training/initial.pt`. The LAN `latest.pt` file changes
during publication, so it must not be used as a historical opponent path.
The two evaluations are small samples; round five still fails every
max-damage matchup and does not demonstrate robust strength.

A traced round-five loss completed in eight decisions with no timeout,
protocol error, or action adjustment. It included an early switch into
Gourgeist that was KO'd on the following turn and an Earth Power choice
targeting an ally. This points to action selection and target evaluation as
remaining weaknesses. Increasing PPO rollout volume alone did not remove the
weakness; the next improvement should make search or target-aware action
features influence decisions, then compare at a fixed decision-time budget.

Artifacts:

- Rollouts and round checkpoints: `outputs/phase12_headless/robust_v1/`
- Paired max-damage results: `outputs/phase12_headless/paired_round5_maxdamage/`
- Paired historical-policy results: `outputs/phase12_headless/paired_round5_vs_initial_correct/`
- Traced loss: `outputs/phase12_headless/loss_review_round5/max_damage_trace.jsonl`

## Vocabulary-v1 follow-up

The legacy token tables covered only a small part of the generated Champions
rosters, so `src/features/vocab.py` now hashes unknown canonical species,
moves, items, and abilities into unused embedding rows. The vocabulary
migration zero-initializes those rows, preserving the source policy's initial
behavior while allowing PPO to learn them. The 5-round run under
`outputs/phase12_headless/vocab_v1/robust_v1/` completed 160 games and
published round five to the LAN `latest.pt` path.

On 12 paired, side-swapped games per opponent, its round-five checkpoint scored
1/12 against max-damage, 0/12 against counterplay, 11/12 against random, and
9/12 against the previous robust_v4 policy. All 48 games completed without
timeouts or action adjustments. These results show it did not improve
worst-case heuristic strength. A second-stage run started from that checkpoint
with 30% max-damage and 30% counterplay opponents (up from 25% and 10%). It
completed rollout collection but stopped at PPO's initial log-probability
consistency guard before its first update, so it published no checkpoint and
is no longer running. The LAN path remains on the completed vocabulary-v1
round-five checkpoint. The reported maximum discrepancy was 0.0494 with
approximate KL 0.000124; this needs diagnosis before retrying.

## Speed snapshot (2026-10-19)

Keep the following rates distinct: raw engine throughput excludes policy
inference and training, while rollout and round rates include progressively
more of the end-to-end pipeline.

- Raw simulator benchmark: 800 complete matches in 17.06 s across eight CPU
  workers (46.9 matches/s aggregate), or 1,876 state transitions/s. Worker
  RSS averaged about 502 MB.
- Recent PPO rounds: 32 completed rollout matches plus PPO work in 25.45–26.95
  s (1.19–1.26 matches/s end to end).
- Most recent rollout jobs: 0.46–0.62 games/s per job, varying with opponent
  and episode length. In the earlier headless profile the Showdown engine
  spent about 0.06–0.13 s per match, leaving inference, feature/data handling,
  and process startup as the larger costs.
- Current bottleneck: the mixed-opponent retry is blocked by the PPO
  log-probability consistency guard, not by simulator throughput. After that
  is resolved, profile CPU policy inference and rollout-worker overhead before
  increasing simulation/search complexity. No GPU simulator rewrite is
  justified by current measurements.

The same 828-match corpus provides 17,798 player-view states for a
vocabulary-v1 value model. The corrected roster-disjoint split evaluates 3,596
held-out states and excludes every held-out match from training. Best Brier
score was 0.224 at epoch 3, with log loss 0.672 and ECE 0.112; later epochs
overfit. The model is saved at
`outputs/phase12_headless/state_value_vocab_v1_disjoint/best.pt` and remains
experimental. The optional LAN opening search can use it with
`--state-value-checkpoint .../best.pt`; `--state-value-device auto` selects
GPU when available. The model is not used by default because its calibration
and competitive value are still weak.

## Speed-first and evaluator refresh (2026-10-19)

The old speed snapshot above is retained as historical context. The optimized
worker and legal-mask path now reaches 4.08 policy-vs-random games/s on the
matched 8-game workload (1.96 s versus 13.08 s before optimization), with
exact seeded parity across all 16 player-view records. A full mixed-opponent
32-game rollout plus PPO round completed in 10.11 s, or 3.17 completed games/s
end to end, exceeding the initial 2.4 games/s target. Current measured
per-match costs are about 0.109 s for policy/action selection, 0.025 s for
observation and legal-mask construction, 0.039 s for worker advancement and
state parsing, and 0.05 s for team generation/setup. Policy/action selection
is now the largest measured component; engine-only throughput is 46.9 games/s
across eight CPU workers. The isolated speed-profile PPO checkpoint was not
published to the LAN service.

The refreshed win-probability evaluator uses 22,705 states from 1,002 closed-
sheet game IDs, split by roster into 10,713 training, 2,024 calibration, and
4,495 final test examples. The calibrated 32-dimensional model selected epoch
16 and measured Brier 0.2016, log loss 0.5877, accuracy 0.6863, and ECE 0.0647
on the untouched test set, compared with the constant-probability Brier
baseline of 0.25. It returns one calibrated win likelihood in `[0, 1]` and is
saved at
`outputs/phase12_headless/state_value_v2/training_calibrated_small/best.pt`.
This improves on the older vocabulary-v1 model, but remains experimental
until equal-budget search strength tests support it. Detailed commands and
profiles are in `docs/PHASE2_RUNTIME_PROFILE.md` and `docs/PHASE4_DATA.md`.

## Earlier speed status before the 480-game replay-alignment run (2026-10-19)

At that point, training throughput evidence was 3.17 completed games/s
for the 32-game mixed rollout plus PPO round, and 4.08 games/s for the matched
policy-vs-random rollout workload. No fresh full PPO round was run today. The
measured training bottleneck remains policy/action selection, not Showdown
turn advancement.

For the corrected-context evaluator refresh, a separate four-job collection
run generated 1,000 games in 77 seconds (13.0 aggregate games/s, about 21.4
hours per million collection-only games). With Torch CPU threads capped to one
per job, policy inference ranged from 0.065 to 0.18 seconds per match across
opponent mixes. This was separate from the then-current 17.83 games/s mixed PPO rate;
it is a distinct workload. See `docs/PHASE2_RUNTIME_PROFILE.md` for the
one-step search benchmark and its separate latency profile.

Search-enabled full matches are tracked separately. A two-game learned-search
run using four persistent local Showdown workers took 2.17 s total (0.92
games/s). Individual rates were 0.61 and 2.25 games/s; this is a small latency
sample, not a stable throughput estimate. Search subprocess wall fell from
567 ms cold to 239 ms warm; adapter wall fell from 1,121 ms to 325 ms. A
persistent hidden-team sampler reduced setup from 280 ms cold to 6 ms warm.
Learned frontier application fell from 273 ms cold to 79 ms warm, including
199 ms versus 5.1 ms for model forward. The current warm search bottleneck is
Showdown search, followed by frontier feature/model processing; training and
search rates must not be conflated.

Both complete matches finished with zero invalid-action fallbacks. Search used
12 worlds and four candidates per side (192 simulated turn advances per
opening). The latest eight-seed strength pilot against max-damage produced
1/8 wins for material search and 0/8 for either learned frontier. It is too
small for a strength conclusion. See `PHASE6_SEARCH.md` and `PLAN.md` for
search details and limitations.

A same-seed worker scaling check selected identical openings at 2, 4, and 8
workers. Warm adapter times were 461, 325, and 307 ms, while complete-run
rates were 0.82, 0.92, and 0.92 games/s. Four remains the default pending
memory measurement because eight workers gave no overall throughput gain.

## State-value refresh with per-decision public roster inputs (2026-10-19)

The dataset builder now reconstructs each example with the species-only
opponent roster snapshot recorded for that decision, matching live inference
and preserving the closed-team-sheet boundary. Focused state-value tests pass.
The combined corpus contains 4,102 safe player-view records from 1,546
matches. With the existing 64-dimensional, two-layer architecture, the model
selected epoch 2 and scored held-out Brier 0.1830, log loss 0.5431, accuracy
0.7175, and ECE 0.0254 across 8,569 states. This beats the constant-0.5 Brier
baseline (0.25) but does not establish convergence: training data are still
far below the planned million-game scale, and match states are correlated.
Checkpoint: `outputs/phase12_headless/state_value_v7_recent_roster_snapshot/evaluator_1331/best.pt`.

## PPO opponent-group balance experiment (2026-10-19)

The headless PPO runner now forwards its explicit Showdown checkout path to
the local-server evaluator through `POKEMON_SHOWDOWN_PATH`. It also exposes
`--opponent-balanced-loss` and `--opponent-balance-power`, reusing the native
trainer's group weighting without changing the default behavior. This
prevents local-server evaluation from failing when the caller supplied
`--showdown-path` but had not separately exported the environment variable.

A 480-game round with a 25% random, max-damage, counterplay, and policy-pool
mixture and tempered balance power 0.5 completed 4,690 transitions / 148
optimizer steps. It ran in 29.49 seconds (16.27 games/s including PPO) with
31.52 aggregate rollout games/s. Against the same eight side-swapped pairs
per opponent, the balanced branch scored 13/16 vs random, 2/16 vs max-damage,
1/16 vs counterplay, and 7/16 vs the prior policy. Its unbalanced one-round
comparison scored 13/16, 0/16, 2/16, and 7/16 on the same team pairs. This
small comparison is inconclusive and remains experimental; the balance option
does not solve the severe worst-case gap. Full tables and resource timings
are in `docs/PHASE11_EVALUATION.md` and `docs/PHASE2_RUNTIME_PROFILE.md`.

The balanced branch was continued for four more 480-game rounds (1,920 games,
18,366 transitions). Its final checkpoint scored 54/64 vs random, 7/64 vs
max-damage, 9/64 vs counterplay, and 41/64 vs the starting policy, using the
same 32 side-swapped team pairs per opponent as the matched comparison. It
improved against max-damage but slipped against counterplay; its worst rate
was still 10.9%. It remains an experimental training-pool checkpoint, not a
LAN promotion. Full results and aggregate speed are recorded in the Phase 11
evaluation and Phase 2 runtime profile.

## Mixed PPO throughput and input-replay alignment (2026-10-19)

A legal-action-only action-feature optimization reduced feature calculation
from 1.514 to 0.219 ms in a real-state microbenchmark and full prediction from
3.85 to 2.62 ms. Its first mixed PPO continuation failed the pre-update
log-probability guard on CUDA (max absolute error 0.0772, approximate KL
0.000110); retrying with a smaller minibatch still failed. Replaying saved
records on CPU and CUDA produced the same mismatch, ruling out a CUDA-only
numerical issue. The rollout policy was using the public opponent species
roster from the battle preview, while PPO reconstructed model inputs from the
match-level team order. The exact per-decision, species-only opponent roster is
now recorded and used for PPO input reconstruction and Fake Out context; the
perspective projection validates and retains only that acting player's public
snapshot. This adds no opponent moves, item, or ability data.

A 16-game mixed smoke round completed the consistency check and PPO update
(155 transitions). A fresh 480-game round then completed 4,705 transitions
and 38 optimizer steps, passing the same guard. End-to-end time was 27.94 s
(17.18 games/s), including the PPO update. Rollouts took 16.11 s (29.79
aggregate games/s), the PPO update 11.79 s, peak aggregate memory was 10,134
MiB (9.90 GiB), and average CPU use was 10.11 cores. Weighted per-match
rollout components were 0.108 s for action selection, 0.076 s for policy
inference within that, 0.056 s for Showdown advancement, 0.038 s for
observation encoding, and 0.014 s for trajectory writing. A 1-million-game
projection at this full mixed PPO rate is 16.2 hours before evaluation,
conversion, idle time, and workload changes; this is a throughput projection,
not evidence of evaluator convergence.

Eight side-swapped team pairs per opponent were played on the local server.
All 64 games completed without action adjustments, timeouts, or protocol
errors. The checkpoint scored 15/16 vs random, 4/16 vs max-damage, 4/16 vs
counterplay, and 15/16 vs the prior policy. It remains experimental and was
not published to the LAN service because worst-case performance is still
poor. Run details and exact commands are in `docs/PHASE11_EVALUATION.md`.

## Expanded evaluator corpus with mixed-opponent states (2026-10-19)

The safe-record assembler merged 94 rollout sources into 10,822 player-view
records from 5,411 matches with no exact duplicate rows. This corpus adds the
new mixed-opponent PPO trajectories while retaining per-step species-only
opponent roster snapshots. The roster-disjoint split contained 51,626 train,
10,477 calibration, and 21,833 final held-out states. The 1,331-input model
selected epoch 1 and scored held-out Brier 0.1635, log loss 0.4922, accuracy
0.7500, and ECE 0.0211. Opening states (2,987 examples) were harder: Brier
0.2293 and ECE 0.0407. These metrics use the expanded test roster split and
are not a controlled comparison to the prior corpus. The corpus is still
orders of magnitude below the planned one-million-game scale, so this does
not establish convergence. Checkpoint:
`outputs/phase12_headless/state_value_v8_robust_training_expansion/evaluator_1331/best.pt`.

## 4,800-game balanced continuation and evaluator refresh (2026-10-19)

Ten balanced-loss PPO rounds completed 4,800 mixed-opponent games at 14.66
games/s including updates (25.47 rollout games/s). The candidate reached
17.2% against max-damage and 18.8% against counterplay in 32 paired,
side-swapped matches each, up from 10.9% and 14.1% for the earlier
1,920-game checkpoint. It remains unsuitable for LAN promotion.

The safe-record assembler merged the v8 corpus with these new trajectories
into `outputs/phase12_headless/state_value_v9_4800_more_mixed/corpus.jsonl`:
20,422 player-view records from 10,211 matches, 140 source files, and no
exact duplicate rows. The corpus maintains the per-decision public opponent
roster snapshots. The roster-disjoint split retained 96,191 training, 19,515
calibration, and 40,819 held-out states. The 1,331-input evaluator selected
epoch 1 and scored held-out Brier 0.1542, log loss 0.4665, accuracy 0.7660,
and ECE 0.0150. Opening states (5,663 examples) scored Brier 0.2246 and ECE
0.0366. These are useful predictive metrics, but not convergence evidence:
there are 10,211 matches in this corpus, not one million, and states within a
match are correlated. Checkpoint:
`outputs/phase12_headless/state_value_v9_4800_more_mixed/evaluator_1331/best.pt`.

Calibration bins make the probability scale easier to interpret: predictions
in 0.0–0.1 averaged 5.3% and outcomes won 5.0%; 0.5–0.6 averaged 54.9% and
won 53.1%; 0.6–0.7 averaged 64.9% and won 60.3%; 0.7–0.8 averaged 75.0%
and won 71.7%; 0.8–0.9 averaged 85.5% and won 82.4%; and 0.9–1.0 averaged
94.3% and won 94.7%. ECE summarizes these gaps as 1.5 percentage points,
though the mid-high bins suggest some overconfidence. The 76.6% thresholded
accuracy is not a per-state probability error measure. A 1,000-replicate
percentile bootstrap resampled the 3,678 held-out match clusters and gave
95% intervals of 0.1490–0.1602 for Brier, 0.4527–0.4822 for log loss,
0.7561–0.7750 for thresholded accuracy, and 0.0115–0.0237 for ECE. The
predicted-minus-observed intervals for the 0.6–0.7 and 0.8–0.9 bins were
0.0224–0.0680 and 0.0105–0.0538, respectively, indicating overconfidence in
those ranges on this evaluation distribution. This estimates uncertainty in
held-out aggregate performance; it does not give a confidence interval for a
single live game's probability. Reproduce with:

```bash
PYTHONPATH=. venv/bin/python scripts/bootstrap_state_value_calibration.py \
  --trajectory-path outputs/phase12_headless/state_value_v9_4800_more_mixed/corpus.jsonl \
  --checkpoint outputs/phase12_headless/state_value_v9_4800_more_mixed/evaluator_1331/best.pt \
  --output outputs/phase12_headless/state_value_v9_4800_more_mixed/evaluator_1331/match_bootstrap_calibration.json \
  --device cuda --batch-size 512 --replicates 1000 --seed 20261019
```

## Second balanced extension and evaluator v10 (2026-10-19)

The second 4,800-game continuation added 4,800 matches / 45,624 transitions.
The merged corpus now has 30,022 player-view records from 15,011 matches,
135 sources, and no exact duplicates. Its roster-disjoint split retained
140,023 train, 28,108 calibration, and 59,429 held-out states. The new
1,331-input evaluator selected epoch 1 and scored Brier 0.1507, log loss
0.4561, accuracy 0.7692, and ECE 0.0112 on the test split. Opening states
(8,320 examples) scored Brier 0.2235 and ECE 0.0463. The overall values are
better than v9's numbers, but the expanded corpus changed the roster split,
so this is not a controlled head-to-head comparison. Opening ECE is worse;
do not treat the aggregate calibration metric as sufficient evidence. A
1,000-replicate match-cluster bootstrap over 5,423 test matches gave 95%
intervals of 0.1464–0.1550 for Brier, 0.4446–0.4674 for log loss,
0.7616–0.7767 for thresholded accuracy, and 0.0086–0.0172 for ECE. The
opening-state intervals were Brier 0.2202–0.2266, log loss 0.6246–0.6393,
accuracy 0.5808–0.6019, and ECE 0.0383–0.0567. The opening calibration gap
is larger than match-level sampling noise on this held-out roster set. The
turns 2–3 and turn 4+ intervals are in the bootstrap JSON artifact.
Checkpoint:
`outputs/phase12_headless/state_value_v10_second_4800_more_mixed/evaluator_1331/best.pt`.
Bootstrap artifact:
`outputs/phase12_headless/state_value_v10_second_4800_more_mixed/evaluator_1331/match_bootstrap_calibration.json`.
Reproduce the v10 training run with:

```bash
PYTHONPATH=. venv/bin/python scripts/train_state_value.py \
  --trajectory-path outputs/phase12_headless/state_value_v10_second_4800_more_mixed/corpus.jsonl \
  --output-dir outputs/phase12_headless/state_value_v10_second_4800_more_mixed/evaluator_1331 \
  --epochs 25 --patience 5 --batch-size 512 --lr 0.0003 \
  --test-fraction 0.2 --calibration-fraction 0.15 --seed 20261019 \
  --device cuda --embedding-dim 64 --transformer-layers 2 \
  --attention-heads 4 --ff-dim 128 --field-feature-dim 1331 \
  --dropout 0.1 --weight-decay 0.01 --turn-stratified-calibration
```

The evaluator trainer and match-bootstrap utility now consume player-view
JSONL records as an iterator, instead of keeping all parsed records in memory
beside the built feature dataset. On the v9 bootstrap workload, this reduced
observed RSS from over 20,752,564 KiB in the prior run to a measured peak of
4,382,672 KiB (~4.18 GiB) in the streaming run, which took 178.77 seconds and
reproduced all prior point metrics exactly. The v10 trainer's sampled RSS
reached about 6.2 GB while processing the larger corpus, leaving the local
host with substantial memory headroom for the next corpus expansion.

To train directly from completed PPO runs, `scripts/train_state_value.py`
accepts repeated `--rounds-root` arguments. It reads each run's completed
`rounds.jsonl` manifest and consumes the referenced rollout shards directly,
without first assembling a duplicate merged corpus. Build the disk-sharded
feature cache after the rollout run is complete:

```bash
PYTHONPATH=. venv/bin/python scripts/train_state_value.py \
  --rounds-root outputs/phase5/your_completed_run \
  --disk-backed-cache outputs/phase4/your_run/state_value_cache \
  --prepare-cache-only --device cuda
```

Then rerun with the same arguments, omit `--prepare-cache-only`, and set
`--output-dir` to the evaluator checkpoint directory. The cache manifest
fingerprints all source shard paths, sizes, modification times, split settings,
and the feature schema, so changed or incomplete source manifests cannot reuse
an incompatible cache.

To estimate held-out uncertainty without materializing the full state dataset
in RAM, bootstrap match clusters from that cache. Cache schema 3 stores the
match ID next to each example inside per-shard sidecars, preserving exact row
alignment:

```bash
PYTHONPATH=. venv/bin/python scripts/bootstrap_state_value_calibration.py \
  --disk-backed-cache outputs/phase4/your_run/state_value_cache \
  --checkpoint outputs/phase4/your_run/evaluator/best.pt \
  --output outputs/phase4/your_run/evaluator/match_bootstrap_calibration.json \
  --replicates 1000 --seed 20261019
```

The bootstrap performs inference in shard-local batches. It verifies that the
cached split settings and held-out roster hashes match the evaluator
checkpoint, and that every test tensor row has exactly one match ID.

## Corrected large-corpus evaluator and match-cluster uncertainty (2026-10-19)

While validating the cache-backed bootstrap, a shard-boundary bug was found in
the disk-cache builder: after flushing a full shard mid-record, the remaining
examples in that record were appended to a detached buffer and silently lost.
The builder now refreshes its buffer after every flush. Cache schema 3 stores
one match ID for every cached example, so cluster resampling uses the same rows
that model inference scores.

The corrected cache was rebuilt from the 1,394 rollout shards. Its 670,820
states split into 408,970 train, 85,090 calibration, and 176,760 held-out
states. Sidecar ID counts exactly match tensor counts in each split. The fresh
1,331-input evaluator selected epoch 4. Its held-out metrics are Brier 0.14302,
log loss 0.43297, accuracy 0.77962, and ECE 0.00538. Opening states (26,823)
have Brier 0.22428 and ECE 0.02314; turns 2–3 have Brier 0.15742, and turn 4+
has Brier 0.11063.

A 1,000-replicate bootstrap over 17,335 whole-match clusters produced 95%
intervals of 0.14065–0.14535 for Brier, 0.42655–0.43923 for log loss,
0.77577–0.78379 for accuracy, and 0.00379–0.00845 for ECE. Opening Brier's
interval is 0.22280–0.22593 and opening ECE's is 0.01801–0.02909. The bootstrap
point metrics match the saved held-out report exactly. These intervals measure
uncertainty across held-out matches, not uncertainty for a single live state's
probability. The data include the 48,000 games used in PPO training, so the
roster-disjoint evaluator test is not an independent post-training-policy or
convergence assessment.

Artifacts are under
`outputs/phase4/state_value_fourth_balanced_48000_20261019/`; the corrected
checkpoint is `evaluator_corrected/best.pt`, and bootstrap output is
`evaluator_corrected/match_bootstrap_calibration.json`.

## Reusing Showdown evaluation rounds for evaluator training

Paired local-server evaluations can now optionally save each completed match
as two schema-v2 player-view JSONL records, with only that player's full team
and the opponent's public preview roster. Each decision row includes the
player-visible observation, action mask, action, public mechanics reconstructed
from Showdown's replay events, and terminal win/loss label. Enable collection
with `--player-view-output-dir` on `scripts/evaluate_native_paired.py`. Then
pass the resulting files to `scripts/train_state_value.py` using repeated
`--trajectory-path` arguments. The trainer splits by team before fitting and
calibration, but the rounds used to fit it are no longer an independent final
test; run a new unseen evaluation after training.

The 256-pair evaluation completed earlier in this project did not enable this
option. Its files contain only result, step count, and action-adjustment
summary rows, so those particular matches cannot be reused for state-value
training. The new collection path applies to future evaluations.

## PPO-generation holdout and opponent-family weighting audit (2026-10-31)

Round490 fit five action-value models to Max Damage, Random, legacy Counterplay,
Public Counterplay, Round476 PPO, and Round430 PPO. Round102 PPO was included in
the branch outcomes but held out of model fitting. The capture covered 375 states
across decisions 1, 3, and 5 and 134,655 branch matches (62.20 matches/s). The
internal five-split maximin result was +2.13 pp [0.05, 4.30] on 258 held-out team
clusters, but the Round102 holdout result was +0.81 pp [-1.78, 3.44].

Round491 evaluated the frozen models on fresh random seeds, with 371 held-out
team clusters and seven response types. The maximin action selector's paired
worst-response change was -1.31 pp [-2.39, -0.28]; equal-weight choice was
-1.23 pp [-2.31, -0.20]. The selector lost to legacy Counterplay by 2.13 pp and
to the two fitted PPO checkpoints by 1.95 and 1.79 pp. The unseen Round102 PPO
delta was -0.79 pp, with uncertainty spanning zero. Round430 remains the live
policy; neither action selector is promoted.

To see whether multiple PPO checkpoints were overweighted, we averaged their
predicted values as one PPO family before selecting actions, then scored those
choices on the exact same Round491 branches. Family maximin selected the same
actions as per-checkpoint maximin on all five fits and had a -1.31 pp
worst-response change [-2.43, -0.31]. Family equal-weight was -1.22 pp
[-2.34, -0.22], essentially unchanged. These results do not support PPO
checkpoint multiplicity as the cause of selector regression; the more direct
issue is that predicted action rankings do not transfer to observed outcomes.
The next work should inspect ranking/calibration by candidate and response,
hidden-world coverage, and whether the candidate set omits valuable actions.

Artifacts: `outputs/phase12_headless/round490_ppo_generation_holdout_128x8_20261031/`
and `outputs/phase12_headless/round491_external_ppo_generation_eval_128x8_20261031/`.

## Pairwise action-value pilot and fresh confirmation (2026-10-31)

Round491 also diagnosed within-state action ranking. Across five independent
model fits, the response-conditioned win-probability model ranked worst-response
action pairs correctly 56.7% of the time. Using four of eight chance repeats to
select the best empirical maximin action and the other four to score it gained
3.54 pp [2.16, 5.05] over the incumbent. This uses outcomes directly and is an
oracle-style headroom estimate, not a legal live policy.

Round492 trained five response-wise models on Round490, combining absolute
outcome BCE with pairwise ranking loss; Round102 PPO was held out. On the same
Round491 branches, maximin choice scored -0.69 pp [-1.84, 0.37] against the
incumbent. A matched BCE-only ablation scored -1.21 pp [-2.37, 0.02]. The
pairwise model's paired improvement over BCE-only was +0.52 pp [0.03, 1.01],
and over the prior value selector was +0.62 pp [0.05, 1.21]. Worst-response
pairwise accuracy increased from 56.7% to 58.4%. The gain is small and the
selector still does not reliably beat the incumbent, so Round430 remains the
active policy.

Round493 is a pre-registered-style fresh confirmation using new seeds, 128
states at each of decision indices 1, 3, and 5, eight chance repeats, and the
same seven response types. Its opening phase captured all 128 states with no
missing states, executing 46,120 branch matches at 51.5 wall-clock matches/s.
The remaining phases are still collecting. The pairwise model and BCE-only
control will be compared on these same new states before deciding whether to
advance the action-value approach.

Artifacts:
`outputs/phase12_headless/round491_external_ppo_generation_eval_128x8_20261031/action_ranking_diagnostics.json`,
`outputs/phase12_headless/round492_pairwise_response_ranker_20261031/`,
`outputs/phase12_headless/round492_bce_ablation_20261031/`, and
`outputs/phase12_headless/round493_pairwise_confirmation_128x8_20261031/`.
