#!/usr/bin/env python3
"""Experimental paired action-advantage ranker for counterfactual data.

Learns candidate action value relative to the incumbent action in the same
player-visible state. Teams and chance repeats are split independently.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.train_action_conditioned_win_model import _inputs


class PairedAdvantage(nn.Module):
    def __init__(self, width: int, responses: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width * 3, 256), nn.ReLU(), nn.Dropout(.10),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.05),
            nn.Linear(128, responses),
        )

    def forward(self, candidate, incumbent):
        score = self.net(torch.cat((candidate, incumbent, candidate - incumbent), dim=-1))
        baseline_score = self.net(torch.cat((incumbent, incumbent,
                                             torch.zeros_like(incumbent)), dim=-1))
        # The incumbent's predicted advantage must be exactly zero. Without
        # anchoring, a small common prediction bias can make an unproven
        # alternative beat the policy action in every state.
        return score - baseline_score


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=202611569)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    ap.add_argument('--test-cluster-min', type=int, default=None,
                    help='reserve every cluster at or above this ID for a fresh test set')
    args = ap.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA unavailable')
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed); device = torch.device(args.device)
    payload = json.loads(args.data.read_text())
    raw = payload['rows']; responses = sorted(raw[0]['outcomes_by_opponent'])
    action_width = len(raw[0]['legal_action_mask']) // 2
    feature_dim = len(_inputs(raw[0]['observation'], raw[0]['action'], action_width,
        raw[0]['native_action_features'], raw[0]['public_context_features'],
        raw[0].get('policy_log_probability'), True))
    grouped = {}
    for row in raw:
        key = (int(row['team_cluster']), int(row['state_repeat']))
        actions = grouped.setdefault(key, {})
        action = tuple(map(int, row['action']))
        old = actions.get(action)
        if old is None:
            actions[action] = {**row, 'candidate_names': {str(row['candidate'])}}
        else:
            for f in ('observation','native_action_features','public_context_features','outcomes_by_opponent'):
                if old[f] != row[f]: raise ValueError(f'inconsistent duplicate {key} {action} {f}')
            old['candidate_names'].add(str(row['candidate']))
    states = []
    for key, action_map in grouped.items():
        candidates = list(action_map.values())
        baseline = [r for r in candidates if 'policy' in r['candidate_names']]
        if len(baseline) != 1: raise ValueError(f'{key}: expected one incumbent action')
        base = baseline[0]
        bfeat = _inputs(base['observation'], base['action'], action_width,
            base['native_action_features'], base['public_context_features'], base.get('policy_log_probability'), True)
        entries=[]
        for r in candidates:
            feat = _inputs(r['observation'], r['action'], action_width,
                r['native_action_features'], r['public_context_features'], r.get('policy_log_probability'), True)
            ys=[]
            for name in responses:
                a=np.asarray(r['outcomes_by_opponent'][name],dtype=np.float32)
                b=np.asarray(base['outcomes_by_opponent'][name],dtype=np.float32)
                if len(a)!=8 or len(b)!=8: raise ValueError('expected 8 chance repeats')
                ys.append([float(np.mean(a[:4]-b[:4])), float(np.mean(a[4:]-b[4:]))])
            entries.append({'row':r,'x':feat,'y':ys})
        states.append({'key':key,'rows':entries,'base_action':tuple(map(int,base['action']))})
    clusters=np.asarray(sorted({s['key'][0] for s in states}))
    if args.test_cluster_min is None:
        split=rng.permutation(clusters); nt=max(1,round(.2*len(split)))
        test=set(map(int,split[:nt])); remaining=split[nt:]
    else:
        test=set(map(int,clusters[clusters >= args.test_cluster_min]))
        remaining=rng.permutation(clusters[clusters < args.test_cluster_min])
        if not test or len(remaining) < 2:
            ap.error('test-cluster-min must leave both fresh test and training clusters')
    nv=max(1,round(.2*len(remaining))); val=set(map(int,remaining[:nv])); train=set(map(int,remaining[nv:]))
    train_rows=[]; val_rows=[]
    for state in states:
        dest=train_rows if state['key'][0] in train else val_rows if state['key'][0] in val else None
        if dest is None: continue
        base=next(e for e in state['rows'] if tuple(map(int,e['row']['action']))==state['base_action'])
        for e in state['rows']:
            dest.append((e['x'],base['x'],np.asarray(e['y'])[:,0]))
    def tensors(rows):
        return tuple(torch.tensor(np.stack([r[i] for r in rows]),dtype=torch.float32,device=device) for i in range(3))
    tx, tb, ty=tensors(train_rows); vx,vb,vy=tensors(val_rows)
    model=PairedAdvantage(feature_dim,len(responses)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=2e-3)
    best=float('inf'); best_state=None; stale=0; log=[]
    for epoch in range(1,args.epochs+1):
        model.train(); order=rng.permutation(len(tx))
        for start in range(0,len(order),256):
            ix=torch.as_tensor(order[start:start+256],device=device)
            pred=model(tx[ix],tb[ix]); loss=nn.functional.smooth_l1_loss(pred,ty[ix],beta=.25)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval()
        with torch.no_grad(): val_loss=float(nn.functional.smooth_l1_loss(model(vx,vb),vy,beta=.25).item())
        log.append({'epoch':epoch,'train_loss':float(loss.item()),'validation_loss':val_loss})
        if val_loss<best-1e-5:
            best=val_loss; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=args.patience: break
    model.load_state_dict(best_state); model.eval()
    team_values={name:{} for name in responses}; worst_deltas={}; chosen_counts={}
    with torch.no_grad():
        for s in states:
            if s['key'][0] not in test: continue
            base=next(e for e in s['rows'] if tuple(map(int,e['row']['action']))==s['base_action'])
            bx=torch.as_tensor(base['x'][None,:],device=device)
            xs=torch.as_tensor(np.stack([e['x'] for e in s['rows']]),device=device)
            pred=model(xs,bx.expand(len(xs),-1)).cpu().numpy()
            selected=int(np.argmax(np.min(pred,axis=1)))
            chosen=s['rows'][selected]
            key=str(s['key'][0]); chosen_name=','.join(sorted(chosen['row']['candidate_names']))
            chosen_counts[chosen_name]=chosen_counts.get(chosen_name,0)+1
            actual=np.asarray(chosen['y'])[:,1]
            # Candidate label is already paired against incumbent for each response.
            worst_deltas.setdefault(key,[]).append(float(np.min(actual)))
            for j,name in enumerate(responses): team_values[name].setdefault(key,[]).append(float(actual[j]))
    def boot(vals,seed):
        x=np.asarray([np.mean(v) for v in vals.values()]); rg=np.random.default_rng(seed)
        b=x[rg.integers(0,len(x),(20000,len(x)))].mean(axis=1)
        return [float(q) for q in np.quantile(b,[.025,.975])]
    aggregate={name:float(np.mean([np.mean(v) for v in d.values()])) for name,d in team_values.items()}
    worst={k:float(np.mean(v)) for k,v in worst_deltas.items()}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    ckpt=args.output_dir/'paired_action_advantage.pt'
    torch.save({'state_dict':model.state_dict(),'feature_dim':feature_dim,'responses':responses,'seed':args.seed},ckpt)
    report={'data':str(args.data),'checkpoint':str(ckpt),'responses':responses,'team_split':{'train':len(train),'validation':len(val),'test':len(test)},'best_epoch':len(log)-stale,'best_validation_loss':best,'epochs':log,'test_selected_candidate_counts':chosen_counts,'test_selected_paired_delta_by_response':aggregate,'test_selected_statewise_worst_response_delta':float(np.mean(list(worst.values()))),'test_worst_response_cluster_bootstrap_95_ci':boot(worst_deltas,args.seed),'information_boundary':'player-visible observation, candidate action features, and incumbent action features'}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
