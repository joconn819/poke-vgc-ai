#!/usr/bin/env python3
"""Train a shared action encoder with within-state paired ranking targets."""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


class StructuredRanker(nn.Module):
    def __init__(self, observation_dim: int, context_dim: int,
                 action_feature_dim: int, response_count: int, action_width: int):
        super().__init__()
        self.action_width = action_width
        self.slot_embedding = nn.Embedding(2, 8)
        self.action_embedding = nn.Embedding(action_width, 12)
        self.action_encoder = nn.Sequential(
            nn.Linear(12 + action_feature_dim + 8, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(observation_dim + context_dim, 96), nn.ReLU(),
            nn.Linear(96, 64), nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.ReLU(), nn.Linear(64, response_count),
        )

    def score(self, observation, context, actions, action_features):
        # actions [batch, 2], action_features [batch, 2, feature_dim]
        slots = torch.arange(2, device=actions.device).view(1, 2).expand_as(actions)
        action_tokens = torch.cat((
            self.action_embedding(actions), action_features,
            self.slot_embedding(slots),
        ), dim=-1)
        encoded = self.action_encoder(action_tokens).flatten(1)
        state = self.state_encoder(torch.cat((observation, context), dim=-1))
        return self.value(torch.cat((state, encoded), dim=-1))

    def forward(self, observation, context, candidates, base_actions,
                candidate_features, base_features):
        candidate_score = self.score(observation, context, candidates, candidate_features)
        base_score = self.score(observation, context, base_actions, base_features)
        return candidate_score - base_score


def _cluster_ci(values: dict[int, float], seed: int, draws: int = 20000):
    row = np.asarray(list(values.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = row[rng.integers(0, len(row), (draws, len(row)))].mean(axis=1)
    return [float(x) for x in np.quantile(boot, [.025, .975])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=202611573)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    ap.add_argument('--test-cluster-min', type=int, default=None)
    args=ap.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available(): ap.error('CUDA unavailable')
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed); device=torch.device(args.device)
    payload=json.loads(args.data.read_text()); raw=payload['rows']
    responses=sorted(raw[0]['outcomes_by_opponent']); width=len(raw[0]['legal_action_mask'])//2
    obs_dim=len(raw[0]['observation']); context_dim=len(raw[0]['public_context_features'])
    feature_dim=len(raw[0]['native_action_features'][0][0])
    state_map=defaultdict(dict)
    for row in raw:
        key=(int(row['team_cluster']),int(row['state_repeat'])); action=tuple(map(int,row['action']))
        prior=state_map[key].get(action)
        if prior is None:
            state_map[key][action]={**row,'names':{str(row['candidate'])}}
        else:
            for field in ('observation','public_context_features','native_action_features','outcomes_by_opponent'):
                if prior[field] != row[field]: raise ValueError(f'inconsistent duplicate {key}/{action}/{field}')
            prior['names'].add(str(row['candidate']))
    states=[]
    for key, action_map in state_map.items():
        rows=list(action_map.values()); base=[r for r in rows if 'policy' in r['names']]
        if len(base)!=1: raise ValueError(f'{key}: expected one incumbent action')
        states.append({'key':key,'rows':rows,'base':base[0]})
    cluster_ids=np.asarray(sorted({s['key'][0] for s in states}))
    if args.test_cluster_min is None:
        split=rng.permutation(cluster_ids); n=max(1,round(.2*len(split)))
        test=set(map(int,split[:n])); remaining=split[n:]
    else:
        test=set(map(int,cluster_ids[cluster_ids>=args.test_cluster_min]))
        remaining=rng.permutation(cluster_ids[cluster_ids<args.test_cluster_min])
        if not test or len(remaining)<2: ap.error('invalid test cluster threshold')
    nv=max(1,round(.2*len(remaining))); validation=set(map(int,remaining[:nv])); train=set(map(int,remaining[nv:]))

    def state_tensors(s, half):
        rows=s['rows']; base=s['base']
        actions=np.asarray([r['action'] for r in rows],dtype=np.int64)
        action_features=np.asarray([[r['native_action_features'][slot][int(r['action'][slot])] for slot in range(2)] for r in rows],dtype=np.float32)
        labels=np.asarray([[np.mean(np.asarray(r['outcomes_by_opponent'][name][:4],dtype=np.float32)) for name in responses] for r in rows],dtype=np.float32)
        base_action=np.asarray([base['action']],dtype=np.int64)
        base_features=np.asarray([[base['native_action_features'][slot][int(base['action'][slot])] for slot in range(2)]],dtype=np.float32)
        base_position=next(i for i,r in enumerate(rows) if r is base)
        test_labels=np.asarray([[np.mean(np.asarray(r['outcomes_by_opponent'][name][4:],dtype=np.float32)) for name in responses] for r in rows],dtype=np.float32)
        return {
            'obs':np.asarray(s['rows'][0]['observation'],dtype=np.float32),
            'context':np.asarray(s['rows'][0]['public_context_features'],dtype=np.float32),
            'actions':actions,'features':action_features,'labels':labels,
            'base_action':base_action,'base_features':base_features,
            'base_position':base_position,'test_labels':test_labels,
            'rows':rows,
        }
    tensor_states={s['key']:state_tensors(s,0) for s in states}
    train_states=[tensor_states[s['key']] for s in states if s['key'][0] in train]
    val_states=[tensor_states[s['key']] for s in states if s['key'][0] in validation]
    model=StructuredRanker(obs_dim,context_dim,feature_dim,len(responses),width).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=5e-3)

    def batch(state):
        def t(x,dtype=torch.float32): return torch.as_tensor(x,dtype=dtype,device=device)
        return (t(state['obs'][None,:]),t(state['context'][None,:]),
                t(state['actions'],torch.long),t(state['base_action'],torch.long).expand(len(state['rows']),-1),
                t(state['features']),t(state['base_features']).expand(len(state['rows']),-1,-1),
                t(state['labels']))

    def val_loss():
        model.eval(); values=[]
        with torch.no_grad():
            for s in val_states:
                obs,ctx,actions,base,features,bfeat,labels=batch(s)
                pred=model(obs.expand(len(actions),-1),ctx.expand(len(actions),-1),actions,base,features,bfeat)
                diff=pred[:,None,:]-pred[None,:,:]
                target=labels[:,None,:]-labels[None,:,:]
                mask=torch.triu(torch.ones(diff.shape[:2],device=device,dtype=torch.bool),diagonal=1)
                values.append(nn.functional.smooth_l1_loss(diff[mask],target[mask],beta=.15).item())
        return float(np.mean(values))

    best=float('inf'); best_state=None; stale=0; logs=[]
    for epoch in range(1,args.epochs+1):
        model.train(); order=rng.permutation(len(train_states)); epoch_losses=[]
        for idx in order:
            s=train_states[int(idx)]; obs,ctx,actions,base,features,bfeat,labels=batch(s)
            pred=model(obs.expand(len(actions),-1),ctx.expand(len(actions),-1),actions,base,features,bfeat)
            diff=pred[:,None,:]-pred[None,:,:]; target=labels[:,None,:]-labels[None,:,:]
            mask=torch.triu(torch.ones(diff.shape[:2],device=device,dtype=torch.bool),diagonal=1)
            loss=nn.functional.smooth_l1_loss(diff[mask],target[mask],beta=.15)
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            epoch_losses.append(float(loss.item()))
        v=val_loss(); logs.append({'epoch':epoch,'train_pairwise_loss':float(np.mean(epoch_losses)),'validation_pairwise_loss':v})
        if v<best-1e-6: best=v; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=args.patience: break
    model.load_state_dict(best_state); model.eval()
    team_worst={}; per_response={n:{} for n in responses}; changed=0; counts=defaultdict(int)
    with torch.no_grad():
        for s in states:
            if s['key'][0] not in test: continue
            z=tensor_states[s['key']]; obs,ctx,actions,base,features,bfeat,labels=batch(z)
            pred=model(obs.expand(len(actions),-1),ctx.expand(len(actions),-1),actions,base,features,bfeat).cpu().numpy()
            pick=int(np.argmax(np.min(pred,axis=1))); bp=z['base_position']; chosen=z['rows'][pick]
            delta=z['test_labels'][pick]-z['test_labels'][bp]
            changed+=int(pick!=bp)
            name=','.join(sorted(chosen['names'])); counts[name]+=1
            c=str(s['key'][0]); team_worst[c]=float(np.min(delta))
            for i,n in enumerate(responses): per_response[n][c]=float(delta[i])
    report={'data':str(args.data),'checkpoint':str(args.output_dir/'structured_action_ranker.pt'),'responses':responses,'team_split':{'train':len(train),'validation':len(validation),'test':len(test)},'fresh_test_clusters':args.test_cluster_min is not None,'best_epoch':len(logs)-stale,'best_validation_pairwise_loss':best,'epochs':logs,'test_selected_action_changes':changed,'test_selected_candidate_counts':dict(counts),'test_paired_delta_by_response':{n:float(np.mean(list(v.values()))) for n,v in per_response.items()},'test_worst_response_delta':float(np.mean(list(team_worst.values()))),'test_worst_response_cluster_bootstrap_95_ci':_cluster_ci(team_worst,args.seed),'information_boundary':'player-visible state/action features; counterfactual outcomes used only as training labels'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'observation_dim':obs_dim,'context_dim':context_dim,'action_feature_dim':feature_dim,'response_names':responses,'action_width':width,'seed':args.seed},args.output_dir/'structured_action_ranker.pt')
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
