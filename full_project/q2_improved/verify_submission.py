#!/usr/bin/env python3
# AI-assisted (OpenAI GPT-6, 2026-10-27). Must be reviewed by competitors.
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from q2_run import load_inputs,eligible_chemicals
ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--out-dir',default='results');args=ap.parse_args()
X,Y,names,targs,style,fold,*_=load_inputs(args.data_dir,str(Path(__file__).resolve().parent/'split_assignments.csv'))
o=Path(args.out_dir);res=json.loads((o/'summary.json').read_text('utf8'))
top=pd.read_csv(o/'q2_final_top3_150.csv');assert top.shape[0]==150
assert set(top.sensory)==set(targs)
for _,group in top.groupby('sensory'):
    assert sorted(group['rank'].tolist())==[1,2,3] and group.chemical.nunique()==3
assert set(top.chemical).issubset(set(np.array(names)[eligible_chemicals(names)]))
assert top.bootstrap_top3_frequency.between(0,1).all()
p0=pd.read_csv(o/'oof_spearman.csv')[targs].to_numpy(float)
p1=pd.read_csv(o/'oof_hybrid.csv')[targs].to_numpy(float)
assert p0.shape==(250,50) and p1.shape==(250,50)
for name,pred in [('mae_spearman',p0),('mae_conditional',p1)]:
   found=abs(pred-Y).mean(); claimed=res['fixed_original_5fold'][name]
   assert abs(found-claimed)<1e-10,(name,found,claimed)
assert len(pd.read_csv(o/'signed_network.csv'))==res['graph']['n_edges']
assert len(pd.read_csv(o/'stability_50.csv'))==50
print('PASS: 250 samples, 50 targets, 150 molecules, score recomputation, boot frequencies and graph edges')
