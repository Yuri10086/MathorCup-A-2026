#!/usr/bin/env python3
"""Validate Q2 generated artifacts; GPT-6 assistance 2026-10-27."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from q2_run import load_inputs,eligible_chemicals
R=Path(__file__).resolve().parent/'results'
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='/mnt/data');ap.add_argument('--split-csv',default=str(R.parent/'split_assignments.csv'))
args=ap.parse_args()
X,Y,C,T,style,fold,rho,Rnames,audit=load_inputs(args.data_dir,args.split_csv)
top=pd.read_csv(R/'q2_top3_50x3.csv');bs=np.load(R/'q2_bootstrap_top3_indices.npz')['top3_indices']
assert set(top.sensory)==set(T) and len(top)==150
assert all(sorted(g['rank'])==[1,2,3] and g.chemical.nunique()==3 for _,g in top.groupby('sensory'))
assert top.chemical.isin(set(np.array(C)[eligible_chemicals(C)])).all()
assert top.bootstrap_top3_frequency.between(0,1).all()
assert bs.shape==(100,50,3) and int(bs.max())<231 and int(bs.min())>=0
assert len(pd.read_csv(R/'q2_signed_edges.csv'))==951
assert len(pd.read_csv(R/'q2_chemical_name_map.csv'))==231
assert len(pd.read_csv(R/'q2_top3_50x3_with_chemical_meta.csv'))==150
for n in ['fusion','spearman','random','ridgecoef','all231_ridge','all231_extratrees']:
 d=pd.read_csv(R/f'q2_oof_{n}.csv'); assert d.shape==(250,51) and np.isfinite(d[T].to_numpy(float)).all()
for n in ['true_signed','no_graph','shuffled_signed']:
 d=pd.read_csv(R/f'q2_oof_deep_{n}.csv'); assert d.shape==(250,51)
extra=pd.read_csv(R/'q2_chemical_anchor_audit.csv');assert len(extra)==12
assert (pd.read_csv(R/'q2_stability_vs_spearman.csv').shape==(50,5))
print('PASS: 250 beer, 231 chemicals, 50 targets; 150 molecular Top3 rows; 100 bootstrap ranks; signed edges; neural OOF; metadata')
