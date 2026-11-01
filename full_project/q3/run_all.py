#!/usr/bin/env python3
# coding:utf-8
"""Q3 single command reproduction after team-provided raw Q1 reference labels."""
import argparse,subprocess,sys,os
from pathlib import Path
base=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--data-dir',required=True);ap.add_argument('--q1-reference',required=True)
ap.add_argument('--out-dir',default=str(base/'results'));ap.add_argument('--epochs',type=int,default=200)
a=ap.parse_args()
env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
common=['--data-dir',a.data_dir]
steps=[('q3_complete.py',common+['--q1-reference',a.q1_reference,'--out-dir',a.out_dir,'--epochs',str(a.epochs),'--generated-per-style','40']),
('q3_censor_finalize.py',common+['--q1-reference',a.q1_reference,'--results',a.out_dir]),
('q3_censor_repeated.py',common+['--results',a.out_dir]),
('q3_transfer_optimize.py',common+['--q1-reference',a.q1_reference,'--results',a.out_dir]),
('q3_transfer_ablation.py',common+['--q1-reference',a.q1_reference,'--results',a.out_dir]),
('verify_q3.py',common+['--results',a.out_dir])]
for file,args in steps:
 print('RUN',file,flush=True);subprocess.run([sys.executable,str(base/file),*args],env=env,check=True)
subprocess.run([sys.executable,str(base/'plot_results.py')],env=env,check=True)
print('Q3 ALL DONE')
