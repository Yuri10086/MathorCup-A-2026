#!/usr/bin/env python3
# coding: utf-8
"""Repeated style-stratified holdouts for minimum-pileup calibration.
AI-assisted OpenAI GPT-6 (2026-10-27); replicate independently before submission.
"""
import json,argparse
from pathlib import Path
import numpy as np,pandas as pd
from q3_complete import Transformer,StyleGaussian,SEED
from q3_censor_finalize import fit_minimum_masses,apply_mass

def split(st,seed):
 rg=np.random.default_rng(seed);test=[]
 for s in np.unique(st):
  ids=np.flatnonzero(st==s)
  if len(ids)>=5:test.extend(rg.choice(ids,max(1,round(.2*len(ids))),replace=False))
 te=np.sort(test);tr=np.setdiff1d(np.arange(len(st)),te)
 return tr,te

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='/mnt/data');ap.add_argument('--results',default='/mnt/data/MathorCup_A_Q3_Submission/results');a=ap.parse_args()
 f=pd.read_csv(next(iter(sorted(Path(a.data_dir).glob('1*.csv')))))
 x=f.iloc[:,3:].to_numpy(float);names=f.columns[3:];st=f.tasting_category_fine.to_numpy(str);styles=sorted(np.unique(st))
 records=[]
 for seed in (SEED,77,99):
  tr,te=split(st,seed)
  t=Transformer(12).fit(x[tr]);g=StyleGaussian().fit(t.encode(x[tr]),st[tr],styles)
  floor,base_mask,masses=fit_minimum_masses(x[tr],st[tr],threshold=.25)
  rng=np.random.default_rng(seed+888)
  for s in np.unique(st[te]):
   ids=te[st[te]==s]; real=x[ids]
   generated=t.decode(g.sample(s,300,rng))
   upgraded=apply_mass(generated,np.repeat(s,300),floor,base_mask,masses,rng)
   real_rates=np.mean(np.isclose(real,floor,rtol=0,atol=1e-12),axis=0)
   b_rates=np.mean(np.isclose(generated,floor,rtol=0,atol=1e-12),axis=0)
   u_rates=np.mean(np.isclose(upgraded,floor,rtol=0,atol=1e-12),axis=0)
   records.append({'seed':seed,'style':s,'n_test':len(ids),'n_features':int(base_mask.sum()),
         'baseline_mae':float(np.mean(np.abs(real_rates[base_mask]-b_rates[base_mask]))),
         'aware_mae':float(np.mean(np.abs(real_rates[base_mask]-u_rates[base_mask])) )})
 df=pd.DataFrame(records);out=Path(a.results);out.mkdir(exist_ok=True,parents=True)
 df.to_csv(out/'repeated_holdout_floor_metrics.csv',index=False)
 by=[]
 for seed,g in df.groupby('seed'):
  a0=np.average(g.baseline_mae,weights=g.n_test);a1=np.average(g.aware_mae,weights=g.n_test)
  by.append({'seed':int(seed),'baseline':float(a0),'censor_aware':float(a1),'delta':float(a0-a1),'relative_pct':float(100*(a0-a1)/a0),'feature_count':int(g.n_features.iloc[0])})
 # bootstrap block clusters by style across 3 seeds
 rng=np.random.default_rng(9877)
 v=np.array([np.sum((df.loc[df['style']==s,'baseline_mae']-df.loc[df['style']==s,'aware_mae'])*df.loc[df['style']==s,'n_test']) for s in styles])
 w=np.array([np.sum(df.loc[df['style']==s,'n_test']) for s in styles])
 draw=rng.integers(0,len(styles),size=(3000,len(styles)))
 deltas=v[draw].sum(axis=1)/np.maximum(w[draw].sum(axis=1),1)
 summary={'by_seed':by,'mean_absolute_improvement':float(np.mean([z['delta'] for z in by])),
 'improvements_positive_all_splits':all(z['delta']>0 for z in by),
 'cluster_bootstrap_CI95_on_mean_floor_fraction_MAE_improvement':np.percentile(deltas,[2.5,97.5]).tolist(),
 'definition':'Repeated 3 style-stratified 80/20 holdouts; style resampling for uncertainty; floor threshold 25% fixed',
 'notes':'Same 250 cohort; frequent minima proxy detection floors; not independent lab validation.'}
 (out/'q3_censor_repeated_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
