#!/usr/bin/env python3
# coding: utf-8
"""Q3 independent transfer ablation: unconstrained Gaussian OT vs audited constrained variants."""
import json,argparse
from pathlib import Path
import numpy as np,pandas as pd
from q3_complete import Transformer,StyleGaussian,fit_sensory,ot_matrix,SEED

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--results',required=True);ap.add_argument('--q1-reference',required=True);args=ap.parse_args()
 root=Path(args.data_dir);out=Path(args.results)
 df=pd.read_csv(next(iter(sorted(root.glob('1*.csv'))))); yf=pd.read_csv(next(iter(sorted(root.glob('4*.csv')))))
 name=list(df.columns[3:]);targets=list(yf.columns[3:]); X=df[name].to_numpy(float);styles=df.tasting_category_fine.to_numpy(str)
 ref=pd.read_csv(args.q1_reference,encoding='utf-8-sig')[targets].to_numpy(float)
 surrogate=fit_sensory(X,ref); tr=Transformer(12).fit(X); z=tr.encode(X)
 g=StyleGaussian().fit(z,styles,sorted(set(styles)))
 sources=np.flatnonzero(styles=='Pils/Lager'); raw=X[sources].copy();src=tr.imp.transform(raw)
 j4=[name.index(v) for v in ['DMS','X2.3.butanedione','X4VG','X.E..2.nonenal']]
 for j in j4:src[np.isnan(raw[:,j]),j]=tr.low[j]
 zsrc=tr.encode(src); mean=g.means['Tripel']
 matrix=ot_matrix(g.covs['Pils/Lager'],g.covs['Tripel'])
 proposals=tr.decode(mean+(zsrc-g.means['Pils/Lager'])@matrix.T)
 direct=proposals.copy();direct[:,j4]=np.minimum(src[:,j4],direct[:,j4])
 prior=surrogate(src)[:,targets.index('stale_hops')]
 results=[]
 for meth,data in [('unconstrained_OT',proposals),('4_compound_projected_OT',direct),('all5_constrained_OT',pd.read_csv(out/'pils_to_tripel_transferred_compounds_11.csv')[name].to_numpy(float)),('all5_sample_optimized',pd.read_csv(out/'pils_to_tripel_optimized_chemical231.csv')[name].to_numpy(float))]:
  pred=surrogate(data)[:,targets.index('stale_hops')]
  measured=np.isfinite(raw[:,j4]);violation=(data[:,j4]>raw[:,j4]+1e-10)&measured
  before=np.linalg.norm(zsrc-mean,axis=1);after=np.linalg.norm(tr.encode(data)-mean,axis=1)
  results.append({'method':meth,'mean_distance_to_tripel':float(after.mean()),'n_closer':int(np.sum(after<before-1e-4)),
      'observed_compound_violations':int(violation.sum()),'observed_comparisons':int(measured.sum()),
      'stale_hops_proxy_violations':int(np.sum(pred>prior+1e-10)),
      'source_missing_defects':int((~measured).sum()),'mean_stale_proxy_delta':float(np.mean(pred-prior))})
 pd.DataFrame(results).to_csv(out/'transfer_ablation.csv',index=False)
 print(pd.DataFrame(results).to_string(index=False))
if __name__=='__main__':main()
