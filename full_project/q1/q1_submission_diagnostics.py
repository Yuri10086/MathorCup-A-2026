#!/usr/bin/env python3
"""Q1 submission diagnostics: OOF per-dimension errors and simulated missingness.
AI-assisted draft by GPT-6 (2026-10-27). Team must verify independently.
Does not use target values as predictors. Masking applies to observed test features only.
"""
import argparse,csv,json,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from sklearn.model_selection import StratifiedKFold
from q1_reference_rebuild import read_records,file_for,val,make_estimator

def write_file(p,header,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(header);w.writerows(rows)

def main():
 a=argparse.ArgumentParser()
 a.add_argument('--data-dir',required=True)
 a.add_argument('--out-dir',default='results')
 a.add_argument('--repeats',type=int,default=10)
 args=a.parse_args()
 data=Path(args.data_dir);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
 schema=json.loads((Path(__file__).parent/'scorecard_schema.json').read_text(encoding='utf-8'))
 aliases={'A_ester_fruity':'A_esters_fruity','F_ester_fruity':'F_esters_fruity','4vg':'X4vg'}
 names=[aliases.get(x,x) for x in schema['scorecard'][1:-1]]
 che=read_records(file_for(data,'1.csv'))
 raw=read_records(file_for(data,'Raw sensory data anonymized.csv'))
 d=defaultdict(list)
 for row in raw:d[int(float(row['beer_name']))].append(row)
 features=[k for k in che[0] if k not in ('beer','beer_id','tasting_category_fine')]
 X=np.array([[val(r[f]) for f in features] for r in che],dtype=float)
 Y=np.array([[np.mean([val(t[c]) for t in d[k]]) for c in names] for k in range(1,251)])
 # Read original OOF instead of re-generating to check the numeric agreement.
 oof=list(csv.DictReader((out/'ETRaw_oof_scores_250.csv').open(encoding='utf-8-sig')))
 assert len(oof)==250
 pred=np.array([[val(r[n]) for n in names] for r in oof]);assert np.isfinite(pred).all()
 absolute=np.abs(Y-pred)
 per_dim=[[n,float(absolute[:,j].mean()),float(np.sqrt(np.mean((Y[:,j]-pred[:,j])**2))),float(np.mean(np.rint(np.clip(pred[:,j],0,6))==np.rint(Y[:,j]))),float(np.mean(Y[:,j]))] for j,n in enumerate(names)]
 write_file(out/'q1_per_dimension.csv',['dimension','oof_MAE','oof_RMSE','rounded_exact_rate','reference_mean'],per_dim)
 styles=np.array([c['tasting_category_fine'] for c in che]);count=Counter(styles)
 rare=np.array([s if count[s]>=5 else '__RARE__' for s in styles]);splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=2026).split(X,rare))
 af=np.array([j for j,n in enumerate(names) if n.startswith(('A_','F_'))])
 rates=[0,0.05,0.10,0.20]
 oof_robust={(rate,rep):np.full((250,len(names)),np.nan) for rate in rates for rep in range(args.repeats if rate else 1)}
 baseline_cv=np.full_like(Y,np.nan)
 for fold,(tr,te) in enumerate(splits):
  m=make_estimator('ET',2026+fold)
  m.fit(X[tr],Y[tr]);normal=np.clip(m.predict(X[te]),0,6);baseline_cv[te]=normal
  oof_robust[(0,0)][te]=normal
  for rate in rates[1:]:
   for rep in range(args.repeats):
    XX=X[te].copy();valid=np.flatnonzero(np.isfinite(XX))
    rng=np.random.default_rng(2026 + int(rate*1000)*100 + rep*17+fold*10007)
    idx=rng.choice(valid,size=round(len(valid)*rate),replace=False)
    XX.flat[idx]=np.nan
    oof_robust[(rate,rep)][te]=np.clip(m.predict(XX),0,6)
  print('fold',fold+1,'completed',flush=True)
 baseline_mae=np.mean(np.abs(baseline_cv[:,af]-Y[:,af]))
 stored_mae=np.mean(np.abs(pred[:,af]-Y[:,af]))
 assert abs(baseline_mae-stored_mae)<1e-5,(baseline_mae,stored_mae)
 rows=[]
 for rate in rates:
  R=args.repeats if rate else 1
  for rep in range(R):
   p=oof_robust[(rate,rep)]
   rows.append([rate,rep,float(np.mean(np.abs(p[:,af]-Y[:,af]))),float(np.mean(np.abs(p-Y))),float(np.mean(np.abs(p[:,af]-Y[:,af]))-baseline_mae)])
 write_file(out/'q1_missingness_robustness.csv',['masked_fraction_of_observed_validation_features','repeat','MAE_AF30','MAE_all50','delta_AF30_vs_unmasked'],rows)
 means={str(r):[float(np.mean([x[2] for x in rows if x[0]==r])),float(np.std([x[2] for x in rows if x[0]==r],ddof=0))] for r in rates}
 summary={'primary_oof_AF30_MAE_check':float(baseline_mae),'missingness_AF30_MAE_mean_sd_by_fraction':means,'runs_per_missing_fraction':args.repeats,'random_masking':'validation observed chemical entries only; independent from baseline native missing values','warning':'random masking is not an assay-specific detection-limit mechanism; cannot replace left-censoring analysis'}
 (out/'q1_diagnostics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
