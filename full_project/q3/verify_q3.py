#!/usr/bin/env python3
# coding:utf-8
"""Pure-output independent audit; recompute Q3 submitted file consistency."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='/mnt/data');ap.add_argument('--results',default='/mnt/data/MathorCup_A_Q3_Submission/results');a=ap.parse_args()
 data=Path(a.data_dir);out=Path(a.results)
 original=pd.read_csv(next(iter(sorted(data.glob('1*.csv')))))
 official_sens=pd.read_csv(next(iter(sorted(data.glob('4*.csv')))))
 chem=list(original.columns[3:]);sens=list(official_sens.columns[3:]);styles=sorted(original.tasting_category_fine.unique())
 gen=pd.read_csv(out/'generated_22styles_chemical231.csv');pred=pd.read_csv(out/'generated_22styles_sensory_0to6.csv')
 assert len(gen)==len(pred)==880 and len(chem)==231 and len(sens)==50
 assert set(gen['style'])==set(pred['style'])==set(styles) and all(gen['style'].value_counts()==40)
 assert gen[chem].to_numpy().shape==(880,231) and np.isfinite(gen[chem].to_numpy()).all()
 assert pred[sens].to_numpy().shape==(880,50) and np.isfinite(pred[sens].to_numpy()).all()
 assert 0 <= pred[sens].min().min() and pred[sens].max().max() <=6
 assert np.all(gen[chem].to_numpy()>=np.nanmin(original[chem].to_numpy(float),axis=0)-1e-10)
 assert np.all(gen[chem].to_numpy()<=np.nanmax(original[chem].to_numpy(float),axis=0)+1e-10)
 # Mean output must match fingerprint table
 av=pred.groupby('style')[sens].mean().sort_index();fp=pd.read_csv(out/'flavor_fingerprint_by_style.csv').set_index('style').sort_index()
 assert np.allclose(av.to_numpy(),fp.loc[av.index,sens].to_numpy(),atol=1e-8)
 # Parent-child check: output 0-6 grades from Q1 must obey scorecard hierarchy
 parents={'A_malt_all':['A_malt_grain','A_malt_bread','A_malt_cara','A_malt_burn'],
 'A_hops_all':['A_hops_citrus','A_hops_tropical','A_hops_noble','A_hops_woody'],
 'A_esters_all':['A_esters_ethac','A_esters_isoaa','A_esters_flower','A_esters_fruity'],
 'F_malt_all':['F_malt_grain','F_malt_bread','F_malt_cara','F_malt_burn'],
 'F_hops_all':['F_hops_citrus','F_hops_tropical','F_hops_noble','F_hops_woody'],
 'F_esters_all':['F_esters_ethac','F_esters_isoaa','F_esters_flower','F_esters_fruity']}
 nviol=sum(int((pred[ch].to_numpy()>pred[parent].to_numpy()+1e-9).sum()) for parent,chs in parents.items() for ch in chs)
 assert nviol==0,('invalid-scorecard',nviol)
 source=original.loc[original.tasting_category_fine=='Pils/Lager'].reset_index(drop=True)
 trans=pd.read_csv(out/'pils_to_tripel_optimized_chemical231.csv');audit=pd.read_csv(out/'pils_to_tripel_optimized_audit_11.csv')
 assert len(trans)==len(audit)==len(source)==11
 assert list(trans.source_beer.astype(int))==list(source.beer.astype(int))
 defects=['DMS','X2.3.butanedione','X4VG','X.E..2.nonenal']
 observed=source[defects].to_numpy(float);transport=trans[defects].to_numpy(float);mask=np.isfinite(observed)
 assert not np.any((transport>observed+1e-10)&mask)
 assert int((~mask).sum())==2
 assert np.all(audit.after_stale_predicted <=audit.source_stale_predicted+1e-10)
 assert int(np.sum(audit.distance_to_target_after<audit.distance_to_target_before-1e-4))==7
 s=json.loads((out/'q3_censor_repeated_summary.json').read_text());assert all(x['delta']>0 for x in s['by_seed'])
 ab=pd.read_csv(out/'transfer_ablation.csv');assert len(ab)==4
 r={'status':'PASS','generation':f'{len(gen)} x {len(chem)}','style_count':len(styles),
 'fingerprint_dimensions':len(sens),'parent_child_violations':nviol,'observed_defect_comparisons':int(mask.sum()),
 'observed_defect_worsening':int(((transport>observed+1e-10)&mask).sum()),'missing_source_defects':int((~mask).sum()),
 'stale_proxy_worsening':int((audit.after_stale_predicted>audit.source_stale_predicted+1e-10).sum()),
 'closer_to_target':7,'all_three_holdout_splits_floor_better':True}
 (out/'verification.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
