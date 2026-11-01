#!/usr/bin/env python3
# coding: utf-8
"""Q3 additional detection-floor mechanism and OOS audit (not proof of true LOD).
Reads and replaces synthetic 22-style chemical and sensory files with censor-aware version.
AI-assisted OpenAI GPT-6, 2026-10-27; human verification and disclosure required.
"""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from q3_complete import Transformer,StyleGaussian,split_by_style,fit_sensory,SEED

def fit_minimum_masses(X,style,threshold=.25,prior=10):
    # Identify duplicated global minima; indicators are indicative of floor-censoring, not certified lab LOD.
    low=np.nanmin(X,axis=0)
    ctr=np.sum(np.isclose(X,low[None,:],rtol=0,atol=1e-12),axis=0)
    eligible=ctr>=int(np.ceil(threshold*len(X)))
    gl=ctr/len(X)
    cats={}
    for s in np.unique(style):
        ind=style==s;n=np.sum(ind)
        nmin=np.sum(np.isclose(X[ind],low[None,:],rtol=0,atol=1e-12),axis=0)
        cats[s]=(nmin+prior*gl)/(n+prior)
    return low,eligible,cats

def apply_mass(X,styles,minima,eligible,masses,rng):
    out=X.copy();p=np.array([masses[s] for s in styles]);sample=rng.random((len(X),len(minima)))<p
    sample[:,~eligible]=False
    out[sample]=np.broadcast_to(minima,out.shape)[sample]
    return out

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--data-dir',default='/mnt/data');pa.add_argument('--q1-reference',default='/mnt/data/q1_rebuild/results/reference_raw_panel_means_250.csv');pa.add_argument('--results',default='/mnt/data/MathorCup_A_Q3_Submission/results');a=pa.parse_args()
    o=Path(a.results);root=Path(a.data_dir)
    df=pd.read_csv(next(iter(sorted(root.glob('1*.csv')))))
    ydf=pd.read_csv(next(iter(sorted(root.glob('4*.csv')))))
    chem=list(df.columns[3:]);target=list(ydf.columns[3:]);X=df[chem].to_numpy(float)
    style=df.tasting_category_fine.to_numpy(str)
    train,test=split_by_style(style)
    minima,mask,masses=fit_minimum_masses(X[train],style[train]);assert mask.sum()>0
    t=Transformer(12).fit(X[train]);ztr=t.encode(X[train])
    gauss=StyleGaussian().fit(ztr,style[train],sorted(np.unique(style).tolist()))
    rng=np.random.default_rng(SEED+777)
    # Fair held-out comparison: generated floor occupancy versus actual held-out style-feature occupancy
    rows=[]
    for s in np.unique(style[test]):
        tt=X[test][style[test]==s];n_test=len(tt)
        z=gauss.sample(s,400,rng)
        v=t.decode(z)
        vc=apply_mass(v,np.array([s]*len(v)),minima,mask,masses,rng)
        real=np.mean(np.isclose(tt,minima,rtol=0,atol=1e-12),axis=0)
        base=np.mean(np.isclose(v,minima,rtol=0,atol=1e-12),axis=0)
        improved=np.mean(np.isclose(vc,minima,rtol=0,atol=1e-12),axis=0)
        for j in np.where(mask)[0]:
            rows.append({'style':s,'chemical':chem[j],'n_real_test':n_test,
             'real_floor_fraction':float(real[j]),'raw_gaussian_floor_fraction':float(base[j]),
             'censor_aware_floor_fraction':float(improved[j]),
             'error_baseline':float(abs(base[j]-real[j])),
             'error_censor_aware':float(abs(improved[j]-real[j]))})
    detail=pd.DataFrame(rows);detail.to_csv(o/'holdout_floor_calibration.csv',index=False)
    # Weighted by held-out beer counts, not equal small styles
    weights=detail.n_real_test.to_numpy();weighted=lambda v:float(np.average(v,weights=weights))
    e0=weighted(detail.error_baseline);e1=weighted(detail.error_censor_aware)
    # Apply detection floor model to final full-data synthetic spectra
    minimo,mas,pm=fit_minimum_masses(X,style)
    gen=pd.read_csv(o/'generated_22styles_chemical231.csv')
    G=gen[chem].to_numpy(float);labels=gen['style'].to_numpy(str)
    corrected=apply_mass(G,labels,minimo,mas,pm,np.random.default_rng(SEED+1777))
    orig=pd.DataFrame(G,columns=chem);orig.insert(0,'style',gen['style']);orig.insert(1,'draw',gen['draw'])
    orig.to_csv(o/'generated_before_floor_projection.csv',index=False)
    gen[chem]=corrected
    gen.to_csv(o/'generated_22styles_chemical231.csv',index=False,encoding='utf-8-sig')
    ref=pd.read_csv(a.q1_reference,encoding='utf-8-sig')[target].to_numpy(float)
    predict=fit_sensory(X,ref)
    vals=predict(corrected)
    profile=pd.DataFrame(vals,columns=target);profile.insert(0,'style',gen['style']);profile.insert(1,'draw',gen['draw'])
    profile.to_csv(o/'generated_22styles_sensory_0to6.csv',index=False,encoding='utf-8-sig')
    profile.groupby('style')[target].mean().to_csv(o/'flavor_fingerprint_by_style.csv',encoding='utf-8-sig')
    # check on rebuilt outputs, not stale in-memory original
    violated=int(((corrected<minimo-1e-10)|(corrected>np.nanmax(X,axis=0)+1e-10)).sum())
    rec={'indicator_definition':'same observed column minimum in >=25% of training beers (possible detection floor; not proven LOD)',
         'n_minimum_pileup_features_training':int(mask.sum()),'n_minimum_pileup_features_full':int(mas.sum()),
         'holdout_weighted_floor_fraction_mae_baseline':e0,
         'holdout_weighted_floor_fraction_mae_censor_aware':e1,
         'relative_error_reduction_pct':100*(e0-e1)/e0,
         'styles_evaluated':int(len(np.unique(style[test]))),'style_feature_pairs':int(len(detail)),
         'full_generated_spectra':int(len(gen)),'floor_cells_set':int(np.sum(np.isclose(corrected,minimo[None,:],rtol=0,atol=1e-12)&mas[None,:])),
         'physical_column_minmax_violations':violated,
         'limitations':'Floor mass learned only from training beer cohorts for holdout; artificial minima are proxies, not proven instrument censor values.'}
    (o/'q3_censor_audit.json').write_text(json.dumps(rec,indent=2,ensure_ascii=False))
    main_summary=json.loads((o/'q3_summary.json').read_text());main_summary['detection_floor_calibration']=rec
    main_summary['generation']['chemical_bounds_violations_after_floor_projection']=violated
    (o/'q3_summary.json').write_text(json.dumps(main_summary,indent=2,ensure_ascii=False))
    print(json.dumps(rec,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
