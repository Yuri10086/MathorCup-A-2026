#!/usr/bin/env python3
"""Q2 final postprocessing: chemical metadata, reference checks, baseline stability, audit.
AI-assisted (OpenAI GPT-6, 2026-10-27); user must review/adopt/disclose.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from artifact_tool import Blob, SpreadsheetFile
from q2_run import load_inputs,rank_cor,graph_build,eligible_chemicals,SEED,find_file

BASE=Path(__file__).resolve().parent
OUT=BASE/'results';OUT.mkdir(exist_ok=True)


def pair_jaccard(bt):
    rng=np.random.default_rng(20268)
    b,t,_=bt.shape
    pairs=rng.integers(0,b,size=(5000,2));pairs=pairs[pairs[:,0]!=pairs[:,1]]
    return np.array([np.mean([len(set(bt[i,j])&set(bt[k,j]))/len(set(bt[i,j])|set(bt[k,j])) for i,k in pairs]) for j in range(t)])

def boot_spearman(x,y,style,chem,B):
    rng=np.random.default_rng(SEED+712)
    groups=[np.flatnonzero(style==s) for s in np.unique(style)]
    pool=np.flatnonzero(eligible_chemicals(chem))
    out=np.zeros((B,y.shape[1],3),np.int16)
    for b in range(B):
        idx=np.concatenate([rng.choice(ids,size=len(ids),replace=True) for ids in groups])
        xx=x[idx].copy()
        meds=np.nanmedian(xx,axis=0)
        missing=np.where(np.isnan(xx));xx[missing]=np.take(meds,missing[1])
        c=rank_cor(xx,y[idx])
        out[b]=pool[np.argsort(-abs(c[pool]),axis=0).T[:,:3]]
    return out

def ci_diff(a,b,seed=2029,B=3000):
    # a and b are per-beer MAE, improvement >0 means b better
    rng=np.random.default_rng(seed);n=len(a)
    ds=[]
    for _ in range(B):
        ids=rng.integers(0,n,n);ds.append(np.mean(a[ids]-b[ids]))
    return [float(np.mean(a-b)),float(np.percentile(ds,2.5)),float(np.percentile(ds,97.5))]

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='/mnt/data');ap.add_argument('--split-csv',default=str(BASE/'split_assignments.csv'))
    args=ap.parse_args()
    x,y,chem,targets,style,fold,rho,names,audit=load_inputs(args.data_dir,args.split_csv)
    main=pd.read_csv(OUT/'q2_top3_50x3.csv')
    assert main.shape[0]==150
    assert main.groupby('sensory').agg(n=('rank','count'),u=('chemical','nunique')).eq(3).all().all()
    assert not main.chemical.isin(set(chem)-set(np.array(chem)[eligible_chemicals(chem)])).any()
    # Import ONLY using spreadsheet-capable artifact_tool; no external metadata guessed.
    book=SpreadsheetFile.import_xlsx(Blob.load(str(find_file(args.data_dir,'8','.xlsx'))))
    v=book.worksheets.get_item('Sensory thresholds').get_range('A1:K262').values
    meta=pd.DataFrame(v[1:],columns=v[0]); meta=meta[meta.field_name_R.notna()].copy()
    meta=meta.rename(columns={'field_name_R':'chemical','chemical_name':'reference_chemical_name','origin_paper':'origin','threshold_mean':'reference_threshold','unit':'reference_unit','off_flavors':'reference_off_flavor_notes'})
    meta=meta[['chemical','reference_chemical_name','origin','reference_threshold','reference_unit','reference_off_flavor_notes']]
    assert len(meta.chemical.drop_duplicates())==len(meta)
    mapped=main.merge(meta,on='chemical',how='left',validate='many_to_one')
    assert mapped['reference_chemical_name'].notna().all()
    mapped.to_csv(OUT/'q2_top3_50x3_with_chemical_meta.csv',index=False,encoding='utf-8-sig')
    meta.to_csv(OUT/'q2_molecular_chemical_meta.csv',index=False,encoding='utf-8-sig')
    # stability baseline same stratified boot draws, same molecular candidate list
    B=int(json.load(open(OUT/'q2_runspec.json',encoding='utf-8'))['bootstrap'])
    bs=boot_spearman(x,y,style,chem,B)
    np.savez_compressed(OUT/'q2_bootstrap_spearman_top3_indices.npz',top3_indices=bs)
    bst=np.load(OUT/'q2_bootstrap_top3_indices.npz')['top3_indices']
    assert len(bst)==B
    fused_j=pair_jaccard(bst);spearman_j=pair_jaccard(bs)
    pool=np.flatnonzero(eligible_chemicals(chem))
    xx=x.copy();med=np.nanmedian(xx,axis=0);idx=np.where(np.isnan(xx));xx[idx]=np.take(med,idx[1])
    r=rank_cor(xx,y)
    spear_full=pool[np.argsort(-abs(r[pool]),axis=0).T[:,:3]]
    fused_full=main.pivot(index='sensory',columns='rank',values='index_1based').loc[targets].values.astype(int)-1
    fusion_freq=[];spear_freq=[]
    for j in range(len(targets)):
        fusion_freq.append(float(np.mean([q in fused_full[j] for rec in bst[:,j] for q in rec])))
        spear_freq.append(float(np.mean([q in spear_full[j] for rec in bs[:,j] for q in rec])))
    # each full-top3 has 3 items; above pooled fraction is correct as 3*B items / (3*B).
    sta=pd.DataFrame({'sensory':targets,'fusion_pairwise_Jaccard':fused_j,'spearman_pairwise_Jaccard':spearman_j,
                      'fusion_fulltop3_mean_selection_probability':fusion_freq,
                      'spearman_fulltop3_mean_selection_probability':spear_freq})
    sta.to_csv(OUT/'q2_stability_vs_spearman.csv',index=False,encoding='utf-8-sig')
    # direct compound sanity checks. Anchors do not force selection.
    anchors={'bitternes':'bitterness','alcohol':'ethanol..v.v.','body':'glycerol','X4vg':'X4VG','diacetyl':'X2.3.butanedione',
             'dms':'DMS','t2n':'X.E..2.nonenal','A_esters_isoaa':'isopentyl_acetate','acidity':'acetic_acid.g.L.',
             'lactic':'lactic_acid.mg.L.','F_esters_isoaa':'isopentyl_acetate','A_malt_all':'color.EBC.'}
    checks=[]
    for target,cmpd in anchors.items():
        if cmpd not in chem or target not in targets:continue
        ci=chem.index(cmpd);ti=targets.index(target)
        fs=main.loc[main.sensory==target,'chemical'].tolist()
        checks.append({'sensory':target,'expected_direct_marker':cmpd,'marker_type':'non-molecular instrumental measure' if cmpd=='bitterness' or cmpd=='color.EBC.' else 'measured chemical marker',
                       'marker_spearman':float(r[ci,ti]),'marker_spearman_abs_rank_all231':int(np.sum(abs(r[:,ti])>abs(r[ci,ti]))+1),
                       'marker_in_final_top3':cmpd in fs,'final_top3':';'.join(fs),
                       'interpretation':'Direct correspondence is not proof of causal impact; failure requires cautious chemical interpretation.'})
    pd.DataFrame(checks).to_csv(OUT/'q2_chemical_anchor_audit.csv',index=False,encoding='utf-8-sig')
    # signed graph parameter sensitivity: structure ONLY
    sens=[]
    for thr in (.30,.38,.45,.50):
        *_,stat,_=graph_build(xx,chem,threshold=thr,input_rho=rho)
        sens.append({'threshold':thr,**stat})
    pd.DataFrame(sens).to_csv(OUT/'q2_graph_threshold_sensitivity.csv',index=False)
    # boot confidence for OOF neural / sparse models
    yp=np.zeros_like(y)
    for name in ['fusion','spearman','all231_extratrees','all231_ridge']:
        yp[:]=pd.read_csv(OUT/f'q2_oof_{name}.csv')[targets].to_numpy(float)
    maep={name:np.abs(pd.read_csv(OUT/f'q2_oof_{name}.csv')[targets].values-y).mean(axis=1) for name in ['fusion','spearman','all231_extratrees','all231_ridge']}
    deep={m:np.abs(pd.read_csv(OUT/f'q2_oof_deep_{m}.csv')[targets].values-y).mean(axis=1) for m in ['true_signed','shuffled_signed','no_graph']}
    compare={'fusion_vs_spearman_positive_means_fusion_better':ci_diff(maep['spearman'],maep['fusion']),
             'graph_vs_no_graph_positive_means_graph_better':ci_diff(deep['no_graph'],deep['true_signed']),
             'graph_vs_shuffled_positive_means_graph_better':ci_diff(deep['shuffled_signed'],deep['true_signed']),
             'extratrees_vs_graph_positive_means_ET_better':ci_diff(deep['true_signed'],maep['all231_extratrees'])}
    details={'stability_mean_jaccard_fusion':float(fused_j.mean()),'stability_mean_jaccard_spearman':float(spearman_j.mean()),
             'avg_final_top3_frequency_fusion':float(np.mean(fusion_freq)), 'avg_final_top3_frequency_spearman':float(np.mean(spear_freq)),
             'fusion_vs_spearman_jaccard_dims_higher':int((fused_j>spearman_j).sum()),
             'chemical_anchor_in_top3':int(sum(z['marker_in_final_top3'] for z in checks if z['marker_type']=='measured chemical marker')),
             'chemical_anchor_count':int(sum(z['marker_type']=='measured chemical marker' for z in checks)),
             'paired_CI_improvements':compare}
    (OUT/'q2_final_audit.json').write_text(json.dumps(details,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(details,ensure_ascii=False,indent=2))
    # three single-purpose diagrams
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df=pd.read_csv(OUT/'q2_top3_cv_comparison.csv')
    fig,ax=plt.subplots(figsize=(8,3.6));ax.barh(df.method[::-1],df.mae_50[::-1]);ax.set_xlabel('Five-fold OOF MAE (50 sensory targets)');ax.set_title('Q2 predictive value of chemical-factor rankings');fig.tight_layout();fig.savefig(BASE/'figures'/'q2_ranking_mae.png',dpi=175);plt.close(fig)
    nd=pd.read_csv(OUT/'q2_deep_graph_ablation.csv').groupby('model').mae_50.agg(['mean','std']).reset_index()
    fig,ax=plt.subplots(figsize=(7.4,3.5));ax.barh(nd.model,nd['mean']);ax.set_xlim(.30,.47);ax.set_xlabel('Five-fold OOF MAE, average over two seeds');ax.set_title('Signed graph versus controls');fig.tight_layout();fig.savefig(BASE/'figures'/'q2_deep_ablation.png',dpi=175);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.6));ax.hist(fused_j,bins=12,alpha=.7,label='Fusion');ax.hist(spearman_j,bins=12,alpha=.5,label='Spearman');ax.set_xlabel('Pairwise bootstrap Top-3 Jaccard by sensory target');ax.set_ylabel('Number of sensory targets');ax.legend();fig.tight_layout();fig.savefig(BASE/'figures'/'q2_top3_stability.png',dpi=175);plt.close(fig)

if __name__=='__main__':main()
