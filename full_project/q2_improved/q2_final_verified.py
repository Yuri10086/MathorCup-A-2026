#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q2 final candidate: Spearman first-2 + residual incremental third compound.
The original methods are established Spearman and ridge-based conditional forward
selection, not an entirely novel invention. Refactor and verify before submission.
AI-assisted OpenAI GPT-6, 2026-10-27. Cite/disclose under MathorCup AI policy.
Run: python q2_final_verified.py --data-dir DATA --out-dir results --bootstrap 100
"""
import argparse, json, time
from pathlib import Path
import numpy as np,pandas as pd
from q2_run import load_inputs,rank_cor,graph_build,eligible_chemicals,SEED
from q2_conditional_submission import prep_fit,rank_spearman,score_preds,quantile_ci,make_folds,avg_jaccard
from check_hybrids import hybrid_select

def evaluate(X,Y,fold,allowed):
    S=np.zeros_like(Y);H=np.zeros_like(Y); fold_summary=[];selected=[]
    for k in sorted(np.unique(fold)):
        tr=np.flatnonzero(fold!=k);te=np.flatnonzero(fold==k)
        xt,xv=prep_fit(X[tr],X[te]);s=rank_spearman(xt,Y[tr],np.flatnonzero(allowed));h=hybrid_select(xt,Y[tr],allowed,2)
        S[te]=score_preds(xt,xv,Y[tr],s);H[te]=score_preds(xt,xv,Y[tr],h)
        fold_summary.append({'fold':int(k),'n_train':len(tr),'n_validation':len(te),'spearman_mae':float(abs(S[te]-Y[te]).mean()),'hybrid_mae':float(abs(H[te]-Y[te]).mean())})
        for j in range(Y.shape[1]):
            for m,arr in [('spearman',s),('hybrid',h)]:
                for rank,q in enumerate(arr[j],1):selected.append({'fold':int(k),'sensory_index':j,'method':m,'rank':rank,'chemical_index':int(q)})
    return S,H,pd.DataFrame(fold_summary),pd.DataFrame(selected)

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--data-dir',required=True);pa.add_argument('--out-dir',default='results');pa.add_argument('--bootstrap',type=int,default=100)
    args=pa.parse_args();start=time.monotonic();o=Path(args.out_dir);o.mkdir(parents=True,exist_ok=True)
    X,Y,chem,targets,style,fold,R,names,audit=load_inputs(args.data_dir,str(Path(__file__).resolve().parent/'split_assignments.csv'))
    allowed=eligible_chemicals(chem);assert allowed.sum()==218
    s,h,fs,selected=evaluate(X,Y,fold,allowed)
    fs.to_csv(o/'cv_per_fold.csv',index=False)
    selected['sensory']=selected.sensory_index.map(dict(enumerate(targets)))
    selected['chemical']=selected.chemical_index.map(dict(enumerate(chem)))
    selected.to_csv(o/'cv_selection.csv',index=False)
    pd.DataFrame(h,columns=targets).assign(original_beer_index=np.arange(1,251)).to_csv(o/'oof_hybrid.csv',index=False)
    pd.DataFrame(s,columns=targets).assign(original_beer_index=np.arange(1,251)).to_csv(o/'oof_spearman.csv',index=False)
    main_ci=quantile_ci(Y,s,h,5000)
    repeated=[]
    for seed in [77,99]:
        folds=make_folds(Y,style,seed); sp,hp,_,_=evaluate(X,Y,folds,allowed)
        ci=quantile_ci(Y,sp,hp,3000);ci['seed']=seed;repeated.append(ci)
        print('additional split',seed,ci,flush=True)
    xfull,_=prep_fit(X);ss=rank_spearman(xfull,Y,np.flatnonzero(allowed));hh=hybrid_select(xfull,Y,allowed,2)
    rng=np.random.default_rng(SEED+712); groups=[np.flatnonzero(style==u) for u in np.unique(style)]
    bs=np.zeros((args.bootstrap,len(targets),3),np.int16);bh=bs.copy()
    for b in range(args.bootstrap):
        sample=np.concatenate([rng.choice(ii,len(ii),replace=True) for ii in groups]);z,_=prep_fit(X[sample]);yy=Y[sample]
        bs[b]=rank_spearman(z,yy,np.flatnonzero(allowed));bh[b]=hybrid_select(z,yy,allowed,2)
        if (b+1)%25==0:print('bootstrap',b+1,flush=True)
    np.savez_compressed(o/'bootstrap_indices.npz',spearman=bs,hybrid=bh)
    js,jh=avg_jaccard(bs),avg_jaccard(bh)
    f_s=np.zeros((len(targets),len(chem)));f_h=f_s.copy()
    for j in range(len(targets)):
        for q in bs[:,j,:].ravel():f_s[j,q]+=1
        for q in bh[:,j,:].ravel():f_h[j,q]+=1
    f_s/=args.bootstrap;f_h/=args.bootstrap
    _,A,P,N,modules,graphstats,G=graph_build(xfull,chem,input_rho=R)
    graph_edges=[{'chemical_a':chem[i],'chemical_b':chem[j],'rho':float(R[i,j]),'sign':'+' if R[i,j]>0 else '-', 'community_a':int(modules[i]),'community_b':int(modules[j])} for i,j in G.edges()]
    pd.DataFrame(graph_edges).to_csv(o/'signed_network.csv',index=False,encoding='utf-8-sig')
    cor=rank_cor(xfull,Y)
    top=[]
    for j,t in enumerate(targets):
        for r,q in enumerate(hh[j],1):
            top.append({'sensory':t,'rank':r,'chemical':chem[q],'name_in_2csv':names[q],
              'spearman_rho':float(cor[q,j]),'bootstrap_top3_frequency':float(f_h[j,q]),
              'molecular_graph_community':int(modules[q]),'also_in_spearman_top3':bool(q in ss[j]),
              'claim':'association; not causation; chemically verify compound'})
    result=pd.DataFrame(top);result.to_csv(o/'q2_final_top3_150.csv',index=False,encoding='utf-8-sig')
    stability=pd.DataFrame({'sensory':targets,'spearman_jaccard':js,'hybrid_jaccard':jh,'spear_final_top3_mean_freq':[float(np.mean(f_s[j,ss[j]])) for j in range(len(targets))],'hybrid_final_top3_mean_freq':[float(np.mean(f_h[j,hh[j]])) for j in range(len(targets))]})
    stability.to_csv(o/'stability_50.csv',index=False)
    per_target=pd.DataFrame({'sensory':targets,'spearman_mae':abs(s-Y).mean(0),'hybrid_mae':abs(h-Y).mean(0)})
    per_target['improvement_mae']=per_target.spearman_mae-per_target.hybrid_mae
    per_target.to_csv(o/'mae_by_target.csv',index=False)
    # Explicitly audit chemistry direct markers; no post-hoc overriding selected feature list.
    anchors={'alcohol':'ethanol..v.v.','X4vg':'X4VG','diacetyl':'X2.3.butanedione','dms':'DMS','t2n':'X.E..2.nonenal',
             'body':'glycerol','lactic':'lactic_acid.mg.L.','F_esters_isoaa':'isopentyl_acetate','A_esters_isoaa':'isopentyl_acetate'}
    anchor_rows=[]
    for t,c in anchors.items():
        if t not in targets or c not in chem:continue
        ti=targets.index(t);ci=chem.index(c)
        anchor_rows.append({'sensory':t,'direct_marker':c,'marker_abs_spearman':float(abs(cor[ci,ti])),
              'direct_marker_in_top3':bool(ci in hh[ti]),'selected_top3':';'.join(chem[q] for q in hh[ti])})
    pd.DataFrame(anchor_rows).to_csv(o/'direct_chemistry_anchor_checks.csv',index=False)
    summary={'fixed_original_5fold':main_ci,'other_stratified_5fold_splits':repeated,
      'mean_spearman_jaccard':float(js.mean()),'mean_hybrid_jaccard':float(jh.mean()),
      'hybrid_higher_jaccard_dimensions':int(np.sum(jh>js)),'hybrid_better_mae_dimensions':int(np.sum(per_target.improvement_mae>0)),
      'molecular_candidates':int(allowed.sum()),'graph':graphstats,'bootstrap':args.bootstrap,
      'ranking':'Top two absolute Spearman (training fold) + third compound maximizing residual conditional ridge gain',
      'predictive_evaluator':'Same three-input Ridge alpha=8 for both. Official 4.csv targets. Preprocessing within folds.',
      'bias_note':'Exploratory selection used existing whole-cohort CV. Improvements not externally confirmed; paired-bootstrap only describes fixed CV predictions.',
      'reliability_note':'Stability is worse than Spearman; cannot claim across-metric dominance.',
      'runtime_seconds':round(time.monotonic()-start,1)}
    (o/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    assert len(result)==150 and all(result.groupby('sensory').size()==3) and result.chemical.isin(np.array(chem)[allowed]).all()
    print('FINAL',json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
