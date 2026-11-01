#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MathorCup 2026 A Q2: conditional incremental sparse selection, audited vs Spearman.
AI-assisted drafting: OpenAI GPT-6, 2026-10-27. Review and disclose AI use.
Run: python q2_conditional_submission.py --data-dir /mnt/data --out-dir results --bootstrap 100
Q2 Y = official 4.csv standardized sensory targets, X = official 1.csv chemical data.
No Q2 official 2.csv or official 6.csv whole-cohort correlations used in CV selection.
Official 2.csv for FINAL descriptive graph only. Reference label from original paper unused.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
import sys
from collections import Counter
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from q2_run import load_inputs,eligible_chemicals,rank_cor,graph_build,SEED

def conditional_select(X,Y,allowed):
    """Forward 3-feature conditional squared gain (ridge residualization of both X and y).
    Scores compare each candidate after projecting out features already selected.
    Input: training-fold robustly scaled chemical X, 50 official standardized sensory Y.
    Constant/duplicate features protected. Do NOT use held-out y.
    """
    n,p=X.shape;t=Y.shape[1]
    xx=X.astype(float).copy(); yy=Y.astype(float).copy()
    xx-=xx.mean(0); yy-=yy.mean(0)
    xx/=np.maximum(np.linalg.norm(xx,axis=0),1.e-8)
    yy/=np.maximum(np.linalg.norm(yy,axis=0),1.e-8)
    top=np.zeros((t,3),dtype=np.int16)
    for j in range(t):
        prior=[]
        for _ in range(3):
            if not prior: xr=xx;yr=yy[:,j]
            else:
                z=xx[:,prior]; inv=np.linalg.inv(z.T@z+np.eye(len(prior))*.02)
                xr=xx-z@(inv@z.T@xx)
                yr=yy[:,j]-z@(inv@z.T@yy[:,j])
            gain=(xr.T@yr)**2/(np.sum(xr*xr,axis=0)+.02)
            gain[~allowed]=-np.inf
            gain[prior]=-np.inf
            idx=int(np.argmax(gain)); prior.append(idx)
        top[j]=prior
    return top

def prep_fit(Xtr,Xte=None):
    imp=SimpleImputer(strategy='median');sc=RobustScaler(quantile_range=(10,90))
    tr=np.clip(sc.fit_transform(imp.fit_transform(Xtr)),-4,4)
    te=None if Xte is None else np.clip(sc.transform(imp.transform(Xte)),-4,4)
    return tr,te

def rank_spearman(xt,yt,pool):
    cor=rank_cor(xt,yt)
    return pool[np.argsort(-abs(cor[pool]),axis=0).T[:,:3]]

def score_preds(xtr,xtest,ytr,idx,alpha=8.):
    p=np.zeros((len(xtest),ytr.shape[1]))
    for j in range(ytr.shape[1]):
        model=Ridge(alpha=alpha)
        model.fit(xtr[:,idx[j]],ytr[:,j])
        p[:,j]=model.predict(xtest[:,idx[j]])
    return p

def cv_one(X,Y,fold,allowed):
    t=Y.shape[1];p0=np.zeros_like(Y);p1=np.zeros_like(Y);models=[];selection=[]
    for k in sorted(np.unique(fold)):
        tr=np.where(fold!=k)[0];te=np.where(fold==k)[0]
        xtr,xte=prep_fit(X[tr],X[te]); pool=np.where(allowed)[0]
        b=rank_spearman(xtr,Y[tr],pool)
        a=conditional_select(xtr,Y[tr],allowed)
        p0[te]=score_preds(xtr,xte,Y[tr],b)
        p1[te]=score_preds(xtr,xte,Y[tr],a)
        models.append({'fold':int(k),'n_train':len(tr),'n_val':len(te),
                       'spearman_mae':float(np.abs(p0[te]-Y[te]).mean()),
                       'conditional_mae':float(np.abs(p1[te]-Y[te]).mean())})
        for target in range(t):
            for meth,sel in [('spearman',b),('conditional',a)]:
                for r,q in enumerate(sel[target]):selection.append({'fold':int(k),'sensory_index':target,'method':meth,'rank':r+1,'compound_index':int(q)})
    return p0,p1,pd.DataFrame(models),pd.DataFrame(selection)

def quantile_ci(y,p0,p1,B=5000):
    e0=np.mean(np.abs(y-p0),axis=1);e1=np.mean(np.abs(y-p1),axis=1)
    rg=np.random.default_rng(12019); n=len(y)
    inds=rg.integers(0,n,(B,n))
    delta=(e0-e1)[inds].mean(axis=1)
    return {'mae_spearman':float(e0.mean()),'mae_conditional':float(e1.mean()),
      'mae_improvement':float((e0-e1).mean()),'relative_improvement_pct':float(100*(e0-e1).mean()/e0.mean()),
      'paired_bootstrap_CI95':np.percentile(delta,[2.5,97.5]).round(6).tolist(),
      'per_beer_win_rate':float(np.mean(e1<e0))}

def make_folds(y,style,seed):
    c=Counter(style); classes=np.array([v if c[v]>=5 else 'RARE' for v in style])
    folds=np.full(len(y),-1,dtype=int)
    for k,(_,te) in enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=seed).split(y,classes)):folds[te]=k
    return folds

def strat_boot(X,Y,style,allowed,B=100):
    ids_by_style=[np.where(style==s)[0] for s in np.unique(style)]
    rg=np.random.default_rng(SEED+712)
    T=Y.shape[1];tb=np.zeros((B,T,3),dtype=np.int16);ts=np.zeros_like(tb)
    for b in range(B):
        ix=np.concatenate([rg.choice(ids,len(ids),replace=True) for ids in ids_by_style])
        xx,_=prep_fit(X[ix]);yy=Y[ix]
        tb[b]=conditional_select(xx,yy,allowed)
        ts[b]=rank_spearman(xx,yy,np.where(allowed)[0])
        if (b+1)%25==0:print('bootstrap',b+1,'/',B,flush=True)
    return tb,ts

def avg_jaccard(bt,Bpairs=3000):
    rng=np.random.default_rng(12018);b,t,_=bt.shape
    pp=rng.integers(0,b,size=(Bpairs,2));pp=pp[pp[:,0]!=pp[:,1]]
    a=[list(map(set,bt[:,i].tolist())) for i in range(t)]
    return np.array([np.mean([len(a[j][q]&a[j][z])/len(a[j][q]|a[j][z]) for q,z in pp]) for j in range(t)])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='/mnt/data');ap.add_argument('--out-dir',default=str(BASE/'results'));ap.add_argument('--bootstrap',type=int,default=100);ap.add_argument('--repeat-seeds',default='77,99');args=ap.parse_args()
    t0=time.time(); out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    # Must reuse original five fold protocol to compare exact earlier baseline.
    X,Y,chem,targets,style,fold,official,names,audit=load_inputs(args.data_dir,'/mnt/data/MathorCup_A_Q2_Submission/split_assignments.csv')
    assert len(X)==250 and len(chem)==231 and len(targets)==50
    allowed=eligible_chemicals(chem); pool=np.where(allowed)[0]
    p0,p1,fl,sel=cv_one(X,Y,fold,allowed)
    fixed=quantile_ci(Y,p0,p1)
    pd.DataFrame(p0,columns=targets).assign(beer_index=np.arange(1,251)).to_csv(out/'q2_oof_spearman.csv',index=False)
    pd.DataFrame(p1,columns=targets).assign(beer_index=np.arange(1,251)).to_csv(out/'q2_oof_conditional.csv',index=False)
    fl.to_csv(out/'q2_fold_comparison.csv',index=False)
    sel['sensory']=sel.sensory_index.map(dict(enumerate(targets)));sel['chemical']=sel.compound_index.map(dict(enumerate(chem)))
    sel.to_csv(out/'q2_cv_selected_5fold.csv',index=False)
    repeated=[]
    for seed in [int(v) for v in args.repeat_seeds.split(',') if v]:
        f=make_folds(Y,style,seed)
        a,b,fs,_=cv_one(X,Y,f,allowed)
        d=quantile_ci(Y,a,b,B=3000);d['seed']=seed
        repeated.append(d)
        print('repeat',seed,d,flush=True)
    tr,_=prep_fit(X)
    main_s=rank_spearman(tr,Y,pool); main_c=conditional_select(tr,Y,allowed)
    bt,bs=strat_boot(X,Y,style,allowed,args.bootstrap)
    np.savez_compressed(out/'q2_bootstrap_top3_comparison.npz',conditional=bt,spearman=bs)
    j0=avg_jaccard(bs);j1=avg_jaccard(bt)
    freq1=np.zeros((len(targets),len(chem)));freq0=np.zeros_like(freq1)
    for j in range(len(targets)):
        for q in bt[:,j,:].ravel():freq1[j,q]+=1
        for q in bs[:,j,:].ravel():freq0[j,q]+=1
    freq0/=len(bs);freq1/=len(bs)
    # descriptive signed graph from official 2.csv is separate from model selection
    _,adj,P,N,mod,stat,G=graph_build(tr,chem,input_rho=official)
    edges=[]
    for a,b in G.edges():edges.append({'chemical_1':chem[a],'chemical_2':chem[b],'rho':float(official[a,b]),'sign':'+' if official[a,b]>0 else '-','module_1':int(mod[a]),'module_2':int(mod[b])})
    pd.DataFrame(edges).to_csv(out/'q2_signed_network.csv',index=False,encoding='utf-8-sig')
    cor=rank_cor(tr,Y)
    final=[]
    for j,n in enumerate(targets):
        for r,q in enumerate(main_c[j],1):
            final.append({'sensory':n,'rank':r,'chemical':chem[q],'chemical_2csv_name':names[q],
                          'association_spearman':float(cor[q,j]),'module':int(mod[q]),
                          'conditional_top3_bootstrap_frequency':float(freq1[j,q]),
                          'spearman_top3_bootstrap_frequency':float(freq0[j,q]),
                          'full_spearman_top3_contains':bool(q in main_s[j]),
                          'interpretation':'association only, mechanism not causally verified'})
    pd.DataFrame(final).to_csv(out/'q2_final_top3_50x3.csv',index=False,encoding='utf-8-sig')
    stability=pd.DataFrame({'sensory':targets,'conditional_jaccard':j1,'spearman_jaccard':j0,
         'conditional_final3_avg_frequency':np.array([np.mean(freq1[j,main_c[j]]) for j in range(len(targets))]),
         'spearman_final3_avg_frequency':np.array([np.mean(freq0[j,main_s[j]]) for j in range(len(targets))])})
    stability.to_csv(out/'q2_stability.csv',index=False)
    errorby=pd.DataFrame({'sensory':targets,'spearman_MAE':np.abs(p0-Y).mean(0),'conditional_MAE':np.abs(p1-Y).mean(0)})
    errorby['delta_MAE']=errorby.spearman_MAE-errorby.conditional_MAE
    errorby.to_csv(out/'q2_per_sensory_cv.csv',index=False)
    report={'protocol':'Official 1.csv X + 4.csv Y; molecular candidates only; no whole-cohort 2.csv/6.csv in selection; fold-only preprocessing; top3 Ridge alpha=8 both models',
       'prior_fixed_5fold':fixed,'repeated_stratified_5fold':repeated,
       'conditional_average_Jaccard':float(j1.mean()),'spearman_average_Jaccard':float(j0.mean()),
       'conditional_avg_final3_boot_frequency':float(stability.conditional_final3_avg_frequency.mean()),
       'spearman_avg_final3_boot_frequency':float(stability.spearman_final3_avg_frequency.mean()),
       'conditional_improved_dimensions':int(np.sum(errorby.delta_MAE>0)),
       'conditional_more_stable_dimensions':int(np.sum(j1>j0)),
       'graph_stats':stat,'n_molecular_candidates':int(allowed.sum()),'bootstrap_reps':args.bootstrap,
       'interpretation':'Exploratory same-cohort CV. No true external validation. Greedy conditional selection is a known family of forward selection, not an invented algorithm.',
       'runtime_s':round(time.time()-t0,2)}
    (out/'q2_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINAL',json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
