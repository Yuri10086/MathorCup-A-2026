#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026 MathorCup A / Q2: reproducible signed-network, stability and attribution study.
AI-assisted research/code scaffold: OpenAI GPT-6, 2026-10-27.
Team must review, rerun, adapt, cite and disclose AI use according to MathorCup rules.

Usage: python q2_run.py --data-dir /mnt/data --out-dir results --bootstrap 100
Inputs: 1(1).csv, 2(1).csv, 4(1).csv, optional frozen split_assignments.csv.
Trains on OFFICIAL standardized sensory labels (4.csv), NOT the external raw panel labels.
OOF evaluations recompute transformations AND signed graphs in training folds only.
The published 2.csv whole-cohort graph is used for *descriptive* final network only.
Top3 is an association ranking, not a causal-effect estimate.
"""
import argparse, csv, json, os, time, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
import networkx as nx
warnings.filterwarnings('ignore',category=RuntimeWarning)

SEED=2026
# Explicit molecular-candidate protocol. Physicochemical measurements, material pools,
# and computed aggregate features may explain flavor, but are not individual compounds.
EXCLUDED_NONMOLECULAR={
 'color.EBC.','pH','bitterness','CO2.PSI.','GFS','unfermentables',
 'kcalperc','betaglucan.mg.L.','protein.g.L.','esters_sum',
 'aroma_hops_sum','acids_sum','sulfur_sum'
}
def eligible_chemicals(chem):
    return np.array([c not in EXCLUDED_NONMOLECULAR for c in chem],dtype=bool)


def find_file(d,stem,suffix):
    arr=sorted(Path(d).glob(f'{stem}*{suffix}'))
    if len(arr)!=1:raise RuntimeError(f'Expected one {stem}*{suffix} at {d}, found {arr}')
    return arr[0]

def rank_cor(a,b):
    ra=rankdata(a,axis=0);rb=rankdata(b,axis=0)
    ra-=ra.mean(0);rb-=rb.mean(0)
    na=np.sqrt((ra*ra).sum(0));nb=np.sqrt((rb*rb).sum(0))
    c=(ra.T@rb)/np.maximum(np.outer(na,nb),1.e-12)
    return np.nan_to_num(c)

def graph_build(Xraw, names, threshold=.38, per_sign=6, input_rho=None):
    # Xraw is train-only matrix, imputed within fold/bootstrap before correlation
    rho=rank_cor(Xraw,Xraw) if input_rho is None else np.array(input_rho,float).copy()
    rho=np.nan_to_num(rho,nan=0.,posinf=0.,neginf=0.)
    rho=(rho+rho.T)/2
    np.fill_diagonal(rho,0)
    n=rho.shape[0]; A=np.zeros((n,n),float)
    for i in range(n):
        for sign in (1,-1):
            ids=np.where(sign*rho[i]>=threshold)[0]
            if len(ids):
                selected=ids[np.argsort(-(sign*rho[i,ids]))[:per_sign]]
                A[i,selected]=rho[i,selected]
    # union rather than intersection, preserving sign
    A=np.where(abs(A)>=abs(A.T),A,A.T)
    mask=abs(A)>0
    G=nx.Graph();G.add_nodes_from(range(n))
    pairs=np.argwhere(np.triu(mask,1))
    for i,j in pairs:
        G.add_edge(int(i),int(j),weight=float(abs(rho[i,j])),sign=int(np.sign(rho[i,j])))
    if G.number_of_edges():
        comm=nx.community.louvain_communities(G,weight='weight',seed=SEED,resolution=1)
        modularity=float(nx.community.modularity(G,comm,weight='weight'))
    else: comm=[{i} for i in range(n)];modularity=0.
    module=np.zeros(n,dtype=int)
    for k,grp in enumerate(sorted(comm,key=lambda s:(-len(s),min(s))),1):
        module[list(grp)]=k
    P=np.maximum(A,0);N=np.maximum(-A,0)
    P=P/np.maximum(P.sum(1,keepdims=True),1e-10)
    N=N/np.maximum(N.sum(1,keepdims=True),1e-10)
    stats=dict(n_compounds=n,n_edges=int(len(pairs)),positive_edges=int((np.triu(A,1)>0).sum()),negative_edges=int((np.triu(A,1)<0).sum()),modules=int(len(comm)),modularity=modularity,threshold=threshold,per_sign=per_sign)
    return rho,A,P,N,module,stats,G

def centered_within_style(X,Y,style):
    xx=X.copy();yy=Y.copy()
    for s in np.unique(style):
        ii=np.flatnonzero(style==s)
        xx[ii]-=xx[ii].mean(0)
        yy[ii]-=yy[ii].mean(0)
    return xx,yy

def pct_rank(mat):
    # shape [features,targets]; tied zeros stay at rank tied low
    return np.apply_along_axis(lambda a:(rankdata(a,method='average')-1)/(len(a)-1),0,mat)

def fit_evidence(X,Y,style,module,cor_mat,seed=SEED):
    imp=SimpleImputer(strategy='median');xs=np.clip(RobustScaler(quantile_range=(10,90)).fit_transform(imp.fit_transform(X)),-4,4)
    marginal=rank_cor(xs,Y)
    xr,yr=centered_within_style(xs,Y,np.array(style))
    partial=rank_cor(xr,yr)
    rg=Ridge(alpha=35.0).fit(xs,Y)
    ridgecoef=abs(rg.coef_.T)
    encoefs=np.zeros((xs.shape[1],Y.shape[1]));
    for t in range(Y.shape[1]):
        en=ElasticNet(alpha=.065,l1_ratio=.75,tol=1e-3,max_iter=2200,random_state=seed)
        en.fit(xs,Y[:,t]);encoefs[:,t]=abs(en.coef_)
    strength=(.27*pct_rank(abs(marginal))+.18*pct_rank(abs(partial))+
              .35*pct_rank(encoefs)+.20*pct_rank(ridgecoef))
    return strength,marginal,partial,encoefs,ridgecoef

def pick3(strength,cormat,module,diversity=True,allowed=None):
    n,nt=strength.shape;out=np.zeros((nt,3),dtype=int)
    for t in range(nt):
        used=[]
        for slot in range(3):
            score=strength[:,t].copy()
            if diversity and used:
                for j in used:
                    penalty=np.maximum(abs(cormat[:,j])-.65,0)/.35
                    penalty=np.clip(penalty,0,1)*(module==module[j])*.12
                    score-=penalty
            score[used]=-np.inf
            if allowed is not None:score[~allowed]=-np.inf
            k=int(np.argmax(score));used.append(k)
        out[t]=used
    return out

def load_inputs(data_dir,split_path):
    xdf=pd.read_csv(find_file(data_dir,'1','.csv'),low_memory=False)
    zdf=pd.read_csv(find_file(data_dir,'4','.csv'),low_memory=False)
    r=pd.read_csv(find_file(data_dir,'2','.csv'),low_memory=False)
    chem=list(xdf.columns[3:]);targets=list(zdf.columns[3:])
    assert xdf.shape==(250,234) and zdf.shape==(250,53)
    assert len(chem)==231 and len(targets)==50
    assert np.array_equal(xdf.beer.values,zdf.beer.values)
    assert np.array_equal(xdf.tasting_category_fine.values,zdf.tasting_category_fine.values)
    assert list(r.columns[1:])==list(r.iloc[:,0]),'2.csv row column ordering mismatch'
    assert r.shape==(231,232)
    X=xdf[chem].to_numpy(float);Y=zdf[targets].to_numpy(float)
    if not np.isfinite(Y).all():raise ValueError('Missing targets not supported')
    R=r.iloc[:,1:].to_numpy(float)
    # Test positional mapping by independently recomputing known Spearman pairs.
    checks=[]
    rng=np.random.default_rng(1027)
    for a,b in rng.integers(0,231,size=(35,2)):
        if a==b:continue
        s=spearmanr(X[:,a],X[:,b],nan_policy='omit').statistic
        if np.isfinite(s):checks.append(abs(s-R[a,b]))
    mx=float(max(checks))
    if mx>1e-8:raise ValueError(f'2.csv positional correlation alignment FAILED, max discrepancy {mx}')
    if split_path and Path(split_path).exists():
        sp=pd.read_csv(split_path)
        assert np.array_equal(sp['official_beer'].to_numpy(),xdf.beer.to_numpy())
        fold=sp['fold'].to_numpy(int)
        if set(fold)==set(range(1,6)):fold=fold-1
    else:
        from sklearn.model_selection import StratifiedKFold
        style=xdf.tasting_category_fine.to_numpy()
        vc=pd.Series(style).value_counts()
        strata=np.array([c if vc[c]>=5 else 'rare' for c in style])
        fold=np.full(len(style),-1,int)
        for k,(_,val) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,strata)):
            fold[val]=k
    assert set(fold)==set(range(5)) and all((fold==k).sum()==50 for k in range(5))
    return X,Y,chem,targets,style if False else xdf.tasting_category_fine.to_numpy(),fold,R,r.iloc[:,0].tolist(),{'positional_mapping_max_error':mx,'samples':len(X),'chemicals':len(chem),'sensory_targets':len(targets),'missing_chem_cells':int(np.isnan(X).sum())}

def eval_ranked(X,Y,style,fold,chem,targets):
    methods=['fusion','spearman','ridgecoef','random']
    allowed=eligible_chemicals(chem); pool=np.flatnonzero(allowed)
    pred={s:np.zeros_like(Y) for s in methods}
    fullpred=np.zeros_like(Y); treepred=np.zeros_like(Y); stdspec=[];rank_records=[]; selections={}
    full_importance=np.zeros((len(chem),len(targets)))
    for k in range(5):
        tr=np.flatnonzero(fold!=k);te=np.flatnonzero(fold==k)
        prep=SimpleImputer(strategy='median'); scaler=RobustScaler(quantile_range=(10,90))
        xt=np.clip(scaler.fit_transform(prep.fit_transform(X[tr])),-4,4); xv=np.clip(scaler.transform(prep.transform(X[te])),-4,4)
        c,A,P,N,modules,stats,_=graph_build(xt,chem)
        evid,mar,part,en,rr=fit_evidence(X[tr],Y[tr],style[tr],modules,c)
        ids={
          'fusion':pick3(evid,c,modules,allowed=allowed),
          'spearman':pool[np.argsort(-abs(mar[pool]),axis=0).T[:,:3]],
          'ridgecoef':pool[np.argsort(-rr[pool],axis=0).T[:,:3]],
        }
        rng=np.random.default_rng(SEED+k)
        ids['random']=np.vstack([rng.choice(pool,size=3,replace=False) for _ in targets])
        selections[k]=ids
        for name,top in ids.items():
            for j in range(Y.shape[1]):
                rr3=Ridge(alpha=8.).fit(xt[:,top[j]],Y[tr,j])
                pred[name][te,j]=rr3.predict(xv[:,top[j]])
        model=Ridge(alpha=35).fit(xt,Y[tr]);base=model.predict(xv);fullpred[te]=base
        tree=ExtraTreesRegressor(n_estimators=260,max_features=.7,min_samples_leaf=2,n_jobs=2,random_state=SEED+k)
        tree.fit(xt,Y[tr]);treepred[te]=tree.predict(xv)
        # Analytical OOF mean-square loss change when permuting one feature in each heldout fold.
        # Use deterministic permutations, compare under same frozen model.
        e=Y[te]-base; rng=np.random.default_rng(9000+k)
        for q in range(len(chem)):
            xd=xv[:,q]-xv[rng.permutation(len(te)),q]
            delta=xd[:,None]*model.coef_[:,q][None,:]
            full_importance[q]+=(np.mean((e+delta)**2-e**2,axis=0))/5.
        stdspec.append({'fold':k,**stats})
        rank_records.append({'fold':k,'n_train':len(tr),'n_val':len(te),**{f'mae_{s}':float(np.abs(pred[s][te]-Y[te]).mean()) for s in methods},'mae_all_ridge':float(np.abs(fullpred[te]-Y[te]).mean()),'mae_all_extratrees':float(np.abs(treepred[te]-Y[te]).mean())})
    summary=[{'method':n,'mae_50':float(np.abs(p-Y).mean()),'rmse_50':float(np.mean((p-Y)**2)**.5),'mae_af30':float(np.abs(p[:,:30]-Y[:,:30]).mean())} for n,p in {**pred,'all231_ridge':fullpred,'all231_extratrees':treepred}.items()]
    return pd.DataFrame(summary),pd.DataFrame(rank_records),pd.DataFrame(stdspec),pred,fullpred,treepred,full_importance,selections

class SignedNet:
    # Adapter to support torch-only optional dependency
    pass

def run_neural(X,Y,fold,chem,targets,model_seeds=(2026,2027),epochs=130):
    try:
        import torch
        from torch import nn
    except ImportError as e:raise RuntimeError('torch required for graph ablation') from e
    torch.set_num_threads(2)
    # graph mixing matrix is fixed from TRAIN-only co-expression
    class Net(nn.Module):
        def __init__(self,p,t,h=64):
            super().__init__()
            self.gate=nn.Parameter(torch.zeros(p))
            self.base=nn.Linear(p,h)
            self.aux=nn.Linear(p,h)
            self.head=nn.Sequential(nn.LayerNorm(h),nn.GELU(),nn.Dropout(.14),nn.Linear(h,t))
        def forward(self,z,adj):
            x=z*torch.sigmoid(self.gate)
            v=torch.nn.functional.gelu(self.base(x)+self.aux(x@adj))
            return self.head(v)
    records=[];pred_all={mode:[] for mode in ('true_signed','shuffled_signed','no_graph')}
    for seed in model_seeds:
        pred={mode:np.zeros_like(Y) for mode in pred_all}
        for k in range(5):
            tr=np.flatnonzero(fold!=k);te=np.flatnonzero(fold==k)
            imp=SimpleImputer(strategy='median');std=RobustScaler(quantile_range=(10,90))
            xt=np.clip(std.fit_transform(imp.fit_transform(X[tr])),-4,4);xv=np.clip(std.transform(imp.transform(X[te])),-4,4)
            c,A,P,N,_,_,_=graph_build(xt,chem)
            adj=(P-N).astype(np.float32)
            # control preserves signed graph spectrum and degrees up to relabeling
            perm=np.random.default_rng(seed+k).permutation(len(chem))
            adj_rand=adj[np.ix_(perm,perm)]
            mode_mats={'true_signed':adj,'shuffled_signed':adj_rand,'no_graph':np.eye(len(chem),dtype=np.float32)}
            ymean=Y[tr].mean(0);ysd=np.maximum(Y[tr].std(0),.05)
            yt=(Y[tr]-ymean)/ysd
            xten=torch.tensor(xt,dtype=torch.float32); yten=torch.tensor(yt,dtype=torch.float32)
            xval=torch.tensor(xv,dtype=torch.float32)
            for mode, mat in mode_mats.items():
                torch.manual_seed(seed+123*k)
                model=Net(xt.shape[1],Y.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.0016,weight_decay=.018)
                mix=torch.tensor(mat,dtype=torch.float32)
                # fully train on 200 train rows; no early stopping on heldout labels
                model.train()
                for ep in range(epochs):
                    gen=torch.Generator().manual_seed(seed+10000*k+ep)
                    permidx=torch.randperm(len(tr),generator=gen)
                    for ids in torch.split(permidx,64):
                        optimizer.zero_grad()
                        out=model(xten[ids],mix)
                        loss=torch.nn.functional.mse_loss(out,yten[ids])+.00035*torch.sigmoid(model.gate).mean()
                        loss.backward();optimizer.step()
                model.eval()
                with torch.no_grad():val_pred=model(xval,mix).numpy()*ysd+ymean
                pred[mode][te]=val_pred
            print(f'[neural] seed={seed}, fold={k}, mae true/shift/no '+', '.join(f'{np.abs(pred[m][te]-Y[te]).mean():.4f}' for m in mode_mats),flush=True)
        for mode in pred:
            pred_all[mode].append(pred[mode])
            records.append({'model':mode,'seed':seed,'mae_50':float(np.abs(pred[mode]-Y).mean()),'mae_af30':float(np.abs(pred[mode][:,:30]-Y[:,:30]).mean()),'rmse':float(np.mean((pred[mode]-Y)**2)**.5)})
    return pd.DataFrame(records),{m:np.mean(v,axis=0) for m,v in pred_all.items()}

def boot_final(X,Y,style,chem,targets,B=100):
    rng=np.random.default_rng(SEED+712)
    # final official graph for submitted 2.csv will be computed separately.
    n,p=X.shape;t=Y.shape[1]
    all_top=np.zeros((B,t,3),dtype=np.int16)
    ids_by_style=[np.flatnonzero(style==s) for s in np.unique(style)]
    for b in range(B):
        idx=np.concatenate([rng.choice(ids,size=len(ids),replace=True) for ids in ids_by_style])
        xb=X[idx];yb=Y[idx];sb=style[idx]
        im=SimpleImputer(strategy='median').fit_transform(xb)
        rho,_,_,_,mod,_,_=graph_build(im,chem)
        strength,*_=fit_evidence(xb,yb,sb,mod,rho,seed=SEED+b)
        all_top[b]=pick3(strength,rho,mod,allowed=eligible_chemicals(chem))
        if (b+1)%20==0:print(f'[bootstrap] {b+1}/{B}',flush=True)
    return all_top

def stability_stats(top,bt,n_features):
    B,T,K=bt.shape;n=n_features
    freq=np.zeros((T,n))
    for t in range(T):
        for q in bt[:,t,:].ravel():freq[t,q]+=1
    freq/=B
    rng=np.random.default_rng(20268)
    pairs=rng.integers(0,B,size=(min(5000,max(1000,50*B)),2));pairs=pairs[pairs[:,0]!=pairs[:,1]]
    jac=[]
    for t in range(T):
        a=[set(z.tolist()) for z in bt[:,t,:]]
        jac.append(float(np.mean([len(a[i]&a[j])/len(a[i]|a[j]) for i,j in pairs])))
    return freq,np.array(jac)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='/mnt/data');p.add_argument('--out-dir',default='results')
    p.add_argument('--split-csv',default=str(Path(__file__).resolve().parent/'split_assignments.csv'))
    p.add_argument('--bootstrap',type=int,default=100);p.add_argument('--epochs',type=int,default=130)
    p.add_argument('--skip-neural',action='store_true');p.add_argument('--skip-bootstrap',action='store_true')
    args=p.parse_args();out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    X,Y,chem,targets,style,fold,official,names,audit=load_inputs(args.data_dir,args.split_csv)
    audit['eligible_molecular_factors']=int(eligible_chemicals(chem).sum())
    audit['excluded_nonsingle_molecule_or_physicochemical_factors']=sorted(EXCLUDED_NONMOLECULAR)
    (out/'q2_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    # A. all-sample OFFICIAL 2.csv signed graph, only for descriptive FINAL analysis
    imp=SimpleImputer(strategy='median').fit_transform(X)
    rho,adj,P,N,module,stat,G=graph_build(imp,chem,input_rho=official)
    edges=[]
    for i,j in G.edges:
        edges.append({'compound_i':chem[i],'compound_j':chem[j],'spearman_rho':float(official[i,j]),'sign':'+' if official[i,j]>0 else '-','abs_rho':float(abs(official[i,j])),'module_i':int(module[i]),'module_j':int(module[j])})
    pd.DataFrame(edges).to_csv(out/'q2_signed_edges.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'chemical_index':np.arange(1,len(chem)+1),'official_1csv_column':chem,'official_2csv_name':names,'module':module}).to_csv(out/'q2_chemical_name_map.csv',index=False,encoding='utf-8-sig')
    (out/'q2_graph_stats.json').write_text(json.dumps(stat,indent=2),encoding='utf-8')
    print('[data]',audit,'[graph]',stat,flush=True)
    # B. no-leak OOF predictive comparison of top3 rankings
    models,folds,graphfold,pred,fullpred,treepred,delta,selections=eval_ranked(X,Y,style,fold,chem,targets)
    models.to_csv(out/'q2_top3_cv_comparison.csv',index=False)
    folds.to_csv(out/'q2_cv_by_fold.csv',index=False)
    graphfold.to_csv(out/'q2_graph_by_fold.csv',index=False)
    for name,v in {**pred,'all231_ridge':fullpred,'all231_extratrees':treepred}.items():
        pd.DataFrame(v,columns=targets).assign(sample_index=np.arange(1,len(X)+1)).to_csv(out/f'q2_oof_{name}.csv',index=False)
    # C. strict graph ablation; no target leakage into the graph or test fold
    if not args.skip_neural:
        neural,nout=run_neural(X,Y,fold,chem,targets,epochs=args.epochs)
        neural.to_csv(out/'q2_deep_graph_ablation.csv',index=False)
        for model,vals in nout.items():pd.DataFrame(vals,columns=targets).assign(sample_index=np.arange(1,len(X)+1)).to_csv(out/f'q2_oof_deep_{model}.csv',index=False)
    # D. full-sample rank, bootstrap stability. Ranking will be somewhat optimistic when citing sample-same OOF delta.
    strength,marg,part,en,rc=fit_evidence(X,Y,style,module,rho)
    finaltop=pick3(strength,rho,module,allowed=eligible_chemicals(chem))
    if not args.skip_bootstrap:
        bt=boot_final(X,Y,style,chem,targets,args.bootstrap)
        freq,jaccard=stability_stats(finaltop,bt,len(chem))
        np.savez_compressed(out/'q2_bootstrap_top3_indices.npz',top3_indices=bt)
    else:
        freq=np.full((len(targets),len(chem)),np.nan);jaccard=np.full(len(targets),np.nan)
    # direct measured descriptors; chemistry mechanism requires independent source checking
    rows=[]
    for j,target in enumerate(targets):
        for rank,f in enumerate(finaltop[j],1):
            rows.append({'sensory':target,'rank':rank,'chemical':chem[f],'chemical_2csv_name':names[f],
                        'index_1based':int(f+1),'association_spearman':float(marg[f,j]),
                        'within_style_spearman':float(part[f,j]),'same_direction_after_style_control':bool(np.sign(marg[f,j])==np.sign(part[f,j])),
                        'fused_score':float(strength[f,j]),'bootstrap_top3_frequency':float(freq[j,f]),
                        'oof_ridge_permutation_delta_mse':float(delta[f,j]),'module':int(module[f]),
                        'chemistry_status':'待根据论文/化学阈值/风格核查；相关不代表因果'})
    top_df=pd.DataFrame(rows)
    top_df.to_csv(out/'q2_top3_50x3.csv',index=False,encoding='utf-8-sig')
    stab=pd.DataFrame({'sensory':targets,'pairwise_bootstrap_jaccard':jaccard,'mean_final_top3_frequency':np.mean([freq[j,finaltop[j]] for j in range(len(targets))],axis=1)})
    stab.to_csv(out/'q2_stability_50.csv',index=False,encoding='utf-8-sig')
    (out/'q2_runspec.json').write_text(json.dumps({'seed':SEED,'bootstrap':args.bootstrap,'epochs':args.epochs,
      'ranker':'0.27 |Spearman| + 0.18 within-style |Spearman| + 0.35 |ElasticNet coefficients| + 0.20 |Ridge coefficients|, all percentile ranks; same-module redundancy penalty 0.12 for |rho|>0.65',
      'outer_folds':'prior Q1 fixed 5 folds, split by beer',
      'graph':'top6 positive/negative edges per node, |rho|>=0.38; each predictive fold recomputes correlation on train data',
      'Q2_target':'official 4.csv 50 standardized panel targets',
      'external_raw_sensory_used':False,
      'molecule_candidate_exclusion':sorted(EXCLUDED_NONMOLECULAR),
      'preprocessing':'Train-only median impute, RobustScaler 10th-90th quantile scale and clip values to [-4,4], including held-out samples; selected after exploratory sensitivity check; no outlier values from test fold are used to tune preprocessing.'},indent=2,ensure_ascii=False),encoding='utf-8')
    print('Q2 50x3 rows',len(rows),'final known mean frequency',stab.mean_final_top3_frequency.mean(),'avg Jaccard',np.nanmean(jaccard),'time',round(time.time()-t0,1),flush=True)

if __name__=='__main__':main()
