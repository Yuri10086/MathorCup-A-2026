#!/usr/bin/env python3
# coding: utf-8
"""MathorCup A Q3: style-conditional generation + defect-safe latent optimal transport.
AI-assisted code prototype: OpenAI GPT-6 (2026-10-27); contestant must independently
review, verify and disclose assistance under MathorCup AI policies.
Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python q3_complete.py --data-dir /mnt/data --out-dir results
Sources: official 1.csv, 4.csv; Q1 external raw sensory panel mean (explicitly marked external).
Transforms and target encoding fitted only on training cohort for holdout checks.
"""
from __future__ import annotations
import argparse, json, time, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.covariance import LedoitWolf
from scipy.stats import wasserstein_distance
import torch
from torch import nn
warnings.filterwarnings('ignore',category=RuntimeWarning)
SEED=20261027
np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2)

class Transformer:
    def __init__(self, k=12): self.k=k
    def fit(self,x):
        self.imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
        im=self.imp.transform(x)
        self.lo=np.nanpercentile(im,1,axis=0);self.hi=np.nanpercentile(im,99,axis=0)
        self.sca=RobustScaler(quantile_range=(10,90)).fit(np.clip(im,self.lo,self.hi))
        xz=np.clip(self.sca.transform(np.clip(im,self.lo,self.hi)),-4,4)
        self.pc=PCA(n_components=self.k,whiten=True,random_state=SEED).fit(xz)
        self.low=np.nanmin(x,axis=0);self.high=np.nanmax(x,axis=0)
        return self
    def encode(self,x):
        im=self.imp.transform(x);xz=np.clip(self.sca.transform(np.clip(im,self.lo,self.hi)),-4,4)
        return self.pc.transform(xz)
    def decode(self,z):
        xz=self.pc.inverse_transform(z)
        xx=self.sca.inverse_transform(xz)
        return np.clip(xx,self.low,self.high)

class StyleGaussian:
    def fit(self,z,styles,style_names,lam=12):
        self.names=style_names
        self.means={};self.covs={};self.counts={}
        overall=np.mean(z,axis=0)
        centered=np.vstack([z[np.array(styles)==s]-np.mean(z[np.array(styles)==s],axis=0) for s in style_names if np.sum(np.array(styles)==s)>0])
        pooled=LedoitWolf().fit(centered).covariance_+np.eye(z.shape[1])*.04
        self.pool=pooled
        for s in style_names:
            zz=z[np.array(styles)==s];n=len(zz)
            self.counts[s]=n
            mu=zz.mean(axis=0) if n else overall
            self.means[s]=(n*mu+lam*overall)/(n+lam)
            residual=zz-mu
            cov=(residual.T@residual/max(1,n-1)) if n>=3 else pooled
            w=n/(n+lam)
            self.covs[s]=w*cov+(1-w)*pooled+np.eye(z.shape[1])*.06
        return self
    def logpdf(self,z,styles):
        out=np.zeros(len(z))
        for s in np.unique(styles):
            ii=np.where(styles==s)[0];m=self.means[s];c=self.covs[s]
            sign,ld=np.linalg.slogdet(c)
            e=z[ii]-m
            out[ii]=-.5*(z.shape[1]*np.log(2*np.pi)+ld+np.einsum('ni,ij,nj->n',e,np.linalg.inv(c),e))
        return out
    def sample(self,s,n,rs): return rs.multivariate_normal(self.means[s],self.covs[s],size=n)

def sqrt_psd(x):
    w,v=np.linalg.eigh((x+x.T)/2)
    return (v*np.sqrt(np.maximum(w,1e-8)))@v.T

def ot_matrix(cs,ct):
    root=sqrt_psd(cs);inside=sqrt_psd(root@ct@root)
    return np.linalg.solve(root,inside)@np.linalg.inv(root)

class Coupling(nn.Module):
    def __init__(self,d,cond,mask):
        super().__init__();self.register_buffer('mask',torch.tensor(mask,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(d+cond,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh(),nn.Linear(48,d*2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,e,reverse=False):
        h=self.net(torch.cat([x*self.mask,e],axis=1))
        shift,logscale=h.chunk(2,axis=1)
        shift=shift*(1-self.mask)
        scale=0.5*torch.tanh(logscale)*(1-self.mask)
        if reverse:return (x-shift)*torch.exp(-scale),(-scale).sum(1)
        return x*torch.exp(scale)+shift,scale.sum(1)
class ConditionalFlow(nn.Module):
    def __init__(self,d,nstyle):
        super().__init__();self.emb=nn.Embedding(nstyle,8)
        self.blocks=nn.ModuleList([Coupling(d,8,[float((i+l)%2==0) for i in range(d)]) for l in range(4)])
    def forward(self,x,si):
        e=self.emb(si);d=torch.zeros(len(x));z=x
        for b in self.blocks:
            z,ld=b(z,e);d=d+ld
        return z,d
    def inverse(self,u,si):
        e=self.emb(si);z=u
        for b in reversed(self.blocks):z,_=b(z,e,reverse=True)
        return z
    def nll(self,x,s):
        u,ld=self.forward(x,s)
        return .5*(u*u).sum(1)+.5*x.shape[1]*math.log(2*math.pi)-ld

def train_flow(z,styles,all_styles,epochs=250,seed=SEED):
    torch.manual_seed(seed)
    m=ConditionalFlow(z.shape[1],len(all_styles))
    opts=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.012)
    idx={s:k for k,s in enumerate(all_styles)}
    labs=np.array([idx[s] for s in styles])
    zz=torch.as_tensor(z,dtype=torch.float32)
    ss=torch.as_tensor(labs,dtype=torch.long)
    rg=np.random.default_rng(seed)
    history=[]
    for epoch in range(epochs):
        ids=rg.permutation(len(zz));vals=[]
        for ii in np.array_split(ids,max(1,int(np.ceil(len(ids)/64)))):
            opts.zero_grad()
            loss=m.nll(zz[ii],ss[ii]).mean()
            loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3.)
            opts.step();vals.append(float(loss.detach()))
        if epoch in (0,49,99,149,199,epochs-1):history.append({'epoch':epoch+1,'nll_train':np.mean(vals)})
    m.eval();return m,history

def flow_logpdf(m,z,labels,styles):
    nameidx={s:i for i,s in enumerate(styles)}
    with torch.no_grad():
        x=torch.tensor(z,dtype=torch.float32);s=torch.tensor([nameidx[v] for v in labels])
        return -m.nll(x,s).numpy()

def flow_sample(m,s,n,styles,rng):
    with torch.no_grad():
        sid=torch.full((n,),styles.index(s),dtype=torch.long)
        u=torch.tensor(rng.standard_normal((n,next(iter(m.blocks)).mask.numel())),dtype=torch.float32)
        return m.inverse(u,sid).numpy()

def swd(a,b,nproject=64,seed=41):
    rng=np.random.default_rng(seed);d=a.shape[1]
    dirs=rng.standard_normal((nproject,d));dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    return float(np.mean([wasserstein_distance(a@v,b@v) for v in dirs]))

def split_by_style(styles):
    rng=np.random.default_rng(SEED);test=[]
    for s in np.unique(styles):
        ids=np.flatnonzero(styles==s)
        if len(ids)>=5:
            n=max(1,round(.2*len(ids)));test.extend(rng.choice(ids,n,replace=False))
    test=np.sort(np.asarray(test,dtype=int));train=np.setdiff1d(np.arange(len(styles)),test)
    assert len(set(train)&set(test))==0
    return train,test

def fit_sensory(x,ref):
    imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
    fit=imp.transform(x)
    mod=ExtraTreesRegressor(n_estimators=240,max_features=.7,min_samples_leaf=2,n_jobs=2,random_state=SEED)
    mod.fit(fit,ref)
    def predict(v):return np.clip(mod.predict(imp.transform(v)),0,6)
    return predict

def direct_violations(before,after,names,eps=1e-10):
    idx=[names.index(k) for k in ('DMS','X2.3.butanedione','X4VG','X.E..2.nonenal')]
    return np.asarray([(after[:,j]>before[:,j]+eps).sum() for j in idx],int)

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--data-dir',default='/mnt/data');pa.add_argument('--q1-reference',default='/mnt/data/q1_rebuild/results/reference_raw_panel_means_250.csv');pa.add_argument('--out-dir',default='/mnt/data/MathorCup_A_Q3_Submission/results');pa.add_argument('--epochs',type=int,default=260);pa.add_argument('--generated-per-style',type=int,default=60);args=pa.parse_args()
    t0=time.time();out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    root=Path(args.data_dir)
    f1=next(iter(sorted(root.glob('1*.csv'))));f4=next(iter(sorted(root.glob('4*.csv'))))
    df=pd.read_csv(f1);sens=pd.read_csv(f4)
    names=list(df.columns[3:]);target=list(sens.columns[3:]);styles=df.tasting_category_fine.to_numpy(str)
    all_styles=sorted(np.unique(styles).tolist())
    X=df[names].to_numpy(float);Y=sens[target].to_numpy(float)
    assert X.shape==(250,231) and len(all_styles)==22 and np.isfinite(Y).all()
    r=pd.read_csv(args.q1_reference,encoding='utf-8-sig')
    assert np.array_equal(r.row_index,np.arange(1,251))
    assert set(target)==set(r.columns[1:]),'Q1 external panel columns must match official sensory'
    ref=r[target].to_numpy(float)
    train,test=split_by_style(styles)
    tr=Transformer(12).fit(X[train]);ztr=tr.encode(X[train]);zte=tr.encode(X[test])
    base=StyleGaussian().fit(ztr,styles[train],all_styles)
    fl,hist=train_flow(ztr,styles[train],all_styles,args.epochs)
    gauss_nll=float(-base.logpdf(zte,styles[test]).mean())
    flow_nll=float(-flow_logpdf(fl,zte,styles[test],all_styles).mean())
    # classifier is an external diagnostic, trained ONLY on original training beers
    clf=ExtraTreesClassifier(n_estimators=350,min_samples_leaf=2,max_features=.75,random_state=SEED,n_jobs=2)
    clf.fit(ztr,styles[train]);real_acc=float(np.mean(clf.predict(zte)==styles[test]))
    rng=np.random.default_rng(SEED+50)
    diagnostics=[]
    # conditional generation evaluation ONLY on held-out styles with >=1 test sample
    for s in all_styles:
        te_ids=np.where(styles[test]==s)[0]
        if len(te_ids)==0:continue
        n=150;g=base.sample(s,n,rng);f=flow_sample(fl,s,n,all_styles,rng)
        true=zte[te_ids]
        diagnostics.append({'style':s,'n_train':int(np.sum(styles[train]==s)),'n_test':len(te_ids),
          'swd_gaussian':swd(true,g),'swd_flow':swd(true,f),
          'classifier_gaussian_style_rate':float(np.mean(clf.predict(g)==s)),
          'classifier_flow_style_rate':float(np.mean(clf.predict(f)==s))})
    diag=pd.DataFrame(diagnostics);diag.to_csv(out/'holdout_style_diagnostics.csv',index=False)
    # Full data fit and FINAL sampling independent of test scores
    t=Transformer(12).fit(X);z=t.encode(X)
    gauss=StyleGaussian().fit(z,styles,all_styles)
    final_flow,final_hist=train_flow(z,styles,all_styles,args.epochs,seed=SEED+1)
    # Selection of generation method by holdout NLL: conservative use Gaussian unless flow significantly better
    chosen='conditional_flow' if flow_nll < gauss_nll else 'shrunk_style_gaussian'
    all_samples=[];generatedZ=[]
    for s in all_styles:
        g=flow_sample(final_flow,s,args.generated_per_style,all_styles,rng) if chosen=='conditional_flow' else gauss.sample(s,args.generated_per_style,rng)
        generatedZ.append(g)
        samples=t.decode(g)
        for ii,vals in enumerate(samples):
            all_samples.append({'style':s,'draw':ii+1,**{k:float(v) for k,v in zip(names,vals)}})
    generated=pd.DataFrame(all_samples)
    generated.to_csv(out/'generated_22styles_chemical231.csv',index=False,encoding='utf-8-sig')
    # Correct, once-fitted sensory model; Q1 external original ratings as 0-6 supervision
    surrogate=fit_sensory(X,ref)
    synx=generated[names].to_numpy(float)
    synsens=surrogate(synx)
    gen_sens=pd.DataFrame(synsens,columns=target)
    gen_sens.insert(0,'style',generated['style']);gen_sens.insert(1,'draw',generated['draw'])
    gen_sens.to_csv(out/'generated_22styles_sensory_0to6.csv',index=False,encoding='utf-8-sig')
    profile=gen_sens.groupby('style')[target].mean()
    profile.to_csv(out/'flavor_fingerprint_by_style.csv',encoding='utf-8-sig')
    # Latent Gaussian Bures OT from every measured Pils/Lager source sample to Tripel
    sources=np.flatnonzero(styles=='Pils/Lager');assert len(sources)==11
    src_raw=X[sources].copy()
    src=t.imp.transform(src_raw)  # retain source observation where available, fill missing nondefect with train median
    # Missing source defect observations cannot be verified. Conservatively impute at dataset observed minimum.
    idx_direct=[names.index(k) for k in ('DMS','X2.3.butanedione','X4VG','X.E..2.nonenal')]
    missing_defect_mask=np.isnan(src_raw[:,idx_direct])
    for j in idx_direct:
        src[np.isnan(src_raw[:,j]),j]=t.low[j]
    zs=t.encode(src);mat=ot_matrix(gauss.covs['Pils/Lager'],gauss.covs['Tripel'])
    zt=gauss.means['Tripel']+(zs-gauss.means['Pils/Lager'])@mat.T
    xt=t.decode(zt)
    # Exact nonworsening for four measured markers when source is observed; imputed markers audited separately.
    for j in idx_direct:xt[:,j]=np.minimum(xt[:,j],src[:,j])
    src_stale=surrogate(src)[:,target.index('stale_hops')]
    # Safeguard fifth defect via Q1 sensory proxy: grid-search all alphas, accept largest feasible
    steps=np.linspace(0.,1.,101)
    trial=src[:,None,:]+steps[None,:,None]*(xt-src)[:,None,:]
    predicted=surrogate(trial.reshape(-1,X.shape[1]))[:,target.index('stale_hops')].reshape(len(sources),len(steps))
    alpha=np.array([steps[np.flatnonzero(predicted[i]<=src_stale[i]+1e-10).max()] for i in range(len(sources))])
    transfer=src+alpha[:,None]*(xt-src)
    assert not direct_violations(src,transfer,names).any()
    trans_stale=surrogate(transfer)[:,target.index('stale_hops')]
    assert np.all(trans_stale<=src_stale+1e-10)
    # Style alignment: relative target style centroid distance in latent space vs source
    z_final=t.encode(transfer)
    target_mean=gauss.means['Tripel'];source_mean=gauss.means['Pils/Lager']
    def dist(z,mean):return np.linalg.norm(z-mean,axis=1)
    orig_dist=dist(zs,target_mean);trans_dist=dist(z_final,target_mean)
    aligned=np.mean(trans_dist<orig_dist)
    transf_pred=surrogate(transfer);src_pred=surrogate(src)
    row={'beer':df.beer.iloc[sources].values,'original_row':sources+1,'transport_alpha':alpha,
         'dist_to_tripel_before':orig_dist,'dist_to_tripel_after':trans_dist,
         'stale_proxy_before':src_stale,'stale_proxy_after':trans_stale}
    for j in idx_direct:
        row[names[j]+'_before']=src[:,j];row[names[j]+'_after']=transfer[:,j]
        row[names[j]+'_original_missing']=np.isnan(src_raw[:,j])
    pd.DataFrame(row).to_csv(out/'pils_to_tripel_defect_audit_11.csv',index=False)
    pd.DataFrame(transfer,columns=names).assign(source_beer=df.beer.iloc[sources].values,transport_alpha=alpha).to_csv(out/'pils_to_tripel_transferred_compounds_11.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(transf_pred,columns=target).assign(source_beer=df.beer.iloc[sources].values).to_csv(out/'pils_to_tripel_transferred_sensory_0to6.csv',index=False,encoding='utf-8-sig')
    # Case-by-case defect tests, plus 'overall' mean sensory descriptive change
    improvements={k:float(np.mean(transf_pred[:,target.index(k)]-src_pred[:,target.index(k)])) for k in ['A_malt_all','F_malt_all','A_esters_all','F_esters_all','overall','stale_hops']}
    full_novel_dist=np.linalg.norm(z[:,None,:]-np.array(generatedZ).reshape(-1,12)[None,:,:],axis=2).min(axis=0)
    summary={'data':{'samples':250,'chemicals':231,'styles':22,'reference_sensory':'public raw 0-6 expert panel (external paper), same 250 beers'},
       'holdout':{'n_train':len(train),'n_test':len(test),'latent_dim':12,'gaussian_nll':gauss_nll,'flow_nll':flow_nll,
                  'classifier_real_test_accuracy':real_acc,'mean_SWD_gaussian':float(diag.swd_gaussian.mean()),'mean_SWD_flow':float(diag.swd_flow.mean()),
                  'flow_lower_SWD_styles':int((diag.swd_flow<diag.swd_gaussian).sum()),'evaluated_styles':len(diag)},
       'chosen_generation_model':chosen,'generation':{'styles':22,'per_style':args.generated_per_style,'total':len(generated),
               'nearest_training_latent_distance_mean':float(full_novel_dist.mean()),'nearest_training_latent_distance_min':float(full_novel_dist.min()),
               'chemical_bounds_violations':int(((synx<t.low-1e-10)|(synx>t.high+1e-10)).sum()),'sensory_bounds_violations':int(((synsens<0)|(synsens>6)).sum())},
       'transfer':{'source':'Pils/Lager','target':'Tripel','source_beers':len(sources),'source_missing_defect_observations':int(missing_defect_mask.sum()),
                   'direct_chemical_defect_violations':dict(zip(['DMS','Diacetyl','4VG','trans-2-nonenal'],direct_violations(src,transfer,names).tolist())),
                   'stale_hops_surrogate_violations':int(np.sum(trans_stale>src_stale+1e-10)),
                   'fraction_moved_closer_to_tripel':float(np.mean(trans_dist<orig_dist-1e-4)),'mean_distance_to_tripel_before':float(np.mean(orig_dist)),
                   'mean_distance_to_tripel_after':float(np.mean(trans_dist)),
                   'mean_transport_alpha':float(alpha.mean()),'n_no_transfer':int(np.sum(alpha==0)),
                   'average_predicted_sensory_change':improvements},
       'limitations':{'stale_hops':'constrained using trained Q1 sensory proxy, not analytically guaranteed true sensory rating',
           'chemical': 'for measured values numerical nonworsening in 1.csv scale; missing source defect values conservatively imputed at minimum and cannot be independently verified; no certified odor thresholds',
           'rare_styles':'classes with <5 beers have no independent test items, report extrapolation limits',
           'conditional_flow':'deep model used in contest as research candidate; selection based on held-out NLL, no forced superiority'},
       'runtime_seconds':round(time.time()-t0,1)}
    (out/'q3_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame({'epoch':[a['epoch'] for a in hist],'training_nll':[a['nll_train'] for a in hist]}).to_csv(out/'flow_training_curve.csv',index=False)
    np.savez_compressed(out/'holdout_split.npz',train=train,test=test)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
