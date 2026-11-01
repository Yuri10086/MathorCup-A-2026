import sys,os,time
from pathlib import Path
sys.path.insert(0,'/mnt/data/MathorCup_A_Q2_Submission')
from q2_run import load_inputs,eligible_chemicals, rank_cor
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
X,Y,chem,targets,style,fold,*_=load_inputs('/mnt/data','/mnt/data/MathorCup_A_Q2_Submission/split_assignments.csv')
allowed=np.flatnonzero(eligible_chemicals(chem));n,t=Y.shape

def greedy(X,Y,mode='robust',penalty=0.,ridge_alpha=8):
    n,p=X.shape; T=Y.shape[1]
    XX=X.copy(); YY=Y.copy()
    if mode in ('rank','ranktarget'):
        XX=rankdata(X,axis=0)/n
        if mode=='ranktarget': YY=rankdata(Y,axis=0)/n
    XX-=XX.mean(0); YY-=YY.mean(0)
    s=np.sqrt(np.sum(XX**2,axis=0)); XX/=np.maximum(s,1e-8)
    YY/=np.maximum(np.sqrt(np.sum(YY**2,axis=0)),1e-8)
    ids=np.zeros((T,3),int)
    for j in range(T):
        selected=[]
        for k in range(3):
            if not selected:
                xr=XX;yr=YY[:,j]
            else:
                z=XX[:,selected]; inv=np.linalg.inv(z.T@z+np.eye(len(selected))*.02)
                xr=XX-z@(inv@z.T@XX);yr=YY[:,j]-z@(inv@z.T@YY[:,j])
            sc=(xr.T@yr)**2/(np.sum(xr*xr,axis=0)+.02)
            sc[~eligible_chemicals(chem)]=-1
            sc[selected]=-1
            if penalty and selected:
                co=np.abs(XX.T@XX[:,selected]);sc-=penalty*np.max(co,axis=1)**2
            choice=int(np.argmax(sc));selected.append(choice)
        ids[j]=selected
    return ids

modes=['ranktarget','rank','robust']
penalties=[0.,.03,.12]
configs=[(m,pen) for m in modes for pen in penalties]
P={f'{mode}_{pen}':np.zeros_like(Y) for mode,pen in configs};P['spearman']=np.zeros_like(Y)
for k in range(5):
 tr=np.where(fold!=k)[0];te=np.where(fold==k)[0]
 imp=SimpleImputer(strategy='median');sc=RobustScaler(quantile_range=(10,90));xt=np.clip(sc.fit_transform(imp.fit_transform(X[tr])),-4,4);xv=np.clip(sc.transform(imp.transform(X[te])),-4,4)
 rt=rank_cor(xt,Y[tr]); top=allowed[np.argsort(-abs(rt[allowed]),axis=0).T[:,:3]]
 def eval_top(name,idx):
  for j in range(t):P[name][te,j]=Ridge(alpha=8.).fit(xt[:,idx[j]],Y[tr,j]).predict(xv[:,idx[j]])
 eval_top('spearman',top)
 for mode,pen in configs:
  top=greedy(xt,Y[tr],mode,pen)
  eval_top(f'{mode}_{pen}',top)
 print('FOLD',k, sorted([(round(abs(P[n][fold==k]-Y[fold==k]).mean(),4),n) for n in P])[:5],flush=True)
for key in sorted(P,key=lambda nm: np.abs(P[nm]-Y).mean()):
 e=abs(P[key]-Y)
 print(key,'MAE',round(e.mean(),6),'AF30',round(e[:,:30].mean(),6),flush=True)
