import sys,time
sys.path.insert(0,'/mnt/data/MathorCup_A_Q2_Submission')
from q2_run import load_inputs,eligible_chemicals,rank_cor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from scipy.stats import rankdata
import numpy as np,pandas as pd
X,Y,chem,targets,style,fold,*_=load_inputs('/mnt/data','/mnt/data/MathorCup_A_Q2_Submission/split_assignments.csv')
allowed=eligible_chemicals(chem);pool=np.where(allowed)[0]; t=Y.shape[1]
def cond_select(X,Y,variant='raw',penalty=0.,style=None):
    n,p=X.shape; YY=Y.copy();XX=X.copy()
    if variant in ('rank','ranktarget'):XX=rankdata(XX,axis=0)/n
    if variant=='ranktarget': YY=rankdata(YY,axis=0)/n
    if variant=='style':
        for lab in np.unique(style):
            ids=(style==lab); XX[ids]-=XX[ids].mean(0);YY[ids]-=YY[ids].mean(0)
    XX-=XX.mean(0); YY-=YY.mean(0)
    XX/=np.maximum(np.linalg.norm(XX,axis=0),1e-8)
    YY/=np.maximum(np.linalg.norm(YY,axis=0),1e-8)
    selection=np.zeros((t,3),int)
    for j in range(t):
        prior=[]
        for k in range(3):
            if k==0:xr=XX; yr=YY[:,j]
            else:
                z=XX[:,prior];inv=np.linalg.inv(z.T@z+np.eye(len(prior))*.02)
                xr=XX-z@(inv@z.T@XX);yr=YY[:,j]-z@(inv@z.T@YY[:,j])
            sc=(xr.T@yr)**2/(np.sum(xr*xr,axis=0)+.02)
            sc[~allowed]=-999;sc[prior]=-999
            if prior and penalty:sc-=penalty*np.max(np.abs(XX.T@XX[:,prior]),axis=1)**2
            m=int(np.argmax(sc));prior.append(m)
        selection[j]=prior
    return selection

def cvtest():
 preds={name:np.zeros_like(Y) for name in ['spearman','greedy','style','rank','spearman_et','greedy_et','style_et','rank_et']}
 selected={m:[] for m in ['spearman','greedy','style','rank']}
 for k in range(5):
    tr=(fold!=k);te=(fold==k); imp=SimpleImputer(strategy='median');sc=RobustScaler(quantile_range=(10,90));xt=np.clip(sc.fit_transform(imp.fit_transform(X[tr])),-4,4);xv=np.clip(sc.transform(imp.transform(X[te])),-4,4)
    s=rank_cor(xt,Y[tr]);ss=pool[np.argsort(-abs(s[pool]),axis=0).T[:,:3]]
    ix={'spearman':ss,'greedy':cond_select(xt,Y[tr]),'style':cond_select(xt,Y[tr],'style',style=style[tr]),'rank':cond_select(xt,Y[tr],'rank')}
    for m, ids in ix.items():
      selected[m].append(ids)
      for j in range(t):
        xtr=xt[:,ids[j]];xte=xv[:,ids[j]]
        preds[m][te,j]=Ridge(alpha=8).fit(xtr,Y[tr,j]).predict(xte)
        md=ExtraTreesRegressor(n_estimators=70,max_features=1.,min_samples_leaf=5,random_state=2026+j,n_jobs=1)
        md.fit(xtr,Y[tr,j]);preds[m+'_et'][te,j]=md.predict(xte)
    print('fold',k,'Ridge',[(nm,round(abs(preds[nm][te]-Y[te]).mean(),4)) for nm in ix],flush=True)
 for nm in preds:
    print(nm,round(abs(preds[nm]-Y).mean(),6),flush=True)
 return preds,selected
if __name__=='__main__':
 p,s=cvtest();np.savez_compressed('/mnt/data/MathorCup_A_Q2_Improved/cond_cv.npz',**p)
 for n,a in s.items():np.save('/mnt/data/MathorCup_A_Q2_Improved/selected_'+n+'.npy',np.stack(a))
