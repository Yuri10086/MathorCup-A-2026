import numpy as np,sys
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
sys.path.insert(0,'/mnt/data/MathorCup_A_Q2_Improved')
from q2_conditional_submission import *

def hybrid_select(X,Y,allowed,n_spearman=1,diverse=0):
  n,p=X.shape; YY=Y-Y.mean(0);XX=X-X.mean(0)
  XX/=np.maximum(np.linalg.norm(XX,axis=0),1.e-8)
  YY/=np.maximum(np.linalg.norm(YY,axis=0),1.e-8)
  spear=rank_spearman(X,Y,np.where(allowed)[0]);result=np.zeros((Y.shape[1],3),int)
  for j in range(Y.shape[1]):
    prior=list(map(int,spear[j,:n_spearman]));
    for kk in range(n_spearman,3):
        z=XX[:,prior];inv=np.linalg.inv(z.T@z+np.eye(len(prior))*.02)
        xr=XX-z@(inv@z.T@XX);yr=YY[:,j]-z@(inv@z.T@YY[:,j])
        scores=(xr.T@yr)**2/(np.sum(xr*xr,axis=0)+.02)
        scores[~allowed]=-np.inf;scores[prior]=-np.inf
        if diverse:
          r=np.max(abs(XX.T@XX[:,prior]),axis=1)
          scores-=diverse*r*r
        prior.append(int(np.argmax(scores)))
    result[j]=prior
  return result

def run():
  X,Y,chem,targets,style,fold,*_=load_inputs('/mnt/data','/mnt/data/MathorCup_A_Q2_Submission/split_assignments.csv')
  allowed=eligible_chemicals(chem)
  models=['spear','cond','spear1cond2','spear2cond1']
  pred={m:np.zeros_like(Y) for m in models}
  for k in range(5):
    tr=np.where(fold!=k)[0];te=np.where(fold==k)[0]
    xt,xv=prep_fit(X[tr],X[te]);tops={'spear':rank_spearman(xt,Y[tr],np.where(allowed)[0]),'cond':conditional_select(xt,Y[tr],allowed)}
    tops['spear1cond2']=hybrid_select(xt,Y[tr],allowed,1)
    tops['spear2cond1']=hybrid_select(xt,Y[tr],allowed,2)
    for m in models:pred[m][te]=score_preds(xt,xv,Y[tr],tops[m])
  print('predictive',[(m,round(abs(pred[m]-Y).mean(),6)) for m in models],flush=True)
  rg=np.random.default_rng(SEED+712);grp=[np.flatnonzero(style==s) for s in np.unique(style)]
  BT={m:np.zeros((100,50,3),np.int16) for m in models}
  for b in range(100):
    ix=np.concatenate([rg.choice(ids,len(ids),replace=True) for ids in grp]);xt,_=prep_fit(X[ix]);
    BT['spear'][b]=rank_spearman(xt,Y[ix],np.where(allowed)[0]);BT['cond'][b]=conditional_select(xt,Y[ix],allowed)
    BT['spear1cond2'][b]=hybrid_select(xt,Y[ix],allowed,1);BT['spear2cond1'][b]=hybrid_select(xt,Y[ix],allowed,2)
  print('stability',[(m,round(avg_jaccard(BT[m]).mean(),6)) for m in models],flush=True)
  for m in models:print('conf',m,quantile_ci(Y,pred['spear'],pred[m],3000),flush=True)
if __name__=="__main__":run()
