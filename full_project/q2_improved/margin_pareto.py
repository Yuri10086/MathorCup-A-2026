import sys
sys.path.insert(0,'/mnt/data/MathorCup_A_Q2_Improved')
from q2_conditional_submission import *

def gate_sel(X,Y,allowed,margin=1.):
    n,p=X.shape;t=Y.shape[1]
    spear=rank_spearman(X,Y,np.flatnonzero(allowed));XX=X-X.mean(0);YY=Y-Y.mean(0)
    XX/=np.maximum(np.linalg.norm(XX,axis=0),1e-8);YY/=np.maximum(np.linalg.norm(YY,axis=0),1e-8)
    ix=spear.copy()
    for j in range(t):
        prior=list(spear[j,:2]);Z=XX[:,prior];I=np.linalg.inv(Z.T@Z+np.eye(2)*.02)
        R=XX-Z@(I@Z.T@XX);ry=YY[:,j]-Z@(I@Z.T@YY[:,j])
        score=(R.T@ry)**2/(np.sum(R*R,axis=0)+.02);score[~allowed]=-np.inf;score[prior]=-np.inf
        best=np.argmax(score);third=spear[j,2]
        if score[best]>score[third]*margin:ix[j,2]=best
    return ix

def run():
 X,Y,chem,targets,style,fold,*_=load_inputs('/mnt/data','/mnt/data/MathorCup_A_Q2_Improved/split_assignments.csv'); allowed=eligible_chemicals(chem)
 ms=[1.,1.2,1.5,2.,3.]
 preds={m:np.zeros_like(Y) for m in ms}
 for k in range(5):
   tr=np.flatnonzero(fold!=k);te=np.flatnonzero(fold==k);xt,xv=prep_fit(X[tr],X[te])
   for m in ms:preds[m][te]=score_preds(xt,xv,Y[tr],gate_sel(xt,Y[tr],allowed,m))
 print('MAE',[(m,round(abs(preds[m]-Y).mean(),6)) for m in ms],flush=True)
 rg=np.random.default_rng(SEED+712);groups=[np.flatnonzero(style==u) for u in np.unique(style)]
 bt={m:np.zeros((100,50,3),np.int16) for m in ms}
 for b in range(100):
   idx=np.concatenate([rg.choice(ii,len(ii),replace=True) for ii in groups]);xt,_=prep_fit(X[idx]);
   for m in ms:bt[m][b]=gate_sel(xt,Y[idx],allowed,m)
 print('JACCARD',[(m,round(avg_jaccard(bt[m]).mean(),6)) for m in ms],flush=True)
run()
