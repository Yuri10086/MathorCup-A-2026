import sys
sys.path.insert(0,'/mnt/data/MathorCup_A_Q2_Improved')
from check_hybrids import hybrid_select
from q2_conditional_submission import *

def cons_sel(X,Y,allowed,seed=20,B=30,style=None,nfirst=2):
    p=X.shape[1];t=Y.shape[1];rg=np.random.default_rng(seed)
    freq=np.zeros((t,p))
    for b in range(B):
      ix=rg.choice(len(X),len(X),replace=True)
      bx=X[ix];by=Y[ix]
      top=rank_spearman(bx,by,np.where(allowed)[0]);
      for j in range(t):freq[j,top[j]]+=1
    # blend normal Spearman rank with stability freq
    cor=np.abs(rank_cor(X,Y))
    selections=np.zeros((t,3),int)
    for j in range(t):
      top=np.lexsort((-cor[:,j],-freq[j]))[::-1] # incorrect? sort final descending intentional
      top=np.argsort(-(freq[j]+0.0001*cor[:,j]))
      prior=list(map(int,[q for q in top if allowed[q]][:nfirst]))
      if nfirst==3:selections[j]=prior;continue
      xx=X-X.mean(0);yy=Y[:,j]-Y[:,j].mean();xx/=np.maximum(np.linalg.norm(xx,axis=0),1e-8); yy/=max(np.linalg.norm(yy),1e-8)
      z=xx[:,prior]; inv=np.linalg.inv(z.T@z+np.eye(len(prior))*.02)
      xr=xx-z@(inv@z.T@xx);yr=yy-z@(inv@z.T@yy)
      gain=(xr.T@yr)**2/(np.sum(xr*xr,axis=0)+.02);gain[~allowed]=-np.inf;gain[prior]=-np.inf
      prior.append(int(np.argmax(gain)));selections[j]=prior
    return selections

def run():
  X,Y,chem,targets,style,fold,*_=load_inputs('/mnt/data','/mnt/data/MathorCup_A_Q2_Submission/split_assignments.csv')
  allowed=eligible_chemicals(chem)
  preds={m:np.zeros_like(Y) for m in ['spear','spear2cond','cons2_cond1','cons3']}
  for k in range(5):
    tr=np.flatnonzero(fold!=k);te=np.flatnonzero(fold==k)
    xt,xv=prep_fit(X[tr],X[te]);tops={'spear':rank_spearman(xt,Y[tr],np.flatnonzero(allowed)),
      'spear2cond':hybrid_select(xt,Y[tr],allowed,2),
      'cons2_cond1':cons_sel(xt,Y[tr],allowed,seed=2026+k,B=30,nfirst=2),
      'cons3':cons_sel(xt,Y[tr],allowed,seed=2026+k,B=30,nfirst=3)}
    for n,ids in tops.items():preds[n][te]=score_preds(xt,xv,Y[tr],ids)
  print('CV',[(m,round(abs(p-Y).mean(),6)) for m,p in preds.items()],flush=True)
  grp=[np.flatnonzero(style==s) for s in np.unique(style)]; rg=np.random.default_rng(SEED+712)
  bt={m:np.zeros((50,50,3),np.int16) for m in preds}
  for b in range(50):
    ids=np.concatenate([rg.choice(grp_i,len(grp_i),replace=True) for grp_i in grp]); xt,_=prep_fit(X[ids]);yt=Y[ids]
    bt['spear'][b]=rank_spearman(xt,yt,np.flatnonzero(allowed))
    bt['spear2cond'][b]=hybrid_select(xt,yt,allowed,2)
    bt['cons2_cond1'][b]=cons_sel(xt,yt,allowed,seed=99+b,B=20,nfirst=2)
    bt['cons3'][b]=cons_sel(xt,yt,allowed,seed=99+b,B=20,nfirst=3)
    if b%10==0:print('outerboot',b,flush=True)
  print('STABILITY',[(m,round(avg_jaccard(z).mean(),6)) for m,z in bt.items()])
run()
