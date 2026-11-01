import sys
sys.path.insert(0,'/mnt/data/MathorCup_A_Q2_Improved')
from q2_conditional_submission import *
from check_hybrids import hybrid_select

def choose_gate(X,Y,sty,allowed,seed=11,inner_k=3):
  fold=make_folds(Y,sty,seed)
  errs=np.zeros((2,len(Y),Y.shape[1]))
  for k in range(inner_k):
    # for each small fold 3 equally sized, use a direct non-stratified shuffled KFold if folds generated 5
    pass
  from sklearn.model_selection import KFold
  f=KFold(n_splits=inner_k,shuffle=True,random_state=seed)
  for k,(tr,te) in enumerate(f.split(X)):
    xt,xv=prep_fit(X[tr],X[te]); yt=Y[tr]
    top_s=rank_spearman(xt,yt,np.flatnonzero(allowed));top_h=hybrid_select(xt,yt,allowed,2)
    errs[0,te]=abs(score_preds(xt,xv,yt,top_s)-Y[te])
    errs[1,te]=abs(score_preds(xt,xv,yt,top_h)-Y[te])
  return errs.mean(axis=1) # method,target

def run():
 X,Y,chem,targets,style,fold,*_=load_inputs('/mnt/data','/mnt/data/MathorCup_A_Q2_Submission/split_assignments.csv');allowed=eligible_chemicals(chem)
 models=['spear','hybrid','gated0','gated5','gated10']
 pred={m:np.zeros_like(Y) for m in models};records=[]
 for k in range(5):
  tr=np.flatnonzero(fold!=k);te=np.flatnonzero(fold==k)
  errors=choose_gate(X[tr],Y[tr],style[tr],allowed,seed=2026+k)
  xt,xv=prep_fit(X[tr],X[te]);s=rank_spearman(xt,Y[tr],np.flatnonzero(allowed));h=hybrid_select(xt,Y[tr],allowed,2)
  pred['spear'][te]=score_preds(xt,xv,Y[tr],s); pred['hybrid'][te]=score_preds(xt,xv,Y[tr],h)
  for th,name in [(.0,'gated0'),(.005,'gated5'),(.01,'gated10')]:
   gate=errors[0]-errors[1]>th
   ix=s.copy();ix[gate]=h[gate]
   pred[name][te]=score_preds(xt,xv,Y[tr],ix)
   records.append({'fold':k,'mode':name,'gate_dim_count':sum(gate),'dev_improvement':float((errors[0]-errors[1])[gate].mean()) if any(gate) else 0})
  print('fold',k, [(m,round(abs(pred[m][te]-Y[te]).mean(),4)) for m in models],flush=True)
 print('SUMMARY')
 for m in models:print(m,abs(pred[m]-Y).mean(), 'CI',quantile_ci(Y,pred['spear'],pred[m],2000)['paired_bootstrap_CI95'],flush=True)
 print(pd.DataFrame(records).to_string(index=False))
run()
