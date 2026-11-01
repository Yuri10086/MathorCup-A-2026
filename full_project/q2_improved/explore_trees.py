import sys,time
sys.path.insert(0,'/mnt/data/MathorCup_A_Q2_Submission')
from q2_run import load_inputs,eligible_chemicals,rank_cor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
import numpy as np,pandas as pd
X,Y,chem,targets,style,fold,*_=load_inputs('/mnt/data','/mnt/data/MathorCup_A_Q2_Submission/split_assignments.csv')
allowed=np.flatnonzero(eligible_chemicals(chem)); poolmask=eligible_chemicals(chem)
P={name:np.zeros_like(Y) for name in ['spear_ridge','spear_xgb','xgb_gain_ridge','xgb_gain_xgb','greedy_ridge','greedy_xgb']}
start=time.time()
for k in range(5):
 tr=np.where(fold!=k)[0];te=np.where(fold==k)[0]
 im=SimpleImputer(strategy='median');sc=RobustScaler(quantile_range=(10,90));xt=np.clip(sc.fit_transform(im.fit_transform(X[tr])),-4,4);xv=np.clip(sc.transform(im.transform(X[te])),-4,4)
 mar=rank_cor(xt,Y[tr]);spe=allowed[np.argsort(-abs(mar[allowed]),axis=0).T[:,:3]]
 for j in range(Y.shape[1]):
  model=XGBRegressor(n_estimators=210,max_depth=2,learning_rate=0.04,subsample=.85,colsample_bytree=.8,min_child_weight=8,reg_lambda=12,n_jobs=1,random_state=2026,tree_method='hist')
  model.fit(xt[:,allowed],Y[tr,j]); gain=model.feature_importances_
  xgt=allowed[np.argsort(-gain)[:3]]
  for nm, ids in [('spear',spe[j]),('xgb_gain',xgt)]:
   P[f'{nm}_ridge'][te,j]=Ridge(alpha=8).fit(xt[:,ids],Y[tr,j]).predict(xv[:,ids])
   m=XGBRegressor(n_estimators=190,max_depth=2,learning_rate=.045,subsample=.88,min_child_weight=9,reg_lambda=8,n_jobs=1,random_state=2026,tree_method='hist')
   m.fit(xt[:,ids],Y[tr,j]); P[f'{nm}_xgb'][te,j]=m.predict(xv[:,ids])
 print('fold',k,'seconds',round(time.time()-start,1),[(nm,round(abs(y[te]-Y[te]).mean(),4)) for nm,y in P.items() if nm not in ['greedy_ridge','greedy_xgb']],flush=True)
for nm,y in P.items():
 if not nm.startswith('greedy'):
  print(nm,'mae',abs(y-Y).mean(),'af30',abs(y[:,:30]-Y[:,:30]).mean())
