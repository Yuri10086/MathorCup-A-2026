"""问题一：数据缺失条件下的感官评分（250x53 评分卡）

本文件由以下模块合并而成，对应论文一个问题的完整可复现代码：
  - run.py
  - q1_complete.py
  - q1_acetaldehyde.py
  - q1_grade_range.py
  - q1_scale_check.py
  - audit_extended.py
"""

import os,json,hashlib,time,warnings,sys,traceback
import sys
import numpy as np,pandas as pd
import torch
from paths import find_input as raw_path
import os
import numpy as np
import pandas as pd
from paths import find_input  # lightweight: does not import the training pipeline
import json


from pathlib import Path
from scipy.stats import kruskal


# --------------------------------------------------------------------------
# 来源: run.py
# --------------------------------------------------------------------------
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
from scipy.stats import rankdata,norm,wasserstein_distance,wilcoxon
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor,ExtraTreesClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import r2_score
from torch import nn
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore',category=RuntimeWarning)
torch.set_num_threads(2);threadpool_limits(2)
R=Path(__file__).resolve().parents[2];O=R/'outputs/rebuild';C=json.loads((R/'configs/rebuild_protocol.json').read_text(encoding='utf-8-sig'))
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
# Input resolution lives in paths.py so lightweight scripts can locate files without importing this module
# (importing run.py executes load() below, which needs the official attachments to exist).
def csv(d,p):pd.DataFrame(d).to_csv(O/p,index=False,encoding='utf-8-sig')
def js(d,p):(O/p).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def seed(s):np.random.seed(s);torch.manual_seed(s)
def load():
 x=pd.read_csv(raw_path('1.csv'));off=pd.read_csv(raw_path('4.csv'));raw=pd.read_csv(raw_path('Raw sensory data anonymized.csv'),encoding='cp1252').rename(columns={'A_ester_fruity':'A_esters_fruity','F_ester_fruity':'F_esters_fruity'})
 assert x.beer.is_unique and off.beer.is_unique and np.array_equal(x.beer,off.beer)
 cols=list(off.columns[3:]);chem=list(x.columns[3:]);Y=raw.groupby(['beer_name','taster'])[cols].mean().groupby('beer_name').mean().reindex(range(1,251)).to_numpy();Yr=raw.groupby('beer_name')[cols].mean().reindex(range(1,251)).to_numpy()
 assert Y.shape==(250,50) and np.isfinite(Y).all() and Y.min()>=0 and Y.max()<=6
 return x,off,raw,cols,chem,x[chem].to_numpy(float),Y,Yr,x.tasting_category_fine.to_numpy()
xdf,off,raw,targets,chem,X,Y,Yrecord,style=load();AF=np.array([j for j,c in enumerate(targets) if c.startswith(('A_','F_'))]);SAFE=np.array([j for j,c in enumerate(targets) if c!='barnyard'])
parents={p:[c for c in targets if c.startswith(p[:-3]) and c!=p] for p in targets if p.endswith('_all')};children=[c for cs in parents.values() for c in cs];base=[c for c in targets if c not in children]
BI=[targets.index(c) for c in base];CI=[targets.index(c) for c in children];PI=[base.index(p) for p,cs in parents.items() for c in cs]
def folds(s,n=3):
 rng=np.random.default_rng(s);f=np.empty(250,int)
 for st in np.unique(style):
  ix=np.flatnonzero(style==st);rng.shuffle(ix);f[ix]=(np.arange(len(ix))+rng.integers(n))%n
 return f
class Prep:
 def fit(self,a):self.imp=SimpleImputer(strategy='median').fit(a);v=self.imp.transform(a);self.scaler=StandardScaler().fit(v);return self
 def transform(self,a):return self.scaler.transform(self.imp.transform(a))
def metric(a,b):return dict(MAE=float(abs(a-b).mean()),RMSE=float(np.sqrt(np.mean((a-b)**2))),R2=float(r2_score(a,b,multioutput='uniform_average')),AF_MAE=float(abs(a[:,AF]-b[:,AF]).mean()),MAE49=float(abs(a[:,SAFE]-b[:,SAFE]).mean()))
def inner(tr,s):
 rng=np.random.default_rng(s);ix=tr.copy();rng.shuffle(ix);n=max(20,int(len(ix)*.2));return ix[n:],ix[:n]
def audit():
 # The manifest records the SHA-256 of the ORIGINAL published inputs. Never overwrite it silently:
 # if the recorded values differ from the inputs present now, write a separate rebuilt copy instead.
 from paths import BASES;seen={};manifest=[];_files=[p for base in BASES if base.is_dir() for p in sorted(base.glob('*')) if p.is_file()];manifest=[{'file':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in _files if not (p.name in seen or seen.setdefault(p.name,True) and False)]
 _rec=O/'audit/raw_manifest.csv';_target='audit/raw_manifest.csv'
 if _rec.exists():
  try:
   _old=pd.read_csv(_rec)
   _same=all(manifest_row['sha256']==dict(zip(_old.file,_old.sha256)).get(manifest_row['file']) for manifest_row in manifest) and len(_old)==len(manifest)
  except Exception:_same=False
  if _same:_target='audit/raw_manifest.csv'
  else:
   _target='audit/raw_manifest.rebuilt.csv';log('inputs differ from the recorded baseline; wrote raw_manifest.rebuilt.csv and left the baseline untouched')
 csv(manifest,_target)
 csv(pd.DataFrame(Y,columns=targets).assign(beer=xdf.beer,raw_beer_name=np.arange(1,251)),'audit/equal_rater_targets.csv');csv(pd.DataFrame(Yrecord,columns=targets),'audit/record_mean_targets.csv')
 rows=[]
 for j,c in enumerate(chem):v=X[:,j];lo=np.nanmin(v);rows.append(dict(feature=c,missing=int(np.isnan(v).sum()),minimum=lo,minimum_fraction=float(np.mean(v==lo)),maximum=np.nanmax(v),unique=int(len(np.unique(v[np.isfinite(v)])))))
 csv(rows,'audit/features.csv');csv(pd.Series(style).value_counts().rename_axis('style').reset_index(name='n'),'audit/styles.csv')
 for s in C['seeds']:csv(pd.DataFrame({'row':np.arange(250),'beer':xdf.beer,'fold':folds(s),'style':style}),f'audit/folds_{s}.csv')
 # Differences from record weighting and panelist additive variability are descriptive only.
 means=raw.groupby('taster')[targets].mean();csv(means.reset_index(),'audit/taster_descriptive_means.csv')
 rr=raw.copy();rr['n_repeat']=rr.groupby(['beer_name','taster']).taster.transform('size');csv(rr.groupby('beer_name').agg(records=('taster','size'),raters=('taster','nunique'),dates=('date','nunique')).reset_index(),'audit/repeats.csv')
 js({'device':DEVICE,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'python':sys.version,'protocol_sha256':hashlib.sha256((R/'configs/rebuild_protocol.json').read_bytes()).hexdigest(),'parent_child_violations':sum(int(((raw[c]!=0)&(raw[c]!=raw[p])).sum()) for p,cs in parents.items() for c in cs),'equal_rater_vs_record_MAE':float(abs(Y-Yrecord).mean()),'equal_rater_vs_record_max':float(abs(Y-Yrecord).max()),'claim':'Internal resampling only; no fresh external cohort'},'audit/environment.json')
 log('audit complete')
class RatingNet(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.enc=nn.Sequential(nn.Linear(231,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,32),nn.ReLU())
  if kind=='MLP':self.head=nn.Linear(32,50)
  else:
   self.base=nn.Linear(32,len(base)*7);self.child=nn.Linear(32,len(children)*7)
   if kind=='HierarchyRater':self.rater=nn.Embedding(raw.taster.nunique(),len(base)*7)
 def forward(self,a,r=None):
  h=self.enc(a)
  if self.kind=='MLP':return self.head(h),None,None
  b=self.base(h).reshape(-1,len(base),7)
  if hasattr(self,'rater'):
   emb=self.rater.weight-self.rater.weight.mean(0,keepdim=True)
   if r is not None:b=b+emb[r].reshape(-1,len(base),7)
  p=b.softmax(-1);q=self.child(h).reshape(-1,len(children),7).sigmoid();k=torch.arange(7,device=a.device)
  out=torch.zeros((len(a),50),device=a.device);out[:,BI]=(p*k).sum(-1);out[:,CI]=(p[:,PI,:]*q*k).sum(-1)
  return out,b,q
class Deep:
 def __init__(self,kind,s):self.kind=kind;self.s=s
 def fit(self,a,y,beer_indices,val_a=None,val_y=None,tag='',fixed_epochs=None):
  seed(self.s);self.net=RatingNet(self.kind).to(DEVICE);opt=torch.optim.AdamW(self.net.parameters(),lr=.002,weight_decay=.001)
  A=torch.tensor(a,dtype=torch.float32,device=DEVICE);T=torch.tensor(y,dtype=torch.float32,device=DEVICE)
  if self.kind!='MLP':
   sub=raw[raw.beer_name.isin(beer_indices+1)].copy();loc={v+1:i for i,v in enumerate(beer_indices)};row=torch.tensor(sub.beer_name.map(loc).to_numpy(),device=DEVICE);obs=torch.tensor(sub[targets].to_numpy(),dtype=torch.float32,device=DEVICE);rat=torch.tensor(pd.Categorical(sub.taster,categories=sorted(raw.taster.unique())).codes.astype('int64'),device=DEVICE)
   counts=sub.groupby(['beer_name','taster']).taster.transform('size').to_numpy();nr=sub.groupby('beer_name').taster.transform('nunique').to_numpy();w=torch.tensor(1/(counts*nr),dtype=torch.float32,device=DEVICE);w=w/w.sum()
  history=[];best=np.inf;beststate=None
  for e in range(fixed_epochs or C['deep_epochs']):
   self.net.train();opt.zero_grad()
   if self.kind=='MLP':pred,_,_=self.net(A);loss=((pred-T)**2).mean()
   else:
    _,b,q=self.net(A[row],rat if self.kind=='HierarchyRater' else None);bt=obs[:,BI].long();ce=nn.functional.cross_entropy(b.reshape(-1,7),bt.reshape(-1),reduction='none').reshape(len(obs),-1).mean(1)
    par=bt[:,PI];chosen=q.gather(2,par.unsqueeze(-1)).squeeze(-1);mask=(par>0).float();binary=(obs[:,CI]>0).float();bc=nn.functional.binary_cross_entropy(chosen,binary,reduction='none');loss=((ce+(bc*mask).sum(1)/mask.sum(1).clamp(min=1))*w).sum()
   loss.backward();nn.utils.clip_grad_norm_(self.net.parameters(),5);opt.step();self.net.eval()
   with torch.no_grad():
    train_mae=float(abs(self.net(A)[0]-T).mean());v=float(abs(self.predict(val_a)-val_y).mean()) if val_a is not None else train_mae
   history.append(dict(tag=tag,model=self.kind,epoch=e+1,loss=float(loss.detach()),train_mae=train_mae,validation_mae=v))
   if fixed_epochs is not None or v<best:self.best_epoch=e+1;best=v;beststate={k:t.detach().cpu().clone() for k,t in self.net.state_dict().items()}
  self.net.load_state_dict(beststate);self.history=history;self.params=sum(p.numel() for p in self.net.parameters());return self
 def predict(self,a):
  self.net.eval()
  with torch.no_grad():return np.clip(self.net(torch.tensor(a,dtype=torch.float32,device=DEVICE))[0].cpu().numpy(),0,6)
def basic(name,param=None,s=20261028):
 if name=='Ridge':return Ridge(alpha=param or 100)
 if name=='PLS':return PLSRegression(n_components=param or 5,scale=False)
 if name=='ET':return ExtraTreesRegressor(n_estimators=180,min_samples_leaf=2,max_features=.7,n_jobs=2,random_state=s)
 if name=='RF':return RandomForestRegressor(n_estimators=180,min_samples_leaf=2,max_features=.7,n_jobs=2,random_state=s)
 if name=='HGB':return MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,l2_regularization=10,random_state=s),n_jobs=1)
def q1():
 results=[];curves=[];preds={m:[] for m in C['q1_models']+['InnerSelected']};selected=[];params=[]
 for s in C['seeds']:
  f=folds(s);out={m:np.zeros_like(Y) for m in preds}
  for fold in range(3):
   te=np.flatnonzero(f==fold);tr=np.flatnonzero(f!=fold);it,iv=inner(tr,s+fold);pi=Prep().fit(X[it]);pt=Prep().fit(X[tr]);a,b=pi.transform(X[it]),pi.transform(X[iv]);aa,bb=pt.transform(X[tr]),pt.transform(X[te]);vscore={}
   for name in C['q1_models']:
    t=time.time();tag=f'{s}_{fold}'
    if name=='Mean':p=np.tile(Y[tr].mean(0),(len(te),1));v=abs(Y[it].mean(0)-Y[iv]).mean();par=None
    elif name in ['MLP','Hierarchy','HierarchyRater']:
     # Inner stopping chooses epoch; outer refit uses exactly that epoch count.
     model=Deep(name,s+fold).fit(a,Y[it],it,b,Y[iv],tag);v=abs(model.predict(b)-Y[iv]).mean();curves+=model.history;epochs=model.best_epoch;model=Deep(name,s+fold).fit(aa,Y[tr],tr,tag=tag+'_refit',fixed_epochs=epochs);p=model.predict(bb);par=model.params
     torch.save(model.net.state_dict(),O/f'models/{name}_{tag}.pt')
    else:
     candidates=C['q1_ridge_alphas'] if name=='Ridge' else C['q1_pls_components'] if name=='PLS' else [None];vv=[]
     for par in candidates:
      m=basic(name,par,s+fold).fit(a,Y[it]);vv.append(abs(np.clip(m.predict(b),0,6)-Y[iv]).mean())
     idx=int(np.argmin(vv));par=candidates[idx];v=vv[idx];model=basic(name,par,s+fold).fit(aa,Y[tr]);p=np.clip(model.predict(bb),0,6)
    out[name][te]=p;vscore[name]=float(v);results.append(dict(seed=s,fold=fold,model=name,n_train=len(tr),n_test=len(te),seconds=time.time()-t,**metric(Y[te],p)));params.append(dict(seed=s,fold=fold,model=name,parameter=par,validation_mae=v));log(f'Q1 {tag} {name} MAE={results[-1]["MAE"]:.4f}')
   winner=min(vscore,key=vscore.get);out['InnerSelected'][te]=out[winner][te];selected.append(dict(seed=s,fold=fold,model=winner,validation_mae=vscore[winner]));results.append(dict(seed=s,fold=fold,model='InnerSelected',n_train=len(tr),n_test=len(te),seconds=0,**metric(Y[te],out[winner][te])))
  for name in out:preds[name].append(out[name]);csv(pd.DataFrame(out[name],columns=targets).assign(beer=xdf.beer),f'q1/oof_{name}_{s}.csv')
 csv(results,'q1/fold_metrics.csv');csv(curves,'q1/training_curves.csv');csv(params,'q1/hyperparameters.csv');csv(selected,'q1/inner_selected.csv')
 per=[];summ=[]
 for name,pp in preds.items():
  for repeat,p in enumerate(pp):summ.append(dict(model=name,seed=C['seeds'][repeat],**metric(Y,p),record_mean_MAE=float(abs(p-Yrecord).mean()),rounded_panel_mean_accuracy=float((np.floor(p[:,AF]+.5)==np.floor(Y[:,AF]+.5)).mean())))
  p=np.mean(pp,0)
  for j,c in enumerate(targets):per.append(dict(model=name,target=c,MAE=float(abs(p[:,j]-Y[:,j]).mean()),RMSE=float(np.sqrt(np.mean((p[:,j]-Y[:,j])**2))),R2=float(r2_score(Y[:,j],p[:,j]))))
 csv(summ,'q1/repeat_metrics.csv');csv(per,'q1/per_target.csv');statistics(preds,Y,'q1','InnerSelected')
 p=np.mean(preds['InnerSelected'],0);csv(pd.DataFrame(np.clip(np.floor(p+.5),0,6).astype(int),columns=targets).assign(beer=xdf.beer),'q1/grades_oof_panel_mean.csv');csv(pd.DataFrame(p,columns=targets).assign(beer=xdf.beer),'q1/continuous_oof.csv')
 # Bootstrap raw individual panel uncertainty (no model fitting involved).
 rng=np.random.default_rng(20261028);panel=[]
 for i in range(250):
  rr=raw[raw.beer_name==i+1].groupby('taster')[targets].mean().to_numpy();boot=rr[rng.integers(len(rr),size=(500,len(rr)))].mean(1);lo,hi=np.quantile(boot,[.025,.975],axis=0)
  for j,c in enumerate(targets):panel.append(dict(beer=int(xdf.beer.iloc[i]),target=c,mean=Y[i,j],lower=lo[j],upper=hi[j],raters=len(rr)))
 csv(panel,'q1/panel_uncertainty.csv')
def statistics(preds,truth,folder,reference):
 rng=np.random.default_rng(428);err={k:abs(np.mean(v,0)-truth).mean(1) for k,v in preds.items()};rows=[]
 for m,e in err.items():
  if m==reference:continue
  d=e-err[reference];boot=d[rng.integers(len(d),size=(2000,len(d)))].mean(1);p=wilcoxon(d).pvalue if np.any(d) else 1
  rows.append(dict(comparator=m,reference=reference,positive_favors_reference=float(d.mean()),CI_low=float(np.quantile(boot,.025)),CI_high=float(np.quantile(boot,.975)),p=float(p)))
 order=np.argsort([r['p'] for r in rows]);last=0
 for k,i in enumerate(order):last=max(last,min(1,rows[i]['p']*(len(rows)-k)));rows[i]['p_holm']=last
 csv(rows,f'{folder}/paired_statistics.csv')


# --------------------------------------------------------------------------
# 来源: q1_complete.py
# --------------------------------------------------------------------------
"""拼接完整的3.xlsx评分表
输出53列结果
由 q1_acetaldehyde.py 计算。
每次运行都从原始化学数据重新计算乙醛分数
输出：
  q1/aroma_flavour_template_with_unobserved.csv    交付主表（53列）
  q1_complete/scorecard_complete.csv                仅50项模型评分，不含乙醛
  q1_complete/scorecard_complete_provenance.csv     每列数据来源说明
  q1_complete/acetaldehyde_graded.csv              各啤酒浓度、分数、推断依据
"""
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

R = Path(__file__).resolve().parents[2]
O = R / "outputs/rebuild"
OUTD = O / "q1_complete"
THRESHOLD_MG_L = 50.0  # 附件8的感官阈值，乙醛，阈值均值

# 3.xlsx 的 summary 工作表列顺序
SCORECARD_ORDER = [
    "A_malt_all", "A_malt_grain", "A_malt_bread", "A_malt_cara", "A_malt_burn",
    "A_hops_all", "A_hops_citrus", "A_hops_tropical", "A_hops_noble", "A_hops_woody",
    "A_esters_all", "A_esters_ethac", "A_esters_isoaa", "A_esters_flower", "A_ester_fruity",
    "F_malt_all", "F_malt_grain", "F_malt_bread", "F_malt_cara", "F_malt_burn",
    "F_hops_all", "F_hops_citrus", "F_hops_tropical", "F_hops_noble", "F_hops_woody",
    "F_esters_all", "F_esters_ethac", "F_esters_isoaa", "F_esters_flower", "F_ester_fruity",
    "body", "bitternes", "co2", "aftertaste", "alcohol", "sweetness", "acidity",
    "dms", "metallic", "4vg", "t2n", "diacetyl", "stale_hops",
    "orange", "coriander", "clove", "lactic", "acetic", "barnyard", "overall",
]
UNOBSERVED = ["A_acetaldehyde", "F_acetaldehyde"]

# 3.xlsx 列名 -> src/rebuild 全局使用的模型目标名
ALIAS = {"A_ester_fruity": "A_esters_fruity", "F_ester_fruity": "F_esters_fruity", "4vg": "X4vg"}

# 标准字段对应3.xlsx评分表的哪一列，就是单啤酒工作表标签
AROMA_SIDE = [
    "A_malt_all", "A_malt_grain", "A_malt_bread", "A_malt_cara", "A_malt_burn",
    "A_hops_all", "A_hops_citrus", "A_hops_tropical", "A_hops_noble", "A_hops_woody",
    "A_esters_all", "A_esters_ethac", "A_esters_isoaa", "A_esters_flower", "A_ester_fruity",
    "body", "bitternes", "co2", "aftertaste", "dms", "metallic", "4vg",
    "orange", "coriander", "clove", "overall",
]
FLAVOUR_SIDE = [
    "F_malt_all", "F_malt_grain", "F_malt_bread", "F_malt_cara", "F_malt_burn",
    "F_hops_all", "F_hops_citrus", "F_hops_tropical", "F_hops_noble", "F_hops_woody",
    "F_esters_all", "F_esters_ethac", "F_esters_isoaa", "F_esters_flower", "F_ester_fruity",
    "alcohol", "sweetness", "acidity", "t2n", "diacetyl", "stale_hops",
    "lactic", "acetic", "barnyard",
]
SIDE = {c: "Aroma" for c in AROMA_SIDE} | {c: "Flavour" for c in FLAVOUR_SIDE}
assert len(AROMA_SIDE) == 26 and len(FLAVOUR_SIDE) == 24 and set(SIDE) == set(SCORECARD_ORDER)


def acetaldehyde_grades():
#每次运行都从原始化学数据，重新计算基于阈值的乙醛评分
    x = pd.read_csv(find_input("1.csv"), usecols=["beer", "acetaldehyde"]).set_index("beer")
    conc = 10 ** x["acetaldehyde"]          # 附件1 is log10-transformed (data/processed/transforms.csv)
    grade = (conc > THRESHOLD_MG_L).astype(int)
    if (grade == 1).any():
#若有样本超过阈值，当前二值规则不再适用
#需要设计分级映射，直接抛出异常
        raise AssertionError(
            f"{int(grade.sum())} beers exceed {THRESHOLD_MG_L} mg/L; the threshold-anchored rule needs review")
    out = pd.DataFrame(
        {
            "beer": conc.index.to_numpy(),
            "acetaldehyde_mg_per_L": conc.to_numpy(),
            "A_acetaldehyde": grade.to_numpy(),
            "F_acetaldehyde": grade.to_numpy(),
            "basis": "附件1实测浓度经log10反变换后仍低于附件8感官检出阈值 50.0 mg/L；按3.xlsx图例 0=Not present 判定",
        }
    )
    assert out.shape == (250, 5)
    return out


def complete_scorecard(with_acetaldehyde=True):
#beer + 50个标准0-6分字段，外加两个基于阈值推断的乙醛列
    grades = pd.read_csv(O / "q1/grades_oof_panel_mean.csv")
    assert grades.shape == (250, 51), grades.shape
    assert "beer" in grades.columns and sorted(grades.beer.tolist()) == list(range(1, 251))

    missing = [c for c in SCORECARD_ORDER if ALIAS.get(c, c) not in grades.columns]
    assert not missing, f"scorecard fields absent from the frozen OOF grades: {missing}"

    out = pd.DataFrame({"beer": grades.beer.to_numpy()})
    for field in SCORECARD_ORDER:
        out[field] = grades[ALIAS.get(field, field)].to_numpy()

    if with_acetaldehyde:
        acet = acetaldehyde_grades().set_index("beer").reindex(out.beer.to_numpy())
        for field in UNOBSERVED:
            out[field] = acet[field].to_numpy()
        assert out[UNOBSERVED].notna().all().all()
        assert out[UNOBSERVED].isin(range(0, 7)).all().all()

    assert out.shape == (250, 51 if not with_acetaldehyde else 53)
    assert out[SCORECARD_ORDER].notna().all().all()
    assert out[SCORECARD_ORDER].isin(range(0, 7)).all().all()
    assert set(AROMA_SIDE) | set(FLAVOUR_SIDE) == set(SCORECARD_ORDER)
    return out


def main_complete():
    OUTD.mkdir(parents=True, exist_ok=True)
    acet = acetaldehyde_grades()
    acet.to_csv(OUTD / "acetaldehyde_graded.csv", index=False, encoding="utf-8")

    out = complete_scorecard(with_acetaldehyde=True)
    out.to_csv(O / "q1/aroma_flavour_template_with_unobserved.csv", index=False, encoding="utf-8-sig")
    out.to_csv(OUTD / "scorecard_complete.csv", index=False, encoding="utf-8-sig")

    prov = pd.DataFrame(
        {
            "column": SCORECARD_ORDER + UNOBSERVED,
            "scorecard_side": [SIDE[c] for c in SCORECARD_ORDER] + ["Aroma", "Flavour"],
            "kind": ["grade_0_6_panel_model"] * len(SCORECARD_ORDER)
            + ["grade_0_6_threshold_anchored"] * len(UNOBSERVED),
            "supervision": ["附件4 + 逐评分员原始记录，同评分员重复记录先平均，评分员间等权"] * len(SCORECARD_ORDER)
            + ["无 0-6 记录；由附件1实测浓度(log10反变换)对附件8感官检出阈值50.0mg/L判定，全部低于阈值故为0=Not present"] * len(UNOBSERVED),
            "is_panel_score": ["yes"] * len(SCORECARD_ORDER) + ["no"] * len(UNOBSERVED),
            "model_target": [ALIAS.get(c, c) for c in SCORECARD_ORDER] + ["", ""],
        }
    )
    assert len(prov) == 52
    prov.to_csv(OUTD / "scorecard_complete_provenance.csv", index=False, encoding="utf-8")
#依赖官方附件存在
    try:
        s = pd.read_csv(find_input("7.csv"), usecols=["beer", "aroma_acetaldehyde", "taste_acetaldehyde"])
    except FileNotFoundError:
        print("skip acetaldehyde_supplementary.csv: 附件7 not found")
    else:
        s = s.set_index("beer").reindex(out.beer.to_numpy())
        sup = pd.DataFrame(
            {
                "beer": out.beer.to_numpy(),
                "A_acetaldehyde_z": s["aroma_acetaldehyde"].to_numpy(),
                "F_acetaldehyde_z": s["taste_acetaldehyde"].to_numpy(),
            }
        )
        assert sup.shape == (250, 3) and sup.notna().all().all()
        sup.to_csv(OUTD / "acetaldehyde_supplementary.csv", index=False, encoding="utf-8")

    print(f"wrote q1/aroma_flavour_template_with_unobserved.csv rows={len(out)} columns={out.shape[1]} "
          f"(50 panel-modelled + 2 threshold-anchored + beer)")
    print(f"Aroma={len(AROMA_SIDE)} Flavour={len(FLAVOUR_SIDE)} graded={len(SCORECARD_ORDER)} "
          f"acetaldehyde_all_zero={bool((acet[UNOBSERVED].to_numpy() == 0).all())} "
          f"NaN cells={int(out.isna().sum().sum())}")


# --------------------------------------------------------------------------
# 来源: q1_acetaldehyde.py
# --------------------------------------------------------------------------
"""根据实测证据填充 A_acetaldehyde / F_acetaldehyde
已从公开数据确认：
  附件1的乙醛数据做了 log10 变换；还原后浓度区间 0.01–18.56 mg/L，中位数4.10。
  附件8中乙醛感官检出阈值为50.0 mg/L；所有样本都低于阈值，感官上无法检出。
  附件7的乙醛香气列大多取下限，和实测浓度几乎无关，只是有无标记，不是0–6强度分。
  官方相关矩阵里没有乙醛感官列，说明没有公开品酒员的乙醛打分。
对照题目说明“0 = 未检出”，所有啤酒合理评分为0。
该分数由浓度和检出阈值推导得出
结果单独输出并标注推断依据。"""
from pathlib import Path

from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

R = Path(__file__).resolve().parents[2]
O = R / "outputs/rebuild"
OUT = O / "q1_complete"
THRESHOLD_MG_L = 50.0  # 附件8 Sensory thresholds, acetaldehyde, threshold_mean


def main_acetaldehyde():
    OUT.mkdir(parents=True, exist_ok=True)
    x = pd.read_csv(find_input("1.csv"), usecols=["beer", "acetaldehyde"]).set_index("beer")
    conc = 10 ** x["acetaldehyde"]

    grade = np.where(conc > THRESHOLD_MG_L, 1, 0).astype(int)
    out = pd.DataFrame(
        {
            "beer": conc.index.to_numpy(),
            "acetaldehyde_mg_per_L": conc.to_numpy(),
            "A_acetaldehyde": grade,
            "F_acetaldehyde": grade,
            "basis": "附件1实测浓度经log10反变换后仍低于附件8感官检出阈值 50.0 mg/L；按3.xlsx图例 0=Not present 判定",
        }
    )
    assert out.shape == (250, 5)
    assert (conc <= THRESHOLD_MG_L).all(), "a beer exceeded the threshold; rule must be revisited"
    assert out[["A_acetaldehyde", "F_acetaldehyde"]].isin([0, 1]).all().all()
    out.to_csv(OUT / "acetaldehyde_graded.csv", index=False, encoding="utf-8")

    # 重新计算，保证论文中的数值可溯源
    s7 = pd.read_csv(find_input("7.csv"), usecols=["beer", "aroma_acetaldehyde", "taste_acetaldehyde"]).set_index("beer")
    s7 = s7.reindex(conc.index)
    floor = float(s7["aroma_acetaldehyde"].min())
    ev = dict(
        transform="log10 (data/processed/transforms.csv, log10_rmse=2.93e-10 over 249 pairs)",
        concentration_mg_per_L=dict(min=round(float(conc.min()), 3), median=round(float(conc.median()), 3),
                                    max=round(float(conc.max()), 3)),
        sensory_detection_threshold_mg_per_L=THRESHOLD_MG_L,
        n_beers_above_threshold=int((conc > THRESHOLD_MG_L).sum()),
        max_concentration_over_threshold_ratio=round(float(conc.max() / THRESHOLD_MG_L), 4),
        panel_floor_fraction_附件7_aroma=round(float((s7["aroma_acetaldehyde"] <= floor + 1e-9).mean()), 4),
        panel_floor_value_附件7_aroma=round(floor, 4),
        chemical_vs_panel_spearman=round(float(spearmanr(conc, s7["aroma_acetaldehyde"])[0]), 4),
        official_6csv_has_acetaldehyde_column=False,
        grade_rule="grade = 0 if C <= threshold else 1, then panel-scale mapping; here every beer is below threshold so grade = 0",
        caveat=(
            "This is a threshold-anchored inference, not a measured panel score. It answers the 0-6 request "
            "only in the degenerate sense that the whole dataset is below the sensory detection threshold, so "
            "every beer takes the 0 = Not present level. The paper reports it separately from the 50-panel "
            "columns and states that no per-taster 0-6 acetaldehyde record exists in any released file."
        ),
    )
    (OUT / "acetaldehyde_grading.json").write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"concentration {conc.min():.2f}-{conc.max():.2f} mg/L, threshold {THRESHOLD_MG_L} mg/L")
    print(f"beers above threshold: {(conc > THRESHOLD_MG_L).sum()}/250 -> grade all 0")
    print(f"panel floor fraction: {ev['panel_floor_fraction_附件7_aroma']:.3f}, chemical-vs-panel spearman {ev['chemical_vs_panel_spearman']:+.4f}")
    print("wrote", str(OUT.relative_to(R)) + "/acetaldehyde_graded.csv")


# --------------------------------------------------------------------------
# 来源: q1_grade_range.py
# --------------------------------------------------------------------------

R = Path(__file__).resolve().parents[2]
O = R / "outputs/rebuild"
OUT = O / "q1_complete"


def main_grade_range():
    OUT.mkdir(parents=True, exist_ok=True)
    off = pd.read_csv(R / "data/raw/4.csv")
    targets = list(off.columns[3:])
    raw = pd.read_csv(R / "data/supplementary/Raw sensory data anonymized.csv", encoding="cp1252").rename(
        columns={"A_ester_fruity": "A_esters_fruity", "F_ester_fruity": "F_esters_fruity"}
    )
    Y = raw.groupby("beer_name")[targets].mean().reindex(range(1, 251)).set_index(off.beer)

    grades = pd.read_csv(O / "q1/grades_oof_panel_mean.csv")
    cont = pd.read_csv(O / "q1/continuous_oof.csv")
    truth_grade = np.floor(Y[targets].to_numpy() + 0.5)
    pred_grade = grades[targets].to_numpy()
    pred_cont = cont[targets].to_numpy()

    rows = []
    for lo, hi, label in [(0, 2, "[0,2)"), (2, 3, "[2,3)"), (3, 4, "[3,4)"), (4, 5, "[4,5)"), (5, 7, "[5,7)")]:
        sel = (truth_grade >= lo) & (truth_grade < hi)
        n = int(sel.sum())
        if not n:
            continue
        rows.append(
            dict(
                truth_band=label,
                n_cells=n,
                truth_mean=float(truth_grade[sel].mean()),
                pred_grade_mean=float(pred_grade[sel].mean()),
                pred_continuous_mean=float(pred_cont[sel].mean()),
                grade_bias=float(pred_grade[sel].mean() - truth_grade[sel].mean()),
                continuous_bias=float(pred_cont[sel].mean() - truth_grade[sel].mean()),
                underpredicted_rate=float((pred_grade[sel] < truth_grade[sel]).mean()),
            )
        )
    band = pd.DataFrame(rows)
    band.to_csv(OUT / "grade_range_diagnostic.csv", index=False, encoding="utf-8")

## 连续预测值上限截断为6
    capped = dict(
        truth_cells_ge5=int((truth_grade >= 5).sum()),
        pred_cells_ge5=int((pred_grade >= 5).sum()),
        pred_cells_ge5_before_rounding=int((pred_cont >= 4.5).sum()),
        pred_continuous_max=float(pred_cont.max()),
        truth_grade_hist={str(k): int(v) for k, v in pd.Series(truth_grade.ravel()).value_counts().sort_index().items()},
        pred_grade_hist={str(k): int(v) for k, v in pd.Series(pred_grade.ravel()).value_counts().sort_index().items()},
        n_cells=int(truth_grade.size),
    )
    summary = dict(
        finding=(
            "The shipped scorecard never exceeds grade 4, while the rounded panel mean does reach 5 (112 cells) "
            "and 6 (9 cells). The compression is not a rounding artefact: the continuous predictions themselves "
            "are pulled toward the middle, so the model systematically under-predicts high-intensity samples and "
            "slightly over-predicts low ones. This is a genuine limitation of predicting a panel mean from "
            "chemistry with MAE ~0.45 on a 0-6 scale, and it must be reported rather than hidden."
        ),
        by_truth_band=band.to_dict("records"),
        counts=capped,
        consequence=(
            "Grades 5-6 are effectively unavailable in the deliverable, so any downstream use that depends on "
            "identifying intense samples (e.g. flagging a beer as strongly hoppy or strongly defective) will miss "
            "them. The 50 predicted columns are unbiased only in the aggregate sense of a small MAE, not in the "
            "tails."
        ),
    )
    (OUT / "grade_range_summary.json").write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(band.to_string(index=False))
    print()
    print("truth cells >=5:", capped["truth_cells_ge5"], "| predicted cells >=5:", capped["pred_cells_ge5"])
    print("continuous max:", round(capped["pred_continuous_max"], 4),
          "| continuous >=4.5 (would round to 5):", capped["pred_cells_ge5_before_rounding"])
    print("wrote", str(OUT.relative_to(R)) + "/grade_range_diagnostic.csv + grade_range_summary.json")


# --------------------------------------------------------------------------
# 来源: q1_scale_check.py
# --------------------------------------------------------------------------
"""将 ET_official_Z_scale 对照结果转换到和交付Q1结果一致的0-6尺度。
supplement.py 用附件4标准化数据训练ET模型，MAE也是在标准化空间计算的，数值不代表原始打分误差。
附件4是评委均值的仿射变换，可基于训练折学到的维度仿射映射还原回原始打分尺度，再和交付结果的MAE对比。
仅输出 scale_comparability.csv，不会改动 q1 目录下文件。"""
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTD = O / "q1_complete"


def main_scale_check():
    official = off[targets].to_numpy()
    rows = []
    for s in C["seeds"]:
        f = folds(s)
        for k in range(3):
            tr = np.flatnonzero(f != k)
            te = np.flatnonzero(f == k)
            pre = Prep().fit(X[tr])
            a = pre.transform(X[tr])

            direct = basic("ET", s=s).fit(a, Y[tr]).predict(pre.transform(X[te]))
            mae_direct = float(np.abs(direct - Y[te]).mean())

            z = basic("ET", s=s).fit(a, official[tr]).predict(pre.transform(X[te]))
            mae_z_units = float(np.abs(z - official[te]).mean())

            # Per-dimension affine map Z -> Y fitted on the training fold only.
            fitted = np.zeros_like(z)
            for j in range(len(targets)):
                yy = Y[tr][:, j]
                zz = official[tr][:, j]
                if np.std(zz) < 1e-12:
                    fitted[:, j] = yy.mean()
                else:
                    b = float(np.polyfit(zz, yy, 1)[0])
                    a0 = float(np.polyfit(zz, yy, 1)[1])
                    fitted[:, j] = a0 + b * z[:, j]
            fitted = np.clip(fitted, 0, 6)
            mae_z_mapped = float(np.abs(fitted - Y[te]).mean())
            rows.append(
                dict(
                    seed=s,
                    fold=k,
                    MAE_direct_score_units=mae_direct,
                    MAE_official_Z_units=mae_z_units,
                    MAE_official_Z_mapped_to_score=mae_z_mapped,
                    R2_official_Z_mapped=float(r2_score(Y[te], fitted)),
                    score_unit_scale=float(mae_z_mapped / mae_z_units) if mae_z_units else np.nan,
                )
            )
            print(f"seed {s} fold {k}: direct {mae_direct:.4f} | Z-units {mae_z_units:.4f} | Z->score {mae_z_mapped:.4f}")

    df = pd.DataFrame(rows)
    OUTD.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTD / "scale_comparability.csv", index=False, encoding="utf-8")
    print("\nmean over 6 outer folds:")
    print(df[["MAE_direct_score_units", "MAE_official_Z_units", "MAE_official_Z_mapped_to_score"]].mean().round(4).to_string())
    print(f"wrote q1_complete/scale_comparability.csv rows={len(df)}")


# --------------------------------------------------------------------------
# 来源: audit_extended.py
# --------------------------------------------------------------------------
def main_audit_extended():
 rows=[]
 for b,d in raw.groupby('beer_name'):
  if d.date.nunique()<2:continue
  for c in targets:
   g=[v[c].to_numpy() for _,v in d.groupby('date')];p=kruskal(*g).pvalue if d[c].nunique()>1 else 1;rows.append(dict(raw_beer_name=b,beer=int(xdf.beer.iloc[b-1]),target=c,n_dates=len(g),p=p))
 order=np.argsort([r['p'] for r in rows]);last=0
 for k,i in enumerate(order):last=max(last,min(1,rows[i]['p']*(len(rows)-k)));rows[i]['p_holm']=last
 csv(rows,'audit/repeated_sessions_tests.csv')

 names=pd.read_csv(R/'data/processed/chemical_names.csv').set_index('feature').matrix_name;mat=pd.read_csv(raw_path('2.csv'),index_col=0).loc[[names[c] for c in chem],[names[c] for c in chem]].to_numpy();sp=pd.DataFrame(X).corr(method='spearman').to_numpy();pe=pd.DataFrame(X).corr(method='pearson').to_numpy();js({'matrix_shape':list(mat.shape),'spearman_max_abs_error':float(np.nanmax(abs(mat-sp))),'pearson_mean_abs_error':float(np.nanmean(abs(mat-pe))),'mapping_unique':bool(names.is_unique)},'audit/correlation_recheck.json')

 tr=pd.read_csv(R/'data/processed/transforms.csv');g=pd.read_csv(O/'q3/generated_22styles_chemical231.csv');out={'style':g['style']};un=[]
 for _,r in tr.iterrows():
  if r['transform']=='log10':out[r.feature]=10**g[r.feature]
  elif str(r['transform']).startswith('identity'):out[r.feature]=g[r.feature]
  else:un.append(r.feature)
 csv(pd.DataFrame(out),'q3/generated_calibrated_physical.csv');csv(tr,'q3/physical_units_and_transforms.csv')

 complete_scorecard(with_acetaldehyde=True).to_csv(O/'q1/aroma_flavour_template_with_unobserved.csv',index=False,encoding='utf-8-sig')

 sp=pd.read_csv(O/'q3/splits.csv');rows=[]
 for s in C['q3_seeds']:
  ti=sp[(sp.seed==s)&(sp.role=='generator')].row.to_numpy();vi=sp[(sp.seed==s)&(sp.role=='validation')].row.to_numpy();p=Prep().fit(X[ti]);a=p.transform(X[vi]);loc=np.unravel_index(np.argmax(abs(a)),a.shape);rows.append(dict(seed=s,beer=int(xdf.beer.iloc[vi[loc[0]]]),feature=chem[loc[1]],largest_abs_training_scaled_value=float(abs(a[loc])),raw_value=float(X[vi[loc[0]],loc[1]])))
 csv(rows,'audit/validation_extremes.csv');log('extended audits complete')

if __name__=='__main__':
 if len(sys.argv)==1:q1()
 elif 'audit_extended' in sys.argv:main_audit_extended()
 elif 'q1_acetaldehyde' in sys.argv:main_acetaldehyde()
 elif 'q1_complete' in sys.argv:main_complete()
 elif 'q1_grade_range' in sys.argv:main_grade_range()
 elif 'q1_scale_check' in sys.argv:main_scale_check()
