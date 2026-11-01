
import networkx as nx


from q1 import *  # 数据层：X / Y / targets / chem / style / C / O / Prep / metric 等
from pathlib import Path
F = O / 'closed'
METHODS = C['q2_methods']
def put(d, p): pd.DataFrame(d).to_csv(F / p, index=False, encoding='utf-8-sig')
def dump(d, p): (F / p).write_text(json.dumps(d, ensure_ascii=False, indent=2, default=lambda v: v.item() if hasattr(v, 'item') else str(v)), encoding='utf-8')
def freeze():
 '''(removed) 哈希指纹闸门，保留空实现以兼容调用点。'''

"""问题二：化合物共表达图与 Top-3 关键因素筛选

本文件由以下模块合并而成，对应论文一个问题的完整可复现代码：
  - q2.py
  - q2_closed.py
  - closed_tables.py
"""

from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform


# --------------------------------------------------------------------------
# 来源: q2.py
# --------------------------------------------------------------------------
EXCLUDE={'color.EBC.','pH','bitterness','CO2.PSI.','GFS','unfermentables','kcalperc','betaglucan.mg.L.','protein.g.L.','esters_sum','aroma_hops_sum','acids_sum','sulfur_sum'}
allowed=np.array([j for j,c in enumerate(chem) if c not in EXCLUDE])
def unit(a):
 a=a-a.mean(0);return a/np.maximum(np.sqrt((a*a).sum(0)),1e-12)
def select(a,y,method):
 z=unit(rankdata(a[:,allowed],axis=0));t=unit(rankdata(y,axis=0));rho=z.T@t;result=[];scores=[]
 for j in range(y.shape[1]):
  chosen=[];ss=[]
  for k in range(3):
   if method=='Spearman' or (method=='Hybrid2plus1' and k<2):score=abs(rho[:,j])
   elif method=='Redundancy':score=abs(rho[:,j])-(.5*np.max(abs(z.T@z[:,chosen]),axis=1) if chosen else 0)
   elif not chosen:score=rho[:,j]**2
   else:
    b=z[:,chosen];inv=np.linalg.inv(b.T@b+.02*np.eye(k));xr=z-b@inv@(b.T@z);yr=t[:,j]-b@inv@(b.T@t[:,j]);score=(xr.T@yr)**2/(np.sum(xr*xr,axis=0)+.02)
   score=score.copy();score[chosen]=-np.inf;ii=int(np.argmax(score));chosen.append(ii);ss.append(float(score[ii]))
  result.append(allowed[chosen]);scores.append(ss)
 return np.array(result),np.array(scores)
def residual(a,sty):
 b=a.copy()
 for st in np.unique(sty):ix=sty==st;b[ix]-=b[ix].mean(0)
 return b

def main_outline():
 rows=[];sr=[];preds={m:[] for m in C['q2_methods']+['Forward3_styleAdjusted']}
 for s in C['seeds']:
  f=folds(s);out={m:np.zeros_like(Y) for m in preds}
  for k in range(3):
   tr=np.flatnonzero(f!=k);te=np.flatnonzero(f==k);pre=Prep().fit(X[tr]);a,b=pre.transform(X[tr]),pre.transform(X[te])
   for m in preds:
    ar,yr=(residual(a,style[tr]),residual(Y[tr],style[tr])) if m.endswith('styleAdjusted') else (a,Y[tr]);idx,score=select(ar,yr,'Forward3' if m.endswith('styleAdjusted') else m)
    for j in range(50):
     model=Ridge(alpha=8).fit(a[:,idx[j]],Y[tr,j]);out[m][te,j]=np.clip(model.predict(b[:,idx[j]]),0,6)
     for rank,q in enumerate(idx[j]):sr.append(dict(seed=s,fold=k,method=m,target=targets[j],rank=rank+1,chemical=chem[q],score=score[j,rank]))
    rows.append(dict(seed=s,fold=k,model=m,**metric(Y[te],out[m][te])));log(f'Q2 {s}_{k} {m} {rows[-1]["MAE"]:.4f}')
  for m in out:preds[m].append(out[m]);csv(pd.DataFrame(out[m],columns=targets).assign(beer=xdf.beer),f'q2/oof_{m}_{s}.csv')
 csv(rows,'q2/fold_metrics.csv');csv(sr,'q2/fold_selections.csv');statistics(preds,Y,'q2','Forward3')
 names=pd.read_csv(R/'data/processed/chemical_names.csv').set_index('feature').matrix_name;matrix=pd.read_csv(raw_path('2.csv'),index_col=0);mapped=[names[c] for c in chem];rho=matrix.loc[mapped,mapped].to_numpy(float)
 G=nx.Graph();G.add_nodes_from(range(231));edges=[]
 for i in range(231):
  for j in range(i):
   if abs(rho[i,j])>=.65:G.add_edge(i,j,weight=float(abs(rho[i,j])),rho=float(rho[i,j]));edges.append(dict(source=chem[i],target=chem[j],rho=rho[i,j],sign='positive' if rho[i,j]>0 else 'negative'))
 csv(edges,'q2/signed_edges.csv');nx.write_graphml(G,O/'q2/network.graphml');groups=fcluster(linkage(squareform(np.clip(1-abs(rho),0,1),checks=False),method='complete'),.2,criterion='distance');csv(pd.DataFrame({'chemical':chem,'alternative_group':groups}),'q2/alternative_groups.csv')
 a=Prep().fit(X).transform(X);rng=np.random.default_rng(C['seeds'][0]);boots={m:[] for m in C['q2_methods']}
 for it in range(C['q2_bootstraps']):
  ix=np.concatenate([rng.choice(np.flatnonzero(style==st),sum(style==st),replace=True) for st in np.unique(style)]);aa=Prep().fit(X[ix]).transform(X[ix])
  for m in boots:boots[m].append(select(aa,Y[ix],m)[0])
  if (it+1)%20==0:log(f'Q2 bootstrap {it+1}')
 np.savez_compressed(O/'q2/bootstrap_selections.npz',**{k:np.array(v) for k,v in boots.items()});top=[];stability=[];compare=[];finals={m:select(a,Y,m) for m in boots};adj,_=select(residual(a,style),residual(Y,style),'Forward3');rawrho=unit(rankdata(a,axis=0)).T@unit(rankdata(Y,axis=0));adjrho=unit(rankdata(residual(a,style),axis=0)).T@unit(rankdata(residual(Y,style),axis=0))
 for m,(idx,score) in finals.items():
  bs=np.array(boots[m])
  for j,t in enumerate(targets):
   jac=[];gjac=[]
   for b in range(len(bs)-1):
    u,v=set(bs[b,j]),set(bs[b+1,j]);gu,gv=set(groups[list(u)]),set(groups[list(v)]);jac.append(len(u&v)/len(u|v));gjac.append(len(gu&gv)/len(gu|gv))
   stability.append(dict(method=m,target=t,compound_jaccard=np.mean(jac),group_jaccard=np.mean(gjac),within_top3_abs_rho=float(np.mean([abs(rho[idx[j,u],idx[j,v]]) for u,v in [(0,1),(0,2),(1,2)]]))))
   for rank,q in enumerate(idx[j]):top.append(dict(method=m,target=t,rank=rank+1,chemical=chem[q],name_in_2csv=names[chem[q]],selection_score=score[j,rank],spearman=rawrho[q,j],style_adjusted_rank_association=adjrho[q,j],frequency=float((bs[:,j,:]==q).any(1).mean()),alternative_group=int(groups[q]),group_frequency=float((groups[bs[:,j,:]]==groups[q]).any(1).mean()),in_style_adjusted_forward=bool(q in adj[j]),label_flag='suspected formula alias' if t=='barnyard' else ''))
   for m2,(i2,_) in finals.items():
    u,v=set(idx[j]),set(i2[j]);compare.append(dict(method1=m,method2=m2,target=t,jaccard=len(u&v)/len(u|v)))
 csv(top,'q2/top3_all_methods.csv');csv([r for r in top if r['method']=='Forward3'],'q2/top3_50x3.csv');csv(stability,'q2/stability.csv');csv(compare,'q2/method_jaccard.csv');js({'n_nodes':231,'edges':len(edges),'excluded':sorted(EXCLUDE),'candidate_count':len(allowed),'final_rule':'Prespecified Forward3; all alternatives reported, not selected by outer test.'},'q2/summary.json')


# --------------------------------------------------------------------------
# 来源: q2_closed.py
# --------------------------------------------------------------------------
def paired(rows,truth,reference=None):
 rng=np.random.default_rng(3481);loss={k:abs(v-truth).mean(1) for k,v in rows.items()};out=[]
 pairs=[(k,reference) for k in rows if k!=reference] if reference else list(itertools.combinations(rows,2))
 for m,n in pairs:
  d=loss[m]-loss[n];b=d[rng.integers(len(d),size=(2000,len(d)))].mean(1);out.append(dict(method=m,reference=n,loss_difference=d.mean(),CI_low=np.quantile(b,.025),CI_high=np.quantile(b,.975),p=wilcoxon(d).pvalue if np.any(d) else 1))
 holm(out);return out
def holm(rows):
 last=0
 for k,i in enumerate(np.argsort([r['p'] for r in rows])):last=max(last,min(1,rows[i]['p']*(len(rows)-k)));rows[i]['p_holm']=last

def fit_predict(indices,test,method):
 pre=Prep().fit(X[indices]);a=pre.transform(X[indices]);b=pre.transform(X[test]);idx,scores=select(a,Y[indices],method);p=np.zeros((len(test),50))
 for j in range(50):p[:,j]=np.clip(Ridge(alpha=8).fit(a[:,idx[j]],Y[indices,j]).predict(b[:,idx[j]]),0,6)
 return p,idx,scores

def choose(indices,s):
 unique=np.unique(indices);it,iv=inner(unique,s);tr=indices[np.isin(indices,it)];va=indices[np.isin(indices,iv)];scores={}
 assert not set(tr)&set(va)
 for m in METHODS:scores[m]=float(abs(fit_predict(tr,va,m)[0]-Y[va]).mean())
 return min(METHODS,key=lambda m:scores[m]),scores

def q2_closed():
 freeze();foldrows=[];choices=[];sel=[];preds={m:[] for m in METHODS+['TrainSelected']}
 for s in C['seeds']:
  f=folds(s);out={m:np.zeros_like(Y) for m in preds}
  for k in range(3):
   tr=np.flatnonzero(f!=k);te=np.flatnonzero(f==k);winner,sc=choose(tr,s+k);choices.append(dict(seed=s,fold=k,winner=winner,**sc))
   for m in METHODS:
    p,ix,score=fit_predict(tr,te,m);out[m][te]=p
    for j,t in enumerate(targets):
     for rank,v in enumerate(ix[j]):sel.append(dict(seed=s,fold=k,method=m,target=t,rank=rank+1,chemical=chem[v],score=score[j,rank]))
   out['TrainSelected'][te]=out[winner][te]
   for m in out:foldrows.append(dict(seed=s,fold=k,method=m,n_test=len(te),**metric(Y[te],out[m][te])))
  for m in out:preds[m].append(out[m]);put(pd.DataFrame(out[m],columns=targets).assign(beer=xdf.beer),f'q2/oof_{m}_{s}.csv')
  log(f'Closed Q2 outer seed {s} complete')
 put(foldrows,'q2/fold_metrics.csv');put(choices,'q2/method_choice_inner.csv');put(sel,'q2/fold_selections.csv');repeat=[]
 for m,pp in preds.items():
  for s,p in zip(C['seeds'],pp):repeat.append(dict(method=m,seed=s,**metric(Y,p)))
 put(repeat,'q2/repeat_metrics.csv');put(paired({m:np.mean(p,0) for m,p in preds.items()},Y,'TrainSelected'),'q2/paired_tests.csv')

 winner,final_inner=choose(np.arange(250),C['seeds'][0]);a=Prep().fit(X).transform(X);finals={m:select(a,Y,m) for m in METHODS};finals['TrainSelected']=finals[winner];groups=pd.read_csv(O/'q2/alternative_groups.csv').alternative_group.to_numpy();rho=pd.DataFrame(X).corr(method='spearman').to_numpy();assoc=unit(rankdata(a,axis=0)).T@unit(rankdata(Y,axis=0));adjassoc=unit(rankdata(residual(a,style),axis=0)).T@unit(rankdata(residual(Y,style),axis=0));rng=np.random.default_rng(C['seeds'][0]);boots={m:[] for m in finals};bw=[]
 for b in range(100):
  ids=np.concatenate([rng.choice(np.flatnonzero(style==st),sum(style==st),replace=True) for st in ST]);aa=Prep().fit(X[ids]).transform(X[ids]);pick,score=choose(ids,C['seeds'][0]+b);bw.append(dict(bootstrap=b,winner=pick,**score))
  for m in METHODS:boots[m].append(select(aa,Y[ids],m)[0])
  boots['TrainSelected'].append(boots[pick][-1])
  if (b+1)%20==0:log(f'Closed Q2 grouped bootstrap {b+1}/100')
 put(bw,'q2/bootstrap_method_choice.csv');np.savez_compressed(F/'q2/bootstrap_selections.npz',**{m:np.array(v) for m,v in boots.items()});top=[];stability=[];cross=[]
 for m,(ix,score) in finals.items():
  bs=np.array(boots[m])
  for j,t in enumerate(targets):
   #固定100个独立对，只采样一次
   # 各方法采用相同的协议
   pairlist=[(b,(b+37)%100) for b in range(100)];cj=[];gj=[]
   for u,v in pairlist:
    su,sv=set(bs[u,j]),set(bs[v,j]);gu,gv=set(groups[list(su)]),set(groups[list(sv)]);cj.append(len(su&sv)/len(su|sv));gj.append(len(gu&gv)/len(gu|gv))
   redundancy=float(np.mean([abs(rho[ix[j,u],ix[j,v]]) for u,v in [(0,1),(0,2),(1,2)]]));stability.append(dict(method=m,target=t,compound_jaccard=np.mean(cj),group_jaccard=np.mean(gj),within_top3_abs_rho=redundancy))
   for rank,v in enumerate(ix[j]):
    fixed=np.array(boots[winner]);top.append(dict(method=m,final_fixed_method=winner if m=='TrainSelected' else m,target=t,rank=rank+1,chemical=chem[v],selection_score=score[j,rank],spearman=assoc[v,j],style_adjusted_rank_association=adjassoc[v,j],bootstrap_frequency=float((bs[:,j,:]==v).any(1).mean()),alternative_group=int(groups[v]),group_frequency=float((groups[bs[:,j,:]]==groups[v]).any(1).mean()),final_fixed_method_frequency=float((fixed[:,j,:]==v).any(1).mean()),label_flag='suspected formula alias' if t=='barnyard' else ''))
   for m2,(v,_) in finals.items():u0,v0=set(ix[j]),set(v[j]);cross.append(dict(method=m,other=m2,target=t,jaccard=len(u0&v0)/len(u0|v0)))
 put(top,'q2/top3_all_methods.csv');final=pd.DataFrame([r for r in top if r['method']=='TrainSelected']);put(final,'q2/top3_50x3.csv');put(stability,'q2/stability.csv');put(cross,'q2/method_jaccard.csv')
 old=O/'q2/top3_50x3.csv';archive=F/'q2/previous_forward3_top3.csv'
 if not archive.exists():archive.write_bytes(old.read_bytes())
 final.to_csv(old,index=False,encoding='utf-8-sig')
 scores=pd.DataFrame(repeat).groupby('method')[['MAE','RMSE','R2']].agg(['mean','std']);scores.columns=['_'.join(v) for v in scores.columns];put(scores.reset_index(),'q2/summary_metrics.csv');dump({'final_method':winner,'final_inner_scores':final_inner,'rows':len(final),'evaluation_method':'TrainSelected outer OOF; final fixed algorithm is selected inside full-data training only','bootstrap':'100 style-stratified beer bootstraps; duplicate source beer cannot cross inner training/validation'},'q2/summary.json')
 log('Closed experiment 1/3 complete')

if __name__=='__main__':
 if len(sys.argv)==1:q2_closed()
