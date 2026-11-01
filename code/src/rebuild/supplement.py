from q3 import Generator,ST
from scipy.stats import wilcoxon
from q1 import *
from q2 import *
from q3 import *
import joblib

def main():
 rows=[];rng=np.random.default_rng(901);official=off[targets].to_numpy();predstd=np.zeros_like(Y)
 for s in C['seeds']:
  f=folds(s)
  for k in range(3):
   tr=np.flatnonzero(f!=k);te=np.flatnonzero(f==k);pre=Prep().fit(X[tr]);a=pre.transform(X[tr]);m=basic('ET',s=s).fit(a,Y[tr]);p=m.predict(pre.transform(X[te]));rows.append(dict(seed=s,fold=k,experiment='ET_equal_rater',**metric(Y[te],p)))
   for frac in [.05,.10,.20]:
    z=X[te].copy();z[rng.random(z.shape)<frac]=np.nan;p=m.predict(pre.transform(z));rows.append(dict(seed=s,fold=k,experiment=f'ET_mask_{frac}',**metric(Y[te],p)))
   rec=basic('ET',s=s).fit(a,Yrecord[tr]);rows.append(dict(seed=s,fold=k,experiment='ET_record_training_equal_eval',**metric(Y[te],rec.predict(pre.transform(X[te])))))
   keep=[j for j,c in enumerate(targets) if c!='barnyard'];m49=basic('ET',s=s).fit(a,Y[tr][:,keep]);p49=m49.predict(pre.transform(X[te]));rows.append(dict(seed=s,fold=k,experiment='ET_without_barnyard',MAE=float(abs(p49-Y[te][:,keep]).mean()),RMSE=float(np.sqrt(np.mean((p49-Y[te][:,keep])**2))),R2=float(r2_score(Y[te][:,keep],p49)),AF_MAE=float(abs(p49[:,:30]-Y[te,:30]).mean()),MAE49=float(abs(p49-Y[te][:,keep]).mean())))
   mz=basic('ET',s=s).fit(a,official[tr]);pz=mz.predict(pre.transform(X[te]));rows.append(dict(seed=s,fold=k,experiment='ET_official_Z_scale',**metric(official[te],pz)))
 csv(rows,'q1/sensitivity.csv')
 pre=Prep().fit(X);model=basic('ET').fit(pre.transform(X),Y);joblib.dump({'preprocessor':pre,'model':model,'targets':targets,'features':chem,'scope':'full refit descriptive, scores are not independent evaluation'},O/'models/ET_full.joblib')
#使用提供的元数据标注所有因子
 top=pd.read_csv(O/'q2/top3_all_methods.csv');meta=pd.read_csv(R/'data/processed/thresholds.csv').rename(columns={'field_name_R':'chemical'});ann=top.merge(meta,on='chemical',how='left',validate='many_to_one');csv(ann,'q2/top3_chemical_annotations.csv')
#用验证集均值，不使用测试集结果
 val=pd.read_csv(O/'q3/validation_metrics.csv');winner=val.groupby('model').SWD.mean().idxmin();genmodel=Generator(winner,20261028).fit(X,style);gen=np.vstack([genmodel.sample(st,40,rng) for st in ST]);csv(pd.DataFrame(gen,columns=chem).assign(style=np.repeat(ST,40)),'q3/generated_22styles_chemical231.csv');csv(pd.DataFrame(np.clip(model.predict(pre.transform(gen)),0,6),columns=targets).assign(style=np.repeat(ST,40)),'q3/generated_sensory50.csv')
 summ=json.loads((O/'q3/summary.json').read_text());summ['production_model']=winner;summ['production_selection']='Minimum mean validation SWD; resolves three-way validation vote tie without test metrics.';js(summ,'q3/summary.json')
#仅对单独核验过的校准特征做物理量换算
 trans=pd.read_csv(R/'data/processed/transforms.csv').set_index('feature');th=pd.read_csv(R/'data/processed/thresholds.csv').set_index('field_name_R');dc=pd.read_csv(O/'q3/defect_checks.csv');out=[]
 for _,r in dc.iterrows():
  c=r.chemical;t=trans.loc[c];threshold=th.loc[c,'threshold_mean'] if c in th.index else np.nan;v=10**r.generated if t['transform']=='log10' else r.generated if str(t['transform']).startswith('identity') else np.nan;b=10**r.source if t['transform']=='log10' else r.source if str(t['transform']).startswith('identity') else np.nan;d=r.to_dict();d.update(transform=t['transform'],unit=t['unit'],source_converted=b,generated_converted=v,reference_threshold=threshold,generated_above_reference=bool(v>threshold) if np.isfinite(threshold) else None,threshold_interpretation='sensory detection reference, not legal/safety/acceptability limit');out.append(d)
 csv(out,'q3/defect_checks_physical.csv')
 mg=pd.read_csv(O/'q3/migration_metrics.csv');stat=[]
 for m,df in mg.groupby('method'):
#以啤酒为重采样单元；计算置信区间前先合并重复划分
  d=df.assign(prob_change=df.target_probability_after-df.target_probability_before).groupby('beer')[['prob_change','independent_stale_change']].mean()
  for measure in d:
   v=d[measure].to_numpy();b=v[rng.integers(len(v),size=(2000,len(v)))].mean(1);stat.append(dict(method=m,measure=measure,unique_beers=len(v),mean=v.mean(),CI_low=np.quantile(b,.025),CI_high=np.quantile(b,.975),p=wilcoxon(v).pvalue if np.any(v) else 1))
 order=np.argsort([r['p'] for r in stat]);last=0
 for k,i in enumerate(order):last=max(last,min(1,stat[i]['p']*(len(stat)-k)));stat[i]['p_holm']=last
 csv(stat,'q3/migration_statistics.csv')
 fs=[];styles=[]
 splits=pd.read_csv(O/'q3/splits.csv')
 for s in C['q3_seeds']:
  tr=splits[(splits.seed==s)&(splits.role=='generator')].row.to_numpy();te=splits[(splits.seed==s)&(splits.role=='test')].row.to_numpy();pr=Prep().fit(X[tr]);low=np.nanmin(X[tr],0);eligible=np.mean(X[tr]==low,0)>=.25;real=(pr.imp.transform(X[te])[:,eligible]==low[eligible]).astype(float);co=real.T@real/len(real)
  for name in C['q3_models']:
   g=pd.read_csv(O/f'q3/test_generated_{name}_{s}.csv');binary=(g[chem].to_numpy()[:,eligible]==low[eligible]).astype(float);cg=binary.T@binary/len(binary);mask=np.triu(np.ones(co.shape,bool),1);fs.append(dict(seed=s,model=name,joint_floor_probability_MAE=float(abs(cg[mask]-co[mask]).mean()),n_floor_variables=int(sum(eligible))))
   for st in ST:
    ix=style[te]==st
    if sum(ix):styles.append(dict(seed=s,model=name,style=st,n_test=int(sum(ix)),marginal_W1=float(np.mean([wasserstein_distance(pr.transform(X[te][ix])[:,j],pr.transform(g.loc[ix,chem].to_numpy())[:,j]) for j in range(231)]))))
 csv(fs,'q3/floor_joint_diagnostics.csv');csv(styles,'q3/per_style_metrics.csv')
 log('supplement diagnostics complete')
if __name__=='__main__':main()
