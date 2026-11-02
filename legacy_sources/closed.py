"""AI-assisted: OpenAI Codex gpt-6.1-sol, 2026-10-28. Exactly three closed supplements."""
from run import *
from pathlib import Path
import joblib,itertools
from q2 import select,unit,residual,allowed
from q3 import Generator,ST,SI
F=O/'closed'
METHODS=C['q2_methods']
def put(d,p):pd.DataFrame(d).to_csv(F/p,index=False,encoding='utf-8-sig')
def dump(d,p):(F/p).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def frozen_files():
 return [R/'src/rebuild/run.py',R/'reports/01_q1_analysis.md',R/'configs/rebuild_protocol.json',O/'q3/splits.csv']+list((O/'q1').glob('*'))
def freeze():
 files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_files() if p.is_file()}
 protocol={'closed_experiments':['Q2 four-method nested selection and bootstrap','Q3 22-style assessment of existing four models','Q3 four-policy migration on nine unique held-out sources'],'frozen_hashes':files,'Q1':'No retraining or output rewriting','Q2':'Original two seeds and three outer folds; global method chosen by equal-target raw-mean MAE in original inner 20% holdout; no added candidates; alpha8; resample styles100 times; bootstrap duplicates kept in one inner role. Final full refit repeats the same inner-choice rule with first seed.','generation':'Original 140/35/40/35 roles and four configurations. 40 generated per style per context; same latent random-seed reset across models; all22styles listed even if n_test=0; rank correlation NA if n_test<4; common eligible correlation pairs for comparability; shrinkage and floors determined on generator training only. Validation distances cannot be reinterpreted as test evidence.','migration':'Nine unique beers from prior held-out source set, first eligible original seed per beer; same fixed independent evaluator within each context, serialized before optimization; original model selected in existing selected.csv. 80 candidates, four alphas and identity; four policies Unconstrained/PointConstraint/UncertaintyConstraint/Reject; all evaluation occurs after candidate choices.','inference':'2000 paired beer bootstrap; Wilcoxon Holm correction; nine sources, no duplicate counting; intervals describe internal sample variation, not external validation.','stop_rule':'After these three groups, stop all model exploration regardless of results.'}
 p=R/'configs/closed_final_protocol.json'
 if p.exists():
  old=json.loads(p.read_text(encoding='utf-8'));assert old['frozen_hashes']==files,'Frozen Q1 or original split changed'
 else:p.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
 log('closed protocol frozen; Q1 hashes recorded')
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
 # Final refit uses exactly the same inner method-choice algorithm as outer evaluation.
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
   # Fixed 100 independent pairs, sampled once; equal protocol across methods.
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

def eval_bundle(ev,s):
 pre=Prep().fit(X[ev]);a=pre.transform(X[ev]);clf=ExtraTreesClassifier(n_estimators=180,min_samples_leaf=2,max_features=.7,n_jobs=2,random_state=s).fit(a,style[ev]);reg=ExtraTreesRegressor(n_estimators=180,min_samples_leaf=2,max_features=.7,n_jobs=2,random_state=s).fit(a,Y[ev]);rng=np.random.default_rng(s+610);boot=[]
 for k in range(12):
  ids=rng.integers(len(ev),size=len(ev));boot.append(ExtraTreesRegressor(n_estimators=60,min_samples_leaf=3,max_features=.7,n_jobs=2,random_state=k+s).fit(a[ids],Y[ev[ids]]))
 return {'pre':pre,'classifier':clf,'regressor':reg,'bootstrap':boot,'training_rows':ev,'seed':s,'scope':'Independent evaluation only; optimization never receives this bundle'}

def conditional_metrics(real,g,model,evaluator,st,scale_projection):
 pre=model.pre;b=pre.transform(g);n=len(real);pred=evaluator['classifier'].predict(evaluator['pre'].transform(g));classes=list(evaluator['classifier'].classes_);prob=evaluator['classifier'].predict_proba(evaluator['pre'].transform(g))[:,classes.index(st)]
 nn=cdist(b,pre.transform(model.a))/np.sqrt(231);d=dict(n_test=n,style_accuracy=float(np.mean(pred==st)),style_score=float(prob.mean()),nearest_train_RMS=float(nn.min(1).mean()),copy_fraction=float((nn.min(1)<1e-8).mean()),diversity=float(np.std(b,axis=0).mean()),SWD=np.nan,marginal_W1=np.nan,correlation_MAE=np.nan,correlation_pairs=0,floor_mass_MAE=np.nan,joint_floor_MAE=np.nan,correlation_status='n_test<4' if n<4 else 'computed')
 if not n:d['correlation_status']='no independent test beer';return d
 a=pre.transform(real);d['SWD']=float(np.mean([wasserstein_distance(a@scale_projection[:,k],b@scale_projection[:,k]) for k in range(48)]));d['marginal_W1']=float(np.mean([wasserstein_distance(a[:,j],b[:,j]) for j in range(231)]))
 lo=model.low;ids=model.floor;ra=(pre.imp.transform(real)[:,ids]<=lo[ids]+1e-10).astype(float);gb=(g[:,ids]<=lo[ids]+1e-10).astype(float);d['floor_mass_MAE']=float(abs(ra.mean(0)-gb.mean(0)).mean());mask=np.triu(np.ones((sum(ids),sum(ids)),bool),1);d['joint_floor_MAE']=float(abs(ra.T@ra/n-gb.T@gb/len(g))[mask].mean())
 if n>=4:
  ca=pd.DataFrame(a).corr(method='spearman').to_numpy();cg=pd.DataFrame(b).corr(method='spearman').to_numpy();mask=np.triu(np.ones((231,231),bool),1)&np.isfinite(ca)&np.isfinite(cg);d['correlation_pairs']=int(mask.sum());d['correlation_MAE']=float(abs(ca[mask]-cg[mask]).mean()) if mask.any() else np.nan
 return d


def finalize_generation_metrics():
 sp=pd.read_csv(O/'q3/splits.csv');path=F/'generation/metrics_22styles_by_context.csv';df=pd.read_csv(path)
 for s in C['q3_seeds']:
  tr=sp[(sp.seed==s)&(sp.role=='generator')].row.to_numpy(int);te=sp[(sp.seed==s)&(sp.role=='test')].row.to_numpy(int);pre=Prep().fit(X[tr]);files={m:pd.read_csv(F/f'generation/generated_{m}_{s}.csv') for m in C['q3_models']}
  for st in ST:
   ids=te[style[te]==st]
   if len(ids)<4:continue
   real=pd.DataFrame(pre.transform(X[ids])).corr(method='spearman').to_numpy();cor={m:pd.DataFrame(pre.transform(g.loc[g['style']==st,chem].to_numpy())).corr(method='spearman').to_numpy() for m,g in files.items()};mask=np.triu(np.ones((231,231),bool),1)&np.isfinite(real)
   for ca in cor.values():mask&=np.isfinite(ca)
   for m,ca in cor.items():
    ix=(df.seed==s)&(df['style']==st)&(df.model==m);df.loc[ix,'correlation_MAE']=float(abs(ca[mask]-real[mask]).mean()) if mask.any() else np.nan;df.loc[ix,'correlation_pairs']=int(mask.sum());df.loc[ix,'correlation_status']='common variable pairs across all four generators'
 put(df,'generation/metrics_22styles_by_context.csv');summ=df.groupby(['model','style'])[['SWD','marginal_W1','style_accuracy','style_score','correlation_MAE','joint_floor_MAE','floor_mass_MAE','nearest_train_RMS','copy_fraction','diversity']].agg(['mean','std']);summ.columns=['_'.join(v) for v in summ.columns];put(summ.reset_index(),'generation/metrics_22styles_summary.csv')
 macro=summ.groupby(level=0).mean();macro.columns=['_'.join(v) if isinstance(v,tuple) else v for v in macro.columns];put(macro.reset_index(),'generation/macro_summary.csv');log('Common-pair correlations finalized without model refitting')

def generation_closed():
 freeze();sp=pd.read_csv(O/'q3/splits.csv');rows=[];evrows=[];manifest=[];curves=[];rng=np.random.default_rng(801);proj=rng.normal(size=(231,48));proj/=np.linalg.norm(proj,axis=0)
 for s in C['q3_seeds']:
  roles={role:sp[(sp.seed==s)&(sp.role==role)].row.to_numpy(int) for role in ['generator','validation','evaluator','test']};tr,va,ev,te=[roles[k] for k in ['generator','validation','evaluator','test']];e=eval_bundle(ev,s);joblib.dump(e,F/f'models/evaluator_{s}.joblib');evpred=e['classifier'].predict(e['pre'].transform(X[te]));evrows.append(dict(seed=s,n_evaluator=len(ev),n_test=len(te),real_accuracy=float(np.mean(evpred==style[te]))));manifest.append(dict(seed=s,training_rows=ev.tolist(),sha256=hashlib.sha256((F/f'models/evaluator_{s}.joblib').read_bytes()).hexdigest()))
  for m in C['q3_models']:
   gen=Generator(m,s).fit(X[tr],style[tr],X[va],style[va]);joblib.dump(gen,F/f'models/generator_{m}_{s}.joblib');allg=[]
   for k,st in enumerate(ST):
    g=gen.sample(st,40,np.random.default_rng(s+100+k));allg.append(pd.DataFrame(g,columns=chem).assign(style=st));real=X[te[style[te]==st]];d=conditional_metrics(real,g,gen,e,st,proj);rows.append(dict(seed=s,model=m,style=st,n_train_style=int(sum(style[tr]==st)),n_evaluator_style=int(sum(style[ev]==st)),**d))
   put(pd.concat(allg,ignore_index=True),f'generation/generated_{m}_{s}.csv')
   for h in gen.history:curves.append(dict(seed=s,model=m,**h))
   log(f'Closed Q3 22-style generation {s} {m}')
 put(rows,'generation/metrics_22styles_by_context.csv');put(evrows,'generation/evaluator_real_accuracy.csv');put(curves,'generation/training_curves.csv');dump(manifest,'models/evaluator_manifest.json')
 df=pd.DataFrame(rows);summary=df.groupby(['model','style'])[['SWD','marginal_W1','style_accuracy','style_score','correlation_MAE','joint_floor_MAE','floor_mass_MAE','nearest_train_RMS','copy_fraction','diversity']].agg(['mean','std']);summary.columns=['_'.join(v) for v in summary.columns];put(summary.reset_index(),'generation/metrics_22styles_summary.csv');coverage=[]
 for st in ST:
  ids=sp[(sp.role=='test')&(sp['style']==st)].row.unique();coverage.append(dict(style=st,n_total=int(sum(style==st)),n_unique_test=len(ids),n_contexts_with_test=int(df[(df.model==C['q3_models'][0])&(df['style']==st)].n_test.gt(0).sum()),n_correlation_contexts=int(df[(df.model==C['q3_models'][0])&(df['style']==st)].correlation_pairs.gt(0).sum()),distribution_status='test evidence available' if len(ids) else 'NA: no held-out beer in frozen protocol'))
 put(coverage,'generation/style_coverage.csv');dump({'rows':len(df),'styles':len(ST),'models':C['q3_models'],'scope':'Same original roles. No-test styles retain NA distribution evidence; n<4 correlations not estimated. Style classifier scoring is diagnostic, never generator selection. Existing selected.csv remains frozen.'},'generation/summary.json');finalize_generation_metrics();log('Closed experiment 2/3 complete')


def migration_closed():
 freeze();sp=pd.read_csv(O/'q3/splits.csv');prior=pd.read_csv(O/'q3/migration_metrics.csv');sources=prior.groupby('beer').seed.min().to_dict();assert len(sources)==9
 chosen=pd.read_csv(O/'q3/selected.csv').set_index('seed').model.to_dict();rows=[];spectra=[];check=[];source_table=[];choices_log=[];sj=targets.index('stale_hops');dc=['DMS','X2.3.butanedione','X4VG','X.E..2.nonenal'];dj=[chem.index(v) for v in dc];bundle_hashes=[]
 for s in C['q3_seeds']:
  tr=sp[(sp.seed==s)&(sp.role=='generator')].row.to_numpy(int);ev=sp[(sp.seed==s)&(sp.role=='evaluator')].row.to_numpy(int);te=sp[(sp.seed==s)&(sp.role=='test')].row.to_numpy(int);pre=Prep().fit(X[tr]);a=pre.transform(X[tr]);gen=joblib.load(F/f'models/generator_{chosen[s]}_{s}.joblib');epath=F/f'models/evaluator_{s}.joblib';h0=hashlib.sha256(epath.read_bytes()).hexdigest();design=[];rng=np.random.default_rng(s)
  for k in range(12):
   ii=rng.integers(len(tr),size=len(tr));design.append(ExtraTreesRegressor(n_estimators=60,min_samples_leaf=3,max_features=.7,n_jobs=2,random_state=s+k).fit(a[ii],Y[tr[ii]]))
  joblib.dump({'pre':pre,'models':design,'train_rows':tr},F/f'models/design_ensemble_{s}.joblib')
  for beer,context in sorted(sources.items()):
   if context!=s:continue
   i=int(np.flatnonzero(xdf.beer.to_numpy()==beer)[0]);assert i in te and i not in tr and i not in ev
   source=gen.pre.imp.transform(X[[i]])[0];cand0=gen.sample('Tripel',80,np.random.default_rng(s+int(beer)));target=gen.pre.transform(cand0).mean(0);zs=gen.pre.transform(source[None])[0];base_design=np.array([m.predict(pre.transform(source[None]))[0,sj] for m in design]);selections={};risks={};costs={}
   for policy in ['Unconstrained','PointConstraint','UncertaintyConstraint','Reject']:
    if policy=='Reject':selections[policy]=source.copy();risks[policy]=0.;costs[policy]=float(np.linalg.norm(zs-target));continue
    cand=np.vstack([source]+[source+alpha*(cand0-source) for alpha in [.25,.5,.75,1]])
    if policy!='Unconstrained':
     for j in dj:
      if np.isfinite(X[i,j]):cand[:,j]=np.minimum(cand[:,j],X[i,j])
    changes=np.array([m.predict(pre.transform(cand))[:,sj]-base_design[k] for k,m in enumerate(design)]);risk=changes.mean(0) if policy=='PointConstraint' else np.quantile(changes,.9,axis=0);cost=np.linalg.norm(gen.pre.transform(cand)-target,axis=1)+.15*np.linalg.norm(gen.pre.transform(cand)-zs,axis=1);valid=np.ones(len(cand),bool) if policy=='Unconstrained' else risk<=1e-10;cost[~valid]=np.inf;k=int(np.argmin(cost));selections[policy]=cand[k];risks[policy]=float(risk[k]);costs[policy]=float(cost[k]);choices_log.append(dict(beer=beer,seed=s,policy=policy,selected_candidate=k,feasible_candidates=int(valid.sum()),budget=len(cand)))
   # Only now load and query the independent evaluator, after all four choices are fixed.
   e=joblib.load(epath);b=e['pre'].transform(source[None]);tj=list(e['classifier'].classes_).index('Tripel');bp=float(e['classifier'].predict_proba(b)[0,tj]);by=float(e['regressor'].predict(b)[0,sj]);bb=np.array([m.predict(b)[0,sj] for m in e['bootstrap']]);source_table.append(dict(beer=beer,row=i,seed=s,n_generator=len(tr),n_evaluator=len(ev),source_role='test',generator=chosen[s],evaluator_sha256=h0))
   for policy,out in selections.items():
    z=e['pre'].transform(out[None]);prob=float(e['classifier'].predict_proba(z)[0,tj]);yy=float(e['regressor'].predict(z)[0,sj]);boot=np.array([m.predict(z)[0,sj] for m in e['bootstrap']])-bb;diff=gen.pre.transform(out[None])[0]-zs;unknown=int(np.isnan(X[i,dj]).sum());atfloor=int(np.sum(np.isfinite(X[i,dj])&(X[i,dj]<=gen.low[dj]+1e-10)));chemical_inc=int(sum(np.isfinite(X[i,j]) and out[j]>X[i,j]+1e-10 for j in dj));refused=bool(np.linalg.norm(out-source)<1e-10)
    rows.append(dict(beer=beer,seed=s,policy=policy,target_score_before=bp,target_score_after=prob,target_gain=prob-bp,stale_before=by,stale_after=yy,stale_change=yy-by,independent_stale_p90=float(np.quantile(boot,.9)),independent_stale_increase=yy-by>1e-10,design_risk=risks[policy],change_RMS=float(np.sqrt(np.mean(diff**2))),changed_features=int(np.sum(abs(out-source)>1e-10)),distance_before=float(np.linalg.norm(zs-target)),distance_after=float(np.linalg.norm(gen.pre.transform(out[None])[0]-target)),refused=refused,chemical_observed_increases=chemical_inc,missing_markers=unknown,floor_markers=atfloor,candidate_cost=costs[policy]))
    spectra.append(dict(beer=beer,seed=s,policy=policy,**dict(zip(chem,out))));
    for c,j in zip(dc,dj):check.append(dict(beer=beer,seed=s,policy=policy,defect=c,source=X[i,j],generated=out[j],encoded_nonincrease=bool(out[j]<=X[i,j]+1e-10) if np.isfinite(X[i,j]) else None,physical_status='unobserved_source' if not np.isfinite(X[i,j]) else 'floor_below_unknown' if X[i,j]<=gen.low[j]+1e-10 else 'observed_encoding_order',risk_kind='direct observed chemical'))
    check.append(dict(beer=beer,seed=s,policy=policy,defect='stale_hops',source=by,generated=yy,encoded_nonincrease=yy<=by+1e-10,physical_status='no unique chemical threshold; independent prediction only',risk_kind='independent sensory proxy'))
   log(f'Closed Q3 source beer={beer} four-policy comparison complete')
  assert hashlib.sha256(epath.read_bytes()).hexdigest()==h0;bundle_hashes.append(dict(seed=s,sha256=h0,unchanged=True))
 put(rows,'migration/per_beer.csv');put(spectra,'migration/spectra.csv');put(check,'migration/defect_checks.csv');put(source_table,'migration/nine_sources.csv');put(choices_log,'migration/candidate_choices.csv');dump(bundle_hashes,'migration/evaluator_hash_check.json');df=pd.DataFrame(rows);rng=np.random.default_rng(541);summary=[];test=[]
 for policy,d in df.groupby('policy'):
  d=d.sort_values('beer');assert len(d)==9 and d.beer.nunique()==9
  for col in ['target_gain','stale_change','independent_stale_p90','change_RMS','refused','independent_stale_increase','chemical_observed_increases']:
   v=d[col].to_numpy(float);boot=v[rng.integers(9,size=(2000,9))].mean(1);summary.append(dict(policy=policy,metric=col,n_beers=9,mean=float(v.mean()),CI_low=float(np.quantile(boot,.025)),CI_high=float(np.quantile(boot,.975))))
  if policy!='Reject':
   for col in ['target_gain','stale_change']:
    v=d[col].to_numpy();test.append(dict(policy=policy,reference='Reject',metric=col,mean_difference=float(v.mean()),p=wilcoxon(v).pvalue if np.any(v) else 1))
 holm(test);put(summary,'migration/summary.csv');put(test,'migration/paired_tests.csv');dump({'n_unique_sources':9,'policies':4,'rows':len(df),'independent_evaluator':'serialized and unchanged; per original context, same bundle for all four policies','stop_model_exploration':True},'migration/summary.json');log('Closed experiment 3/3 complete; model exploration stopped')


if __name__=='__main__':
 if '--freeze' in sys.argv:freeze()
 elif '--q2' in sys.argv:q2_closed()
 elif '--generation' in sys.argv:generation_closed()
 elif '--migration' in sys.argv:migration_closed()
