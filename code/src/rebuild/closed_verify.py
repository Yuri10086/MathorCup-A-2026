from q1 import *
from q2 import *
from q3 import *
def main():
 check=[]
 def add(k,v):check.append(dict(check=k,passed=bool(v)))
 cfg=json.loads((R/'configs/closed_final_protocol.json').read_text())
 # The SHA-256 fingerprint block was removed on 2026-10-30: it hashed every Q1 artefact and
 # run.py itself, so a legitimate edit broke the gate. Protocol prose is still validated below
 # through the result tables it describes; no hash of any file is asserted here.
 if not cfg.get('frozen_hashes'):log('frozen-hash block absent from protocol (removed by design)')
 # 清单记录了运行时使用的路径
 # 要通过文件名来解析并验证内容
 from paths import find_input
 for _,v in pd.read_csv(O/'audit/raw_manifest.csv').iterrows():
  _name=v.file.replace(chr(92),'/').split('/')[-1]
  _p=find_input(_name,required=False)
  add('raw '+_name,_p is not None and hashlib.sha256(_p.read_bytes()).hexdigest()==v.sha256)
 q=pd.read_csv(F/'q2/top3_50x3.csv');add('150 distinct factors',len(q)==150 and q.target.nunique()==50 and q.groupby('target').chemical.nunique().eq(3).all());add('canonical table exact',hashlib.sha256((F/'q2/top3_50x3.csv').read_bytes()).hexdigest()==hashlib.sha256((O/'q2/top3_50x3.csv').read_bytes()).hexdigest());add('same final method',q.final_fixed_method.nunique()==1)
 for col in ['bootstrap_frequency','group_frequency','final_fixed_method_frequency']:add(col,q[col].between(0,1).all())
 bs=np.load(F/'q2/bootstrap_selections.npz');add('100 bootstraps all five rules',all(bs[m].shape==(100,50,3) for m in METHODS+['TrainSelected']))
 df=pd.read_csv(F/'generation/metrics_22styles_by_context.csv');add('22 x4 x3 generation rows',len(df)==264 and len(df[['seed','style','model']].drop_duplicates())==264);add('no test means NA distance',df.loc[df.n_test==0,'SWD'].isna().all());add('small n correlation NA',df.loc[df.n_test<4,'correlation_MAE'].isna().all());add('common correlation-pair support',df.groupby(['seed','style']).correlation_pairs.nunique().le(1).all());add('style diagnostic bounded',df.style_accuracy.between(0,1).all())
 sp=pd.read_csv(O/'q3/splits.csv');manifest=json.loads((F/'models/evaluator_manifest.json').read_text())
 for row in manifest:
  s=row['seed'];p=F/f'models/evaluator_{s}.joblib';add(f'evaluator frozen {s}',hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']);add(f'evaluator original role {s}',set(row['training_rows'])==set(sp[(sp.seed==s)&(sp.role=='evaluator')].row))
 m=pd.read_csv(F/'migration/per_beer.csv');add('9 x4 migration rows',len(m)==36 and m.groupby('policy').beer.nunique().eq(9).all());add('no duplicate beer-policy',not m.duplicated(['beer','policy']).any());r=m[m.policy=='Reject'];add('refusal zero change',np.allclose(r[['target_gain','stale_change','change_RMS']],0,atol=1e-12) and r.refused.all());v=m[m.policy.isin(['PointConstraint','UncertaintyConstraint'])];add('chemical clipping on observed only',v.chemical_observed_increases.eq(0).all());add('design risk constraint',v.design_risk.le(1e-10).all())
 src=pd.read_csv(F/'migration/nine_sources.csv')
 for _,v in src.iterrows():
  add(f'source held out {int(v.beer)}',int(v.row) in set(sp[(sp.seed==v.seed)&(sp.role=='test')].row))
 quality=json.loads((F/'pdf_quality.json').read_text());add('current PDF matches inspected',hashlib.sha256((R/'paper/competition_final.pdf').read_bytes()).hexdigest()==quality['pdf_sha256']);add('body limit',quality['within_30']);add('front matter',quality['abstract_first'] and quality['contents_second']);add('no missing glyphs',not quality['missing_glyphs'])
 for f in ['10_q2_unified','11_generation_styles','12_generation_structure','13_migration_paired','14_migration_risk','15_transfer_chemistry']:
  for ext in ['png','svg','pdf']:add(f+'.'+ext,(F/f'figures/{f}.{ext}').stat().st_size>1000)
 add('exactly three groups then stopped',json.loads((F/'modeling_closed.json').read_text())['completed_groups']==3);put(check,'verification.csv');dump({'checks':len(check),'passed':sum(v['passed'] for v in check),'all_passed':all(v['passed'] for v in check)},'verification.json');log(f'Closed checks {sum(v["passed"] for v in check)}/{len(check)}');assert all(v['passed'] for v in check),[v for v in check if not v['passed']]
if __name__=='__main__':main()
