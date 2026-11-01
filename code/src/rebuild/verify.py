from q1 import *
from q2 import *
from q3 import *
def main():
 checks=[]
 def check(name,ok,detail=''):checks.append(dict(check=name,passed=bool(ok),detail=detail))
 manifest=pd.read_csv(O/'audit/raw_manifest.csv')
 for _,r in manifest.iterrows():check('raw unchanged '+r.file,hashlib.sha256((R/r.file).read_bytes()).hexdigest()==r.sha256)
 for s in C['seeds']:
  for m in C['q1_models']+['InnerSelected']:
   p=pd.read_csv(O/f'q1/oof_{m}_{s}.csv')[targets].to_numpy();check(f'OOF {m} {s}',p.shape==(250,50) and np.isfinite(p).all() and p.min()>=0 and p.max()<=6)
 q1=pd.read_csv(O/'q1/fold_metrics.csv');check('Q1 60 model-fold rows',len(q1)==60);check('Q1 equal outer train size',q1.groupby(['seed','fold']).n_train.nunique().max()==1)
 top=pd.read_csv(O/'q2/top3_50x3.csv');check('Q2 50x3',len(top)==150 and top.target.nunique()==50 and top.groupby('target').chemical.nunique().eq(3).all());check('Q2 100 bootstrap',np.load(O/'q2/bootstrap_selections.npz')['Forward3'].shape==(100,50,3))
 sp=pd.read_csv(O/'q3/splits.csv')
 for s,d in sp.groupby('seed'):check(f'Q3 disjoint {s}',len(d)==250 and d.row.nunique()==250 and sorted(d.groupby('role').size())==[35,35,40,140])
 g=pd.read_csv(O/'q3/generated_22styles_chemical231.csv');check('Q3 880 spectra finite',len(g)==880 and np.isfinite(g[chem].to_numpy()).all() and g['style'].nunique()==22)
 p=pd.read_csv(O/'q3/generated_sensory50.csv');check('Q3 portraits bounded',len(p)==880 and p[targets].min().min()>=0 and p[targets].max().max()<=6)
 d=pd.read_csv(O/'q3/defect_checks.csv');v=d[(d.method!='Unconstrained')&d.source.notna()];check('observed encoded constraints',np.all(v.generated<=v.source+1e-10),'Not a physical guarantee for floor values')
 m=pd.read_csv(O/'q3/migration_metrics.csv');check('risk design constraint',m.loc[m.method=='UncertaintyConstraint','design_risk'].le(1e-10).all())
 diagnostics={'independent_stale_increases_by_method':m.assign(increase=m.independent_stale_change>1e-10).groupby('method').increase.sum().to_dict(),'q3_source_missing_rows':int(d.source.isna().sum()),'q3_floor_unverifiable_rows':int((d.physical_nonincrease_status=='unverifiable_floor_mass').sum())}
 for f in ['01_data','02_q1_accuracy','03_q1_curves','04_q2_selection','05_q2_network','06_q3_quality','07_migration','08_generated_portraits','09_sensitivity']:
  for ext in ['png','pdf','svg']:check(f'Figure {f}.{ext}',(O/f'figures/{f}.{ext}').stat().st_size>1000)
 for f in ['00_data_audit.md','01_q1_analysis.md','02_q2_analysis.md','03_q3_analysis.md','04_innovation_and_ablation.md','05_final_summary.md']:check('Report '+f,(R/'reports'/f).stat().st_size>500)
 check('Paper PDF exists',(R/'paper/competition_draft.pdf').exists());check('AI details PDF exists',(R/'reports/AI工具使用详情说明.pdf').exists());csv(checks,'verification.csv');js(diagnostics,'verification_diagnostics.json');js({'checks':len(checks),'passed':sum(c['passed'] for c in checks),'all_passed':all(c['passed'] for c in checks)},'verification.json')
 src=[{'file':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((R/'src/rebuild').glob('*.py'))];csv(src,'audit/source_manifest.csv');print(json.dumps({'checks':len(checks),'passed':sum(c['passed'] for c in checks)},ensure_ascii=False));assert all(c['passed'] for c in checks),[c for c in checks if not c['passed']]
if __name__=='__main__':main()
