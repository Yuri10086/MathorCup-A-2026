"""三问共用：汇总表与跨问题统计

本文件处理三个问题共同产出的表格：
  - q2/        冲突对、稳定性汇总、bootstrap 选择频率
  - generation/ 共同变量对支持
  - migration/  物理单位换算、验收/拒绝统计、建模收束标记

它不包含任何一题的建模逻辑，因此单列为"三问共用"，与 q1.py / q2.py / q3.py 并列。
来源: closed_tables.py
"""

from q1 import *
from q2 import *
from q3 import *

# --------------------------------------------------------------------------
# 来源: closed_tables.py
# --------------------------------------------------------------------------
def main():
 finalize_generation_metrics()
 names=pd.read_csv(R/'data/processed/chemical_names.csv').rename(columns={'feature':'chemical'});meta=pd.read_csv(R/'data/processed/thresholds.csv').rename(columns={'field_name_R':'chemical'});t=pd.read_csv(F/'q2/top3_50x3.csv');t=t[['method','final_fixed_method','target','rank','chemical','selection_score','spearman','style_adjusted_rank_association','bootstrap_frequency','alternative_group','group_frequency','final_fixed_method_frequency','label_flag']].merge(names,on='chemical',how='left',validate='many_to_one').merge(meta,on='chemical',how='left',validate='many_to_one');put(t,'q2/top3_50x3.csv');t.to_csv(O/'q2/top3_50x3.csv',index=False,encoding='utf-8-sig')
 # 不改变结果的情况下区分流程不确定性与固定方法的稳定性
 stab=pd.read_csv(F/'q2/stability.csv').groupby('method')[['compound_jaccard','group_jaccard','within_top3_abs_rho']].mean().reset_index();put(stab,'q2/stability_summary.csv');bw=pd.read_csv(F/'q2/bootstrap_method_choice.csv');put(bw.winner.value_counts().rename_axis('method').reset_index(name='count'),'q2/bootstrap_choice_summary.csv')
 # 汇总符合条件的关联对,记录哪些风格支持
 df=pd.read_csv(F/'generation/metrics_22styles_by_context.csv');put(df[df.correlation_pairs>0][['seed','style','n_test','model','correlation_pairs','correlation_MAE']],'generation/correlation_support.csv')
 # 添加经过测量的物理单位转换
 d=pd.read_csv(F/'migration/defect_checks.csv');trans=pd.read_csv(R/'data/processed/transforms.csv').set_index('feature');th=pd.read_csv(R/'data/processed/thresholds.csv').set_index('field_name_R');out=[]
 for _,v in d.iterrows():
  rr=v.to_dict();c=v.defect
  if c=='stale_hops':rr.update(unit='sensory score',source_physical=v.source,generated_physical=v.generated,threshold_mean=np.nan,threshold_median=np.nan,conversion='not a unique chemical')
  else:
   tr=trans.loc[c];tt=str(tr['transform']);convert=lambda x:10**x if tt=='log10' else x if tt.startswith('identity') else np.nan;rr.update(unit=tr.unit,source_physical=convert(v.source),generated_physical=convert(v.generated),threshold_mean=th.loc[c,'threshold_mean'] if c in th.index else np.nan,threshold_median=th.loc[c,'threshold_median'] if c in th.index else np.nan,conversion=tt)
  out.append(rr)
 put(out,'migration/defect_checks_physical.csv');m=pd.read_csv(F/'migration/per_beer.csv');accepted=[]
 for policy,g in m.groupby('policy'):
  a=g[~g.refused];accepted.append(dict(policy=policy,n_attempted=len(g),n_accepted=len(a),rejection_rate=g.refused.mean(),all_beer_target_gain=g.target_gain.mean(),accepted_target_gain=a.target_gain.mean() if len(a) else np.nan,all_beer_risk_rate=g.independent_stale_increase.mean(),accepted_risk_rate=a.independent_stale_increase.mean() if len(a) else np.nan))
 put(accepted,'migration/accepted_and_all.csv');dump({'modeling_status':'stopped','completed_groups':3,'new_model_families':0,'Q1_status':'frozen','next_stage':'paper and submission packaging only'},'modeling_closed.json');log('Final evidence tables complete; no additional fitting')

if __name__=='__main__':main()
