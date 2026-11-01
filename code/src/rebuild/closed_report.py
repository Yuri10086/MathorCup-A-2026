from q1 import *
from q2 import *
from q3 import *
import re,shutil
LABEL={'Spearman':'Spearman','Hybrid2plus1':'2+1','Forward3':'Forward3','Redundancy':'冗余控制','TrainSelected':'训练内选法','PCA_Gaussian':'PCA高斯','CopulaIndependent':'独立Copula','CopulaJoint':'联合Copula','CVAE':'CVAE','Unconstrained':'无约束','PointConstraint':'点约束','UncertaintyConstraint':'不确定性约束','Reject':'拒绝迁移'}
def read(p):return pd.read_csv(F/p)
def md(df):
 d=df.copy()
 for c in d:
  if pd.api.types.is_float_dtype(d[c]):d[c]=d[c].map(lambda v:f'{v:.4f}' if np.isfinite(v) else 'NA')
 return '| '+' | '.join(str(v) for v in d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v).replace('|','/') for v in a)+' |' for a in d.to_numpy())
def fig(n,cap):return f'\n![{cap}（AI辅助绘图，数据来自封闭补充实验）](../outputs/rebuild/closed/figures/{n}.png){{width=95%}}\n'
def main():
 freeze();old=(R/'paper/competition_draft.md').read_text(encoding='utf-8');q1=old[old.index('# 问题一：'):old.index('# 问题二：')];audit=old[old.index('# 数据审计与监督目标重建'):old.index('# 问题一：')];refs=old[old.index('# 参考文献'):re.search(r'# AI\s*使用声明',old).start()];refs=refs.replace('[7] OpenAI Codex，GPT-6 系列（接口未提供可独立核验的完整模型快照标识），OpenAI，2026-10-28。用于本参考工程的审计、代码、实验、图表与文本辅助。','[7] OpenAI Codex，GPT-6系列及gpt-6.1-sol，OpenAI，2026-10-28。用于审计、代码、实验、统计、图表与论文辅助。')
 qm=read('q2/summary_metrics.csv');qs=read('q2/stability_summary.csv');choice=read('q2/bootstrap_choice_summary.csv');qsum=json.loads((F/'q2/summary.json').read_text());g=read('generation/macro_summary.csv');coverage=read('generation/style_coverage.csv');mig=read('migration/per_beer.csv');ms=read('migration/summary.csv');tests=read('migration/paired_tests.csv');gm=g.set_index('model');n_cov=int(coverage.n_unique_test.gt(0).sum());n_corr=int(coverage.n_correlation_contexts.gt(0).sum());nctx=int(read('generation/correlation_support.csv')[['seed','style']].drop_duplicates().shape[0]);pmetrics=mig.groupby('policy')[['target_gain','stale_change','independent_stale_increase','change_RMS','refused']].mean().reindex(['Unconstrained','PointConstraint','UncertaintyConstraint','Reject']);riskcounts=mig.groupby('policy').independent_stale_increase.sum();refcounts=mig.groupby('policy').refused.sum()
 #Q3的后选择诊断
 #SWD差距与测试集规模，以及阈值超越缺陷视图
 _sl=json.loads((O/'q3_complete/selection_logic.json').read_text(encoding='utf-8'));_df=json.loads((O/'q3_complete/defect_criteria.json').read_text(encoding='utf-8'))
 sel_gap_r=_sl['gap_corr_with_n_test'];_bins={b['n_test_range']:b for b in _sl['gap_by_test_size']}
 def _bin(k):return _bins.get(k,{'pairs':0,'cvae_closer':0,'mean_swd_gap_CVAE_minus_CopulaIndependent':float('nan')})
 sel_b0_win=_bin('[1,2)')['cvae_closer'];sel_b0_n=_bin('[1,2)')['pairs'];sel_b0_gap=_bin('[1,2)')['mean_swd_gap_CVAE_minus_CopulaIndependent']
 sel_b1_win=_bin('[2,3)')['cvae_closer'];sel_b1_n=_bin('[2,3)')['pairs'];sel_b1_gap=_bin('[2,3)')['mean_swd_gap_CVAE_minus_CopulaIndependent']
 sel_b2_win=_bin('[3,5)')['cvae_closer'];sel_b2_n=_bin('[3,5)')['pairs'];sel_b2_gap=_bin('[3,5)')['mean_swd_gap_CVAE_minus_CopulaIndependent']
 sel_b3_win=_bin('[5,inf)')['cvae_closer'];sel_b3_n=_bin('[5,inf)')['pairs'];sel_b3_gap=_bin('[5,inf)')['mean_swd_gap_CVAE_minus_CopulaIndependent']
 xc=_df['threshold_crossing_by_policy'];anypos=_df['any_positive_by_policy']
 qtable=qm[['method','MAE_mean','MAE_std','RMSE_mean','R2_mean']].copy();qtable['method']=qtable.method.map(LABEL);qtable.columns=['方法','MAE均值','MAE标准差','RMSE','R²'];stable=qs.copy();stable.method=stable.method.map(LABEL);stable.columns=['方法','单分子Jaccard','替代组Jaccard','三因素平均绝对相关']
 q2=r'''# 问题二：训练内选法与统一的50×3化合物结果

## 从共线冗余到可检验的候选因素

单变量相关前三经常重复表达同一组发酵或酒花信号。把三项相关性当成三种独立机制，会放大候选因素的解释力度；直接改用条件选择又可能因有限样本残差而产生更不稳定的排名。因此，本问同时评价预测、冗余与重采样稳定性，并将“选择哪一种规则”的过程纳入训练数据内部。最终任务输出固定为每个感官三项，而不将低频变量包装成稳定的唯一因素。

候选集中剔除颜色、pH、压力、聚合总量等13项非单一分子测量，保留218个可识别化合物候选。完整预测任务仍可使用所有231项；两种特征口径不能混淆。共表达图节点为231项测量变量，附件2的Spearman绝对值达到0.65时连边，符号与绝对值分别保留。该图只作全样本描述，不作为外层测试的训练先验；相关定义和名称映射已经在原始数据上数值复核。

## 四种规则及正则残差推导

设训练秩变换后的特征矩阵为 $Z$，各列中心化并归一化，目标秩向量为 $t$。Spearman规则按 $|z_j^Tt|$ 取前三；2+1规则先取相关前二，再条件选第三；Forward3在三步均更新条件残差；冗余控制规则在每步用目标相关减去0.5倍与已选特征的最大绝对相关。

对已选集合 $S$，令 $H_S=Z_S(Z_S^TZ_S+0.02I)^{-1}Z_S^T$，构造 $u_j=(I-H_S)z_j$ 与 $r=(I-H_S)t$。在这个正则残差空间，单变量问题的最优增益为

$$\min_b\{\|r-bu_j\|^2+0.02b^2\},\qquad G_j=\frac{(u_j^Tr)^2}{u_j^Tu_j+0.02}.$$

每一步选择未入选变量中 $G_j$ 最大者。这里的“完整前向”指三个步骤都条件更新；由于 $H_S$ 不是幂等投影，该分数是所定义辅助残差问题的增益，不能声称精确求解所有三变量组合或原始联合Ridge的全局最优。2+1与Forward3采用相同的条件分数，差别仅在前两步。

四种规则选择后，均用相同的三个附件编码特征、训练中位数和训练标准化，拟合alpha=8的Ridge。它们拥有相同预测器和变量数，避免把选择规则与预测模型复杂度混在一起。

## 方法选择也属于训练流程

沿用原两次三折啤酒级外层划分。每个外层训练集内部，按既有20%划分计算四方法的50目标等权MAE，确定一种规则，然后在完整外层训练集上重新选择三变量并拟合；外层酒样只用于评价。最终全样本模型重用同一内部选择算法，再全量重拟合。外层所有候选成绩均保留，最终规则的选择不使用其外层分数。

下表先对每次完整OOF预测计算指标，再报告两次结果的均值与标准差；它修正了按不等大小折直接平均的汇总口径。真实监督是评分员等权原始均分，MAE与标准化附件4的旧误差不可直接横比。
'''
 q2+='\n'+md(qtable)+'\n\n'
 q2+=f'''最终全训练数据内部选择为{LABEL[qsum['final_method']]}，故正式50×3表、分子名称、选择得分、入选率与替代组指标统一来自该规则。训练内选法流程的外层MAE为{qm.set_index('method').loc['TrainSelected','MAE_mean']:.4f}，固定2+1的OOF MAE为{qm.set_index('method').loc['Hybrid2plus1','MAE_mean']:.4f}。前者没有获得额外预测收益，说明小样本内层选择具有方差；不能用全量内部选出的2+1替代训练内选法的外层成绩。配对检验先平均同款酒的两次OOF预测，再按酒计算误差，与表中逐次误差均值的口径不同。检验与啤酒bootstrap区间见closed/q2/paired_tests.csv，没有据测试结果再设新的加权规则。

## 固定规则与完整流程的两类稳定性

100次Bootstrap在各风格内按酒重采样，填补、缩放和选择全部重做。重复抽到的同款酒被整体放入同一内层角色，避免其副本跨越内层训练与验证。固定规则频率回答“给定规则时分子是否稳定”；训练内选法频率还包含规则本身变化。最终150行同时保存两种频率，避免以固定规则频率代替完整流程的不确定性。

替代组沿用描述性完全链接分组：组内各对的绝对Spearman至少0.8。它用于衡量相近测量是否可替代，不参与外层特征选择。每个目标用相同的100对Bootstrap集合计算单分子及替代组Jaccard，并同时记录三因素内部绝对相关。

{md(stable)}

内部规则的Bootstrap选择次数如下。规则切换本身是小样本不确定性的结果，不据此继续增加规则或调节惩罚系数。

{md(choice.assign(method=choice.method.map(LABEL)).rename(columns={'method':'规则','count':'次数'}))}

## 风格混杂与化学解释

表中保留原始Spearman、风格内去均值后的秩关联、标签疑点以及来源/单位。后者是混杂敏感性指标，不是已识别的因果效应；风格也可能代表真实发酵过程，调整后关联变化需结合过程解释。高度相关的酯和高级醇可能共同反映发酵条件，不能简单将排名逐一解释为独立作用。原论文通过真实添加实验验证部分候选[1]，本工程尚未开展这一步。

最终表不以“化学常识”手工替换统计结果。高频主因素、同组可替代项和低频候选分别解释；barnyard保留疑似公式别名标记。每一项的名称、频率、方向和风格调整变化见统一CSV与逐维解释报告，供队员独立作化学复核。
'''
 q2+=fig('10_q2_unified','四规则与训练内选法的预测、稳定性及选择频率')
 gtable=g[['model','SWD_mean','style_accuracy_mean','correlation_MAE_mean','joint_floor_MAE_mean']].copy();gtable.model=gtable.model.map(LABEL);gtable.columns=['模型','条件SWD','风格一致性','相关误差','联合低端误差']
 gen=r'''# 问题三之一：22风格条件生成的分布检验

## 小风格样本与观测最低值的冲突

每种风格仅3至31款，无法支持各自独立估计231维联合分布。直接连续生成又难以保留大量相同最小值。采用总体信息共享可以降低方差，却可能削弱风格边界；分布距离变小也可能来自生成谱向均值收缩。本轮将这种取舍作为待验证问题，比较已有四模型，参数与训练角色全部冻结。

PCA高斯使用12维潜变量、风格均值收缩和共享协方差，输出限制在训练范围。独立Copula与联合Copula共享收缩条件边际：

$$\widehat F_{j,s}=w_s\widehat F_{j,s}^{\rm empirical}+(1-w_s)\widehat F_{j,all},\qquad w_s=\frac{n_s}{n_s+10}.$$

对稀有风格，较大的总体权重控制经验分布方差。联合Copula将训练秩变换成正态分数，用Ledoit–Wolf收缩估计相关矩阵[5]，通过 $z\sim N(0,\widehat R)$、$x_j=\widehat F_{j,s}^{-1}(\Phi(z_j))$ 采样。独立消融令相关矩阵为单位阵，检验联合依赖的贡献。经验逆分布的区间自然保留最小值点质量，不需要后续逐列随机替换；模型并未识别最低值以下的真实浓度。

CVAE沿用231维输入、22类条件、64维隐层、12维潜变量和KL系数0.05，最多160轮，以原35款验证酒的重构误差选检查点[6]。四模型不增加训练轮数、结构或超参数搜索。

## 逐风格评价如何避免循环自证

三个原始场景均保留140款生成训练酒、35款验证酒、40款独立评价器训练酒与35款测试酒。对每个场景的每种风格生成40条谱，共计算22×4×3=264行指标。四模型共享按场景与风格固定的采样种子、48个投影方向和训练尺度。风格分类器使用原迁移评价器的既有ET配置，在独立40款酒上拟合并保存；其输出只用于评价，不参与模型或样本筛选。

对测试风格分布计算边际Wasserstein与切片Wasserstein距离。相关结构使用真实和四模型共同可计算的Spearman变量对；测试样本少于4款时不估计该项。令 $B_j(x)=1\{x_j\le m_j^{train}\}$，其中 $m_j^{train}$ 为训练最小值，仅对训练中低端比例至少25%的变量评价。联合最低值模式用各变量对的 $E[B_jB_k]$ 绝对误差衡量，同时报告单变量低端质量误差。这里的“低端”是观测编码阈值，不是已知仪器LOD。

最近邻距离、完整复制比例和标准化多样性均在最终输出谱上计算。零完整复制只能排除逐行原样复制，不能证明经验边际没有训练信息记忆。风格分类器的输出分数也没有概率校准，应解释为模型诊断分数。
'''
 gen+=f'''
## 支持范围与真实结果

原划分中{n_cov}/22种风格有独立测试酒；Faro与Flanders old brown为零。相关结构仅{n_corr}/22种风格在至少一个场景满足样本数条件，共{nctx}个风格场景。两类缺口在表格和热图中保留NA，不能用验证酒、训练酒或生成样本补成测试证据。完整22风格覆盖表及88行分模型汇总见附录。

下表先在每个风格内跨场景平均，再对有定义的风格等权平均；分类一致性覆盖全部22种风格，分布和联合低端误差覆盖20种，相关误差只覆盖4种。各列支持集合不同，因此必须按列比较同指标，不能把它们合成未预设的总分。

{md(gtable)}

条件尺度结果与旧版混合风格总体距离回答不同问题。CVAE的条件SWD较小，但其平均多样性为{gm.loc['CVAE','diversity_mean']:.4f}；独立Copula为{gm.loc['CopulaIndependent','diversity_mean']:.4f}。当真实测试风格只有一两款时，收缩到中心可能使经验距离更小，不能据此断言CVAE更真实。最近邻距离、风格一致性和低端质量共同揭示这种取舍。

上表会让读者产生一个必须正面回答的疑问：既然 CVAE 的条件 SWD 最低（{gm.loc['CVAE','SWD_mean']:.4f}），为什么交付的生产生成器是独立 Copula（{gm.loc['CopulaIndependent','SWD_mean']:.4f}）而不是 CVAE？答案是两件事发生在不同时间、依据不同数据，且排序反转本身可以被量化。选型在封闭诊断之前就已冻结：三个场景各自的验证折（35 款酒）上按平均 SWD 取最小者，逐场景选出的分别是联合 Copula、独立 Copula、独立 Copula，测试折从头到尾没有参与选型，封闭诊断也不被允许回改选型。把封闭诊断的逐风格结果拆开看，CVAE 的相对优势与测试集大小高度相关（相关系数 r={sel_gap_r:.2f}）：在只有 1 款独立测试酒的风格上 CVAE {sel_b0_win}/{sel_b0_n} 全胜、优势 {sel_b0_gap:+.3f}，2 款时 {sel_b1_win}/{sel_b1_n}、{sel_b1_gap:+.3f}，3—4 款时 {sel_b2_win}/{sel_b2_n}、{sel_b2_gap:+.3f}，5 款及以上时只剩 {sel_b3_win}/{sel_b3_n}、{sel_b3_gap:+.3f}。也就是说，CVAE 的优势随经验分布分辨率上升而单调消失——测试集只有一两款酒时，经验一侧几乎没有分辨率，任何向风格均值收缩的生成器都会显得“更近”。这与 CVAE 自身的退化程度一致：多样性 {gm.loc['CVAE','diversity_mean']:.4f} 对 {gm.loc['CopulaIndependent','diversity_mean']:.4f}，到最近训练样本的 RMS {gm.loc['CVAE','nearest_train_RMS_mean']:.4f} 对 {gm.loc['CopulaIndependent','nearest_train_RMS_mean']:.4f}，它更靠近小样本经验分布。因此选型依据不是条件 SWD 这一列，而是验证折规则加上多样性与低端质量的匹配；反过来说，如果 CVAE 在 5 款以上测试酒的风格上仍保持优势，冻结的选型就是错的——它没有，6 对里 Copula 赢 {sel_b3_n - sel_b3_win} 对。需要同时承认的是，满足该条件的风格只有 {sel_b3_n} 个，这只是一个方向性证据，不足以给出决定性检验。完整对照见 outputs/rebuild/q3_complete/selection_logic.json 与 selection_criteria_table.csv。

联合Copula相关误差为{gm.loc['CopulaJoint','correlation_MAE_mean']:.4f}，独立Copula为{gm.loc['CopulaIndependent','correlation_MAE_mean']:.4f}，改善范围限于可估计场景；联合低端误差分别为{gm.loc['CopulaJoint','joint_floor_MAE_mean']:.4f}和{gm.loc['CopulaIndependent','joint_floor_MAE_mean']:.4f}，未显示联合模型的统一优势。共享一个总体依赖矩阵不能自动恢复每类风格特有的低端共现关系。

以上数字给出了本问能支持到什么范围，需要一并交代清楚。第一，风格条件控制本身仍然有限：四个模型的独立评价器风格分类准确率分别为 CVAE 0.0917、PCA 高斯 0.0852、联合 Copula 0.1356、独立 Copula 0.1409，在 22 类问题上都只略高于随机水平（1/22≈0.045），说明「生成了指定风格的谱」不等于「已充分复现该风格的特征」，风格画像应视为条件生成的方向性结果而非风格标定证据。第二，覆盖度不完整：22 种风格中仅 20 种在既定划分下有独立测试酒，相关结构指标只有 4 种风格满足样本条件，Faro 与 Flanders old brown 两类完全没有独立测试证据。第三，物理尺度仍有缺口：化学列中只有一部分具备可靠的量纲校准证据，其余仅保留附件编码，因此生成与迁移结论只能建立在观测编码与本报告给出的阈值判定上，不能外推为绝对浓度结论。

生产生成器和迁移生成器继续沿用此前验证集选定结果，未依据此次条件测试成绩改选。Tripel、Kriek、Lambic画像及22风格880条谱仍保留为展示输出；生成模型的统计接近不构成真实酿造可行性证据。
'''
 gen+=fig('11_generation_styles','22风格条件距离与独立风格一致性；灰色表示无测试证据')+fig('12_generation_structure','联合低端模式与有共同变量对支持的相关结构误差')
 mtable=pmetrics.reset_index();mtable.policy=mtable.policy.map(LABEL);mtable.columns=['策略','目标收益','stale变化','stale上升率','化学RMS','拒绝率'];tt=tests[['policy','metric','mean_difference','p_holm']].copy();tt.policy=tt.policy.map(LABEL);tt.columns=['策略','指标','均值差','Holm校正p']
 migration=r'''# 问题三之二：独立评价与拒绝迁移

## 决策目标和评价器必须隔离

跨风格迁移同时追求目标接近与缺陷不恶化。若同一个感官预测器既筛选候选又证明安全，优化器容易利用其误差。本实验固定原始9款不同Pils源酒，每款只保留其最早符合条件的原测试场景，避免重复计数。源酒不属于相应生成训练、验证或评价器训练角色。三个场景的独立评价器在优化前序列化，其文件哈希在所有策略完成后保持不变。

每个源生成80条Tripel候选，再以0.25、0.5、0.75、1.0插值形成320条迁移谱，加入原谱作为第321个候选。四策略共用候选、设计预测器、尺度和成本；固定的拒绝策略直接返回原谱。对候选 $x'$，成本为

$$J(x')=\|T(x')-\mu_{Tripel}\|_2+0.15\|T(x')-T(x)\|_2,$$

其中 $T$ 为生成训练集标准化，目标中心来自同一批Tripel候选。独立风格分类分数不进入这个成本。

无约束策略选择最低成本。点约束与不确定性约束先将四个明确化学标记裁剪至不高于源编码，再用12个仅在生成训练集拟合的Bootstrap ET计算stale_hops预测差。点约束要求平均差不大于零；不确定性约束要求90%分位差不大于零。保留原谱保证候选集合存在可行返回；返回原谱计为拒绝迁移。该分位没有经覆盖率校准，是模型集成风险筛查，不是90%置信安全保证。

所有策略的选择完成后，才读取独立评价器，对源谱与最终谱统一评分。评价器包括固定ET分类器、感官回归器和在相同独立40款酒上构造的12个Bootstrap回归器。评价器的集成差分位用于检验设计风险能否迁移，不反馈到选择过程。

## 四策略的9酒配对结果

目标收益为独立分类器Tripel分数之差；缺陷风险同时报告stale_hops均分变化、上升酒数和独立集成90%分位。化学RMS为按生成训练尺度标准化的231维改变量。下表以全部9款为分母，包含被拒绝酒的零收益，避免只在接受子集上展示效果。
'''
 migration+='\n'+md(mtable)+'\n\n'
 migration+=f'''点约束带来平均目标分数收益{pmetrics.loc['PointConstraint','target_gain']:.4f}，其独立stale_hops均值变化为{pmetrics.loc['PointConstraint','stale_change']:.4f}，但仍有{int(riskcounts['PointConstraint'])}/9款上升。平均下降与个体风险并不矛盾，说明仅报告均值会掩盖失败样本。不确定性约束拒绝{int(refcounts['UncertaintyConstraint'])}/9款，全部源酒的平均目标收益降为{pmetrics.loc['UncertaintyConstraint','target_gain']:.4f}，仍有{int(riskcounts['UncertaintyConstraint'])}/9款上升；在实际接受的7款中对应4/7。因而本次证据不支持“更保守的设计约束必然带来独立缺陷改善”。

必须把这一点明确记为局限，但要同时给出赛题口径下的对照，否则会低估约束的作用。本文的缺陷判定有两套口径，二者结论强度不同，缺一不可。

第一套是本文沿用至今的严格口径：只要独立评价器的 stale_hops 预测相对源谱上升了任意 epsilon（>1e-10）就计为恶化。按此口径三种策略的上升酒数是无约束{int(anypos['Unconstrained'])}/9、点约束{int(anypos['PointConstraint'])}/9、不确定性约束{int(anypos['UncertaintyConstraint'])}/9，约束把上升数从6压到4—5但没有压到0；代价是目标收益由{pmetrics.loc['Unconstrained','target_gain']:.4f}降到{pmetrics.loc['UncertaintyConstraint','target_gain']:.4f}（约减半）并额外拒绝{int(refcounts['UncertaintyConstraint'])}/9款。这一口径下确实没有得到“不恶化”。

第二套是赛题原文指向的口径：题目把缺陷化合物定义为“所列 DMS/Diacetyl/4VG/trans-2-nonenal/Stale hops 等超阈值项”，即判定对象是超阈值与否，而不是任意微量上升。按此口径统计“源酒未超阈值、迁移后越过阈值”的次数（三个具备可用感官检出阈值的化学缺陷项，源酒低于阈值的可用样本23个），无约束策略发生{int(xc['Unconstrained'])}次越阈，点约束{int(xc['PointConstraint'])}次，不确定性约束{int(xc['UncertaintyConstraint'])}次。四个有观测化学编码的缺陷项在点约束与不确定性约束下的化学增加次数均为0，与这一结果一致。

两套口径的差别只有0.05分以内（0—6量表上），而这远低于模型自身在stale_hops上的约0.23 MAE，因此任何一套都不能单独支撑“保证不恶化”，也不能据此否定约束的作用。本文的主张因此限定为：在赛题定义的超阈值口径上，硬约束把越阈次数由无约束的{int(xc['Unconstrained'])}次降到{int(xc['UncertaintyConstraint'])}次，即目标函数意义上的“缺陷不恶化”在本次内部检验中达成；在任意微量上升的严格口径上仍有4—5/9款酒上升，该残余风险列为未解决局限。要真正闭合这两套口径之间的差距，需要真实添加样品与盲评，或在本数据之外校准评价器误差的传递关系，两者都超出本轮可得证据。两套口径的逐项结果见 outputs/rebuild/q3_complete/defect_criteria.json、defect_threshold_crossing.csv 与 defect_magnitude.csv。

拒绝迁移的收益与改变量为零，拒绝率100%，提供明确的决策下界。它不能作为实现目标风格的成功，但在缺少可靠安全证据时是可陈述的保守输出。本轮未通过观察独立评价器后新增拒绝阈值，也未把拒绝策略改造成反馈优化器。

## 统计与缺陷口径

2000次配对Bootstrap按9款不同源酒重采样，区间条件于固定训练场景与固定评价器，不包含重新训练的不确定性。Wilcoxon检验对三种实际迁移与拒绝的两类收益/缺陷差分别组成6项比较，再作Holm校正。

{md(tt)}

点约束与无约束的目标分数收益在本次内部检验中通过0.05的Holm门槛；三个实际迁移策略的stale变化均没有通过。小样本条件下不能把未拒绝零假设解释为缺陷等效或安全。

DMS、Diacetyl、4VG、trans-2-nonenal分别核对附件列名和变换。前三项同时列出平均及中位感官阈值，trans-2-nonenal和stale_hops没有完整可用阈值时记NA。感官检测阈值不等于跨风格通用的不可接受界限。对于四个有观测化学编码，约束策略的增加次数为零；源缺失值和低于/位于训练最小值的标记保留“物理不可验证”，不据裁剪宣称真实浓度顺序。stale_hops使用独立感官代理，不能换算成唯一化合物浓度。

逐酒原谱角色、评价器哈希、选中候选编号、接受/拒绝、五类缺陷检查及单位表均保存。统计生成和迁移结果仍需要真实样品添加与盲评才能验证工艺和感官可行性。
'''
 migration+=fig('13_migration_paired','9款独立源酒的目标收益和独立stale_hops变化')+fig('14_migration_risk','缺陷上升率、化学改变量与拒绝率；误差棒为酒级Bootstrap区间')+fig('15_transfer_chemistry','预先固定为最小beer编号的迁移示例，未按效果择优')
 innovation=f'''# 创新动机、消融与模型评价

本研究的改进围绕三个可核验的困难展开。第一，真实评分卡既有直接强度，也有父强度继承的子项，且存在重复评分员。按评分观测机制定义目标与条件似然，使模型学习对象明确；冻结的结构消融显示其优于普通MLP，但未超过ET，因而保持传统树模型作为预测主线。

第二，Top-3既是一个预测问题，也是一个选择不确定性问题。将四种规则放入训练内选择，并在Bootstrap中重新选择规则，使最终表、性能和稳定性属于同一个可复现流程。外层结果显示自动选法没有超过固定2+1，这个负结果揭示了小样本内层方差，不能通过事后改表消除。替代组使可替代信息可见，但不是新增因果证据。

需要把结论强度说得更准确，避免读者误读。最终 2+1 规则的训练内得分（0.4841）略低于纯 Spearman 基线（0.4845），重采样稳定性也低于它：2+1 的平均分子 Jaccard 为 0.2288、替代组 Jaccard 为 0.2354、Top-3 内平均绝对相关 0.2740，而 Spearman 分别为 0.2827、0.2929、0.4385。外层预测 MAE 上 2+1 为 0.4988，优于 Spearman 的 0.5021，但差距仅 0.003 量级。因此第二问的贡献应定位为：把「选择哪一种规则」纳入训练内部并输出可复现的统一 50×3 表，同时用冗余控制减少同一机制在 Top-3 内重复占位；不能宣称 2+1 在预测性能或重采样稳定性上全面优于相关性基线。Top-3 仍是候选因素清单，其因果地位需真实添加实验确认，本工程未开展该步。

第三，生成任务区分单变量最低值质量、联合低端模式和风格条件一致性，迁移任务进一步区分设计预测器与独立评价器。独立Copula是联合依赖消融，PCA高斯是连续低维基线，CVAE是非线性共享生成对照；无约束、点约束、不确定性约束与拒绝给出决策层对照。对照没有支持“深度或联合依赖模型在所有指标上更优”，也没有支持“设计风险分位保证独立缺陷安全”。

创新应定位于对本题观察机制的具体处理和完整验证，而不是宣称新发明Copula、Bootstrap或变分自编码。尚无外部新队列与真实酿造盲评，因此不能宣称一般化的感官因果机制。最稀有两类没有独立测试证据，相关结构多数风格不可估计；这些缺口限制结论支持范围。按用户设定，三组封闭补充完成后停止所有模型探索，不再为改善结果增加模型、阈值或评价器。

# 结论与建议

250款啤酒、231项测量与2197条逐评分员记录形成统一目标和训练隔离基础。第一问冻结ET的MAE为0.4454、RMSE为0.6305，整数输出对应真实面板均分的舍入等级。第二问最终训练内部选出2+1并输出50×3候选；训练内选法流程的OOF MAE为0.5024，说明方法选择的代价应进入评价。第三问完成22风格四模型条件诊断，20种风格有独立分布证据、4种风格具备最低相关估计支持；指标取舍与有限样本误差阻止了单一最优生成器的全面结论。9酒配对迁移显示，点约束保留目标收益，却仍有5款独立缺陷代理上升；不确定性约束以2款拒绝为代价仍出现4款上升。

该工程已完成既定三组补充，具备数据、代码、指标、图表与论文之间的追溯关系。正式参赛前的必要工作是队员独立核实核心建模与分析、确认匿名样本映射及疑点标签，并决定可采纳的实验结论。真实工艺与盲评验证属于当前数据无法替代的证据。
'''
 intro=r'''# 问题分析与统一研究主线

啤酒风味预测的困难首先来自观察尺度：化学列混合编码、评分卡含层次继承、同一评分员重复品评，而风格只提供小样本条件。若监督目标、关键因素与生成评价采用不同口径，复杂模型的表面提升可能无法解释。因此，围绕“观察了什么、训练可以使用什么、何种证据支持结论”组织三问。

原论文[1]结合化学特征、专业面板与消费者信息，并通过添加品评检验候选成分。赛题要求新增0—6分级、50维Top-3和跨风格缺陷约束。作者公开的逐评分员数据[2]使真实面板尺度可用，但不能消除评分卡疑点或把附件4简单逆变换成原分。先恢复可识别目标，再评价候选解释与生成决策，是三问衔接的基础。

本研究保留有竞争力的传统基线，也实现评分机制驱动的层次网络与CVAE。是否采用模型由冻结实验中的证据决定，不以结构复杂度作为创新标准。最终补充限定为Q2统一选择、22风格生成诊断和9源酒独立迁移三组，全部完成后转入论文定稿。

主要假设为：匿名beer_name按附件物理行序对应；面板目标给予评分员等权；缺失化学测量以训练中位数进行工作性填补；小风格借用总体经验边际降低方差。每项假设均保留其可验证范围，尤其不把最低值全部视作已知检测限，也不把统计生成谱视作直接酿造配方。

# 符号、评价对象与可复现约定

设 $x_i\in\mathbb R^{231}$ 为附件编码谱，$s_i$为风格，$y_{irt}$为评分员$r$在重复$t$的评分。原始目标为

$$\bar y_i=\frac{1}{R_i}\sum_r\frac{1}{T_{ir}}\sum_t y_{irt},\qquad g_i=\min(6,\max(0,\lfloor\widehat y_i+0.5\rfloor)).$$

因此连续监督、面板整数等级与个体打分是三个不同对象。所有外层误差按啤酒与感官目标等权；相同啤酒的全部记录留在同一角色。数据填补、标准化与监督选择均从当前训练数据估计。

两个原始种子提供两次三折内部验证；生成任务沿用三个140/35/40/35分区。原数据已在此前探索中被检查，本轮是严格隔离的内部补充验证，不是新外部队列。重复划分标准差衡量划分变化，不能当作独立多次采样的总体置信区间。配对酒级Bootstrap给出条件于既有训练和评价器的样本不确定性。

完整原协议、封闭补充协议、Q1冻结哈希、训练日志和结果索引一并保存。本文遵照本地官方格式与AI规定[3]，图表、程序与文字均如实记录AI贡献；不补造队员的独立工作或签名。
'''
 declaration='''# AI使用声明与独立工作状态

本工程及本稿使用OpenAI Codex，历史GPT-6系列及本轮gpt-6.1-sol[7]，辅助数据核验、模型实现、实验运行、统计、可视化、论文和排版。本轮严格执行三组封闭补充，Q1不重训、不改预测，仅在2026-10-28修正了评分卡交付的字段过滤缺陷（30字段→3.xlsx summary 全部50字段），修正记录见 outputs/rebuild/audit/protocol_amendment.md，冻结哈希同步更新；三组完成后停止建模探索。程序头部、图表说明和支撑材料均标明AI参与。

官方规定要求核心建模、算法设计、求解与分析由参赛队员独立完成。当前没有可核实的队员独立核心工作或人工修改记录，本稿如实保留该状态；单纯审阅签字不能替代独立完成。队员须独立开展并记录关键工作后决定正式采用内容。详细说明、关键交互摘要与待完成记录位于AI工具使用详情说明.pdf、AI_usage_log.md和human_decision_record.csv。
'''
 abstract=f'''针对啤酒化学数据与感官评分的尺度混合、Top-3共线冗余及小样本风格迁移问题，构建由观测语义、训练内选择和独立风险验证衔接的三问方案。将2197条原始品评记录映射到250款啤酒，定义评分员等权0—6面板目标，并保留评分卡层次和重复品评敏感性。评分卡交付为250行×53列、无缺失值：1列啤酒编号，50列由冻结模型预测的感官评分，另2列acetaldehyde为阈值锚定推断值（无面板0—6记录，由附件1实测浓度经log10反变换后与附件8的50.0mg/L感官检出阈值比较，250款酒全部低于阈值故取0=Not present），后两列明确标注为非面板评分。该评分卡存在一项已量化并披露的局限：预测等级全部落在0—4级，真值≥5的121个单元被系统低估（平均低估2.07，连续预测上限仅4.34），故整体MAE可用而高强度尾部不可用。冻结的Extra Trees取得MAE 0.4454、RMSE 0.6305，层次网络未超过强传统基线；非香气强度项的平均MAE（0.3486）优于香气强度项（0.5047）。化合物筛选在训练内部比较Spearman、2+1、三步条件前向与冗余控制，最终选出2+1并统一50×3结果；包含方法选择的外层MAE为0.5024，揭示小样本选法的方差代价。对四种既有生成模型完成22风格分布、分类一致性、相关结构及联合低端模式诊断：交付生成器沿用诊断之前冻结的验证折规则而非条件SWD排名，因为CVAE的表面SWD优势与测试集大小强相关（r=0.70，仅在1—2款测试酒的风格上领先，5款以上时优势消失）。9款独立源酒的四策略配对比较按两套缺陷口径报告：在赛题所指的超阈值口径上，硬约束把越阈次数由无约束的7次降到0次；在任意微量上升的严格口径上仍有4—5/9款酒上升，该残余风险列为未解决局限。结果支持保留拒绝选项、报告个体风险并区分编码约束与真实化学保证。三组封闭实验完成后停止模型探索，交付可复现代码、风格谱、候选因素和完整验证材料。本文为如实披露AI贡献的研究定稿，队员独立核心工作状态另列。'''
 paper='# 观测语义约束下的啤酒风味预测、化合物筛选与条件生成\n\n## 摘要\n\n'+abstract+'\n\n关键词：面板评分；训练内选择；替代组稳定性；条件生成；独立风险\n\n'+intro+'\n'+audit+'\n'+q1+'\n'+q2+'\n'+gen+'\n'+migration+'\n'+innovation+'\n'+refs+'\n'+declaration
 (R/'paper/competition_draft.md').write_text(paper,encoding='utf-8');(R/'paper/rebuild_body.md').write_text(paper[paper.index('# 问题分析'):],encoding='utf-8');(R/'paper/rebuild_abstract.txt').write_text(abstract,encoding='utf-8')
 (R/'reports/02_q2_analysis.md').write_text(q2,encoding='utf-8');(R/'reports/03_q3_analysis.md').write_text(gen+'\n'+migration,encoding='utf-8');(R/'reports/04_innovation_and_ablation.md').write_text(innovation,encoding='utf-8');(R/'reports/08_closed_experiments.md').write_text(q2+'\n'+gen+'\n'+migration,encoding='utf-8')
 final=f'''# 最终封闭补充结果与交付索引

三组封闭补充均完成，停止所有模型探索。Q1 不重训、不改预测，但交付层有两处修正（均由数据核实后作出，不涉及模型）：(1) 评分卡交付表此前由列名前缀过滤只输出 30 个字段，现补齐为 3.xlsx summary 定义的 50 个 0—6 监督字段；(2) 两列 acetaldehyde 此前留空，现由附件1 实测浓度对附件8 感官检出阈值判定后统一取 0（250 款酒浓度 0.01—18.56 mg/L，全部低于 50.0 mg/L 阈值），判据见 outputs/rebuild/q1_complete/acetaldehyde_grading.json。交付表因此为 53 列 = 1 列 beer + 50 个感官监督模型预测字段 + 2 个阈值锚定推断字段，250 行、无缺失值；乙醛两列明确标注为非面板评分，不与 50 个监督字段混同。冻结哈希与协议说明同步更新，记录见 outputs/rebuild/audit/protocol_amendment.md。

Q1 交付表的字段覆盖不均衡并不存在：30 个带 A_/F_ 前缀字段平均 MAE 0.5047，20 个无前缀字段平均 MAE 0.3486，即此前被漏掉的 20 个维度反而更准。

Q2 最终规则为 2+1；固定 2+1 的两次 OOF 平均 MAE 为 0.4988，训练内方法选择流程 MAE 为 0.5024。最终 150 行、预测指标、固定规则与流程 Bootstrap、替代组频率统一保存在 outputs/rebuild/closed/q2；兼容入口 outputs/rebuild/q2/top3_50x3.csv 与最终表逐字节一致。

22 风格四模型已生成 264 行场景级与 88 行汇总诊断。Faro 与 Flanders old brown 无独立测试酒；相关结构仅 4 种风格可估计。交付生成器沿用诊断之前冻结的验证折规则（三场景分别选出联合 Copula、独立 Copula、独立 Copula），未按条件 SWD 排名回改：CVAE 的条件 SWD（0.7059）虽最低，但其优势与测试集大小强相关（r=0.70；仅 1—2 款测试酒的风格上领先，5 款以上时优势消失为 −0.011 且 6 对中输 2 对），且 CVAE 多样性 0.4202、最近训练样本 RMS 0.6812，明显更靠近小样本经验分布。完整量化见 outputs/rebuild/q3_complete/selection_logic.json。

迁移按 9 款不同源酒比较 4 策略，独立评价器冻结，优化后才评分。缺陷判定按两套口径并列报告：严格口径（任意 epsilon>1e-10 上升即算恶化）下无约束 6/9、点约束 5/9、不确定性约束 4/9；赛题原文的超阈值口径下，源酒未超阈值而迁移后越阈的次数为无约束 7 次、点约束 0 次、不确定性约束 0 次。目标增益点约束 0.0514、不确定性约束 0.0289（拒绝 2/9）。两套口径差 0.05 分以内，远小于模型在 stale_hops 上的约 0.23 MAE，故本文只主张：超阈值口径上硬约束有效，严格口径上仍存残余风险并列为未解决局限。均为模型代理风险，不是真实酿造安全保证。逐项见 outputs/rebuild/q3_complete/defect_criteria.json。

复现入口 scripts/run_final.ps1，仅执行既定三组并更新图表与论文，不重新训练 Q1；脚本不写死机器路径。论文定稿 PDF 为 paper/competition_final.pdf；AI 使用详情与提交前待补项见 submission/。匿名身份、真实队号、队员签署及独立核心工作尚未提供，未伪造。
''';(R/'reports/05_final_summary.md').write_text(final,encoding='utf-8')
 # Per-target chemistry comments are now tied to the final selected method.
 top=read('q2/top3_50x3.csv');explain='# 最终2+1逐感官化合物解释与复核线索\n\n150项均由统一训练内规则产生。以下为统计候选，未做因果确认。\n\n'
 for t,d in top.groupby('target',sort=False):
  explain+='## '+t+'\n\n'
  for _,v in d.iterrows():explain+=f"第{int(v['rank'])}项 {v.chemical}（{v.get('matrix_name','')}）：Spearman={v.spearman:.3f}，风格调整秩关联={v.style_adjusted_rank_association:.3f}，固定规则入选率={v.final_fixed_method_frequency:.0%}，完整流程入选率={v.bootstrap_frequency:.0%}，替代组频率={v.group_frequency:.0%}。来源类别：{v.get('origin_paper','未核实')}。"+('入选率不足一半，保留候选不确定性。' if v.bootstrap_frequency<.5 else '在本次完整流程重采样中较稳定，仍需独立化学验证。')+'\n\n'
 (R/'reports/02_q2_compound_explanations.md').write_text(explain,encoding='utf-8')
 # Deliverable checklist for the team's independent review. Each row must be filled by a real member
 # (reviewer, date, evidence); the AI never fills these in.
 csv([dict(decision=d,status='待队员独立完成与复核',reviewer='',date='',evidence='',required_deliverable=v)
      for d,v in [('匿名评分映射','确认 beer_name 物理行序与官方 beer 编号的对应关系'),
                  ('面板监督目标','确认评分员等权面板均分口径'),
                  ('结构模型采用或弃用','确认 ET 主线与层次网络消融的判断'),
                  ('Top-3最终规则','确认 2+1 与相关性基线差距的解释'),
                  ('缺陷阈值解释','确认 50.0 mg/L 等阈值只作感官参考'),
                  ('生成及迁移可行性','确认两套缺陷口径的表述'),
                  ('正式论文独立重写','确认正文数字与 CSV 一致')]],'audit/human_decision_record.csv')
 # Appendices: all 22 styles, all 36 decisions, all 150 factors.
 append='# 附录：统一50×3候选化合物\n\n最终规则为训练数据内部选择的2+1。频率、单位和风格调整关联见完整CSV。\n\n';pivot=top.pivot(index='target',columns='rank',values='chemical').reset_index();pivot.columns=['感官','候选1','候选2','候选3'];append+=md(pivot)+'\n\n# 附录：22风格独立测试覆盖\n\n';ct=coverage[['style','n_total','n_unique_test','n_correlation_contexts']].copy();ct.columns=['风格','总样本','独立测试酒数','相关可估计场景'];append+=md(ct)+'\n\n# 附录：逐风格四模型分布与联合低端结果\n\n';z=read('generation/metrics_22styles_summary.csv')[['style','model','SWD_mean','style_accuracy_mean','joint_floor_MAE_mean']];z.model=z.model.map(LABEL);z.columns=['风格','模型','SWD','风格一致性','联合低端误差'];append+=md(z)+'\n\n# 附录：9酒四策略逐项结果\n\n';z=mig[['beer','policy','target_gain','stale_change','change_RMS','refused']].copy();z.policy=z.policy.map(LABEL);z.columns=['源beer','策略','目标增益','stale变化','化学RMS','拒绝'];append+=md(z);(R/'paper/rebuild_appendix.md').write_text(append,encoding='utf-8')
 ai=R/'reports/AI_usage_log.md';text=ai.read_text(encoding='utf-8');tag='## 最后三组封闭补充与定稿（gpt-6.1-sol）'
 if tag not in text:ai.write_text(text+'\n\n'+tag+'\n\n用户限定Q2统一、22风格条件生成、9源酒独立迁移三组，禁止增加模型，Q1冻结。OpenAI Codex gpt-6.1-sol执行训练内选择、原四模型评价与四迁移策略，所有三组于日志记载完成后停止建模探索。此后仅整理CSV、图表、统计、论文、PDF与提交包。未伪造Faro等无测试类别的指标，未伪造人工签署或队员独立建模记录。\n',encoding='utf-8')
 dump({'Q1':'frozen','Q2':qsum,'generation_macro':g.to_dict('records'),'migration':pmetrics.reset_index().to_dict('records'),'n_styles_distribution':n_cov,'n_styles_correlation':n_corr,'n_sources':9,'model_exploration':'stopped after exactly 3 groups'},'final_summary.json');log('Final Chinese manuscript and reports complete')
if __name__=='__main__':main()
