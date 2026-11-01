from q1 import *
from q2 import *
from q3 import *
import shutil
def md(df):
 d=df.copy()
 for c in d:
  if pd.api.types.is_float_dtype(d[c]):d[c]=d[c].map(lambda x:f'{x:.4f}' if np.isfinite(x) else 'NA')
 return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in d.to_numpy())
def read(p):return pd.read_csv(O/p)
def figure(n,caption):return f'\n![{caption}](../outputs/rebuild/figures/{n}.png){{width=95%}}\n'
def main():
 q1=read('q1/repeat_metrics.csv');q1g=q1.groupby('model')[['MAE','RMSE','R2','AF_MAE']].mean().sort_values('MAE');q1sd=q1.groupby('model').MAE.std();q2=read('q2/fold_metrics.csv');q2g=q2.groupby('model')[['MAE','RMSE','R2']].mean().sort_values('MAE');q3=read('q3/test_metrics.csv');q3g=q3.groupby('model')[['SWD','marginal_W1','correlation_MAE','floor_fraction_MAE','style_accuracy']].mean();sta=read('q2/stability.csv').groupby('method')[['compound_jaccard','group_jaccard','within_top3_abs_rho']].mean();mig=read('q3/migration_metrics.csv');mig['prob_change']=mig.target_probability_after-mig.target_probability_before;migg=mig.groupby('method')[['prob_change','independent_stale_change','change_RMS']].mean();audit=json.loads((R/'outputs/review_20261028/audit_summary.json').read_text());summary=json.loads((O/'q3/summary.json').read_text());top=read('q2/top3_50x3.csv');stats=read('q1/paired_statistics.csv');sens=read('q1/sensitivity.csv').groupby('experiment').MAE.mean();best=q1g.index[0];best2=q2g.index[0];primary=q1g.loc['InnerSelected'];env=json.loads((O/'audit/environment.json').read_text());ext=read('audit/validation_extremes.csv');params=read('q1/hyperparameters.csv');param= {m:int(params[params.model==m].parameter.dropna().iloc[0]) for m in ['MLP','Hierarchy','HierarchyRater']}
 # Audit and rules form part of the report, not boilerplate assumptions.
 audittext=f'''# 数据审计与监督目标重建

本轮以原始附件和新公开的逐评分员文件为输入，旧代码及网页端结果只用于问题定位。原始文件 SHA-256 清单在 outputs/rebuild/audit/raw_manifest.csv；新文件来源为原作者 Zenodo v4 [2]，MD5 为 a8f911d8c0061e096abf3b087678e164，与本地文件一致。全样本既往已被检查，因此本轮所有划分均属于内部重采样验证，不能称为从未接触过的外部测试。

## 样本与字段

1.csv 实际为 250 款啤酒、231 项化学或理化特征；4.csv 为 50 项感官目标；22 种风格中 Faro 仅 3 款，Tripel 为 28 款。题面中的“226 种啤酒”与实物不符。231 列包含物理测量和聚合量，不能全部称为单一化合物。1、4、7 按唯一 beer 对齐；beer_id 存在精度损失，不作主键。原始逐评分员文件中的 beer_name 是附件物理行序 1—250，不是附件 beer 编号；Beer 是品评位置，不能作啤酒键。正确映射的感官均值与附件 4 中位相关为 {audit['mapping_correlation_median_correct_row']:.4f}，错误按整数 beer 连接仅为 {audit['mapping_correlation_median_wrong_beer_id']:.4f}。这是一致性证据，仍需队员核查作者匿名化映射说明。

新文件包含 2197 条记录、16 名评分员、37 个日期；12 款酒跨日期重复，187 条属于同一啤酒与评分员的重复组合。定义目标时先对同一评分员的重复记录求均值，再在评分员之间等权平均，避免重复次数增加某人的权重。与所有记录直接平均相比，最大单元变化为 {audit['equal_rater_label_change_max']:.4f} 分，整体变化较小，但对重复酒不能忽略。

## 评分卡层次与可识别性

50 个汇总字段由 26 个直接强度字段和 24 个派生细分字段构成。细分项在单条品评中等于父项强度或零，本轮逐条验证无违例。父项为零时，细分零分不能区分未勾选与勾选但强度为零。将这些零值全部作为“不出现”的监督标签会引入语义错误。面板均值满足子项不大于父项，但一般不满足“父均值乘出现频率”；两者的平均绝对差为 {audit['product_of_marginal_means_error_mae']:.4f} 分。

3.xlsx 汇总公式生成的空白零分不是已填评分。新数据使真实面板均分可识别，不再需要分位数生成假标签；然而均分本身连续，最终整数是面板均分的决策等级。评分卡有两个可交付口径：逐张 beer 表上有 45 个带标签打分格（Aroma 23、Flavour 22；Acetaldehyde 一行同服两列，Aftertaste 只属 Aroma），summary 工作表则把三类层次细分展开为 50 个机器可读字段，与 4.csv 及逐评分员记录的目标集合完全一致。本轮 50 个字段全部给出 0—6 分级评分，交付表按 summary 字段名与顺序输出。

acetaldehyde 需要单独说明，因为它是唯一一个“没有面板分数但有答案”的维度。发布数据中确实不存在它的 0—6 记录：3.xlsx 汇总表无此列，4.csv 无此列，作者公开的 6.csv 的 50 个感官列中也找不到 acetaldehyde；附件7 的 aroma_acetaldehyde 在 250 款酒中 240 款落在同一地板值，与实测浓度 Spearman 仅 −0.058，属重编码的出现标记而非强度标签。但“无面板分”不等于“无分级”：附件1 的 acetaldehyde 经 log10 反变换（transforms.csv 的 log10 拟合误差 2.9×10⁻¹⁰，249 对单元格；ln 为 0.89、逐列恒等为 5.27，故底数为 10）后为 0.01—18.56 mg/L、中位 4.10 mg/L；附件8 给出该化合物感官检出阈值 50.0 mg/L。250 款酒全部低于该阈值，最高值仅为阈值的 37%。按 3.xlsx 图例 0=Not present，本数据集在该维度整体处于未呈现档，故两列统一取 0，判据存于 outputs/rebuild/q1_complete/acetaldehyde_grading.json。该 0 是阈值锚定的推断值而非面板评分，正文与结果表均单独标注。

barnyard 与 stale_hops 的汇总公式引用同一单元格；原始记录有 2183/2197 条相同。该问题是疑似标签重复，不能据此擅自改写 barnyard。主表保留 50 项并加疑点标记，另报告去除该列的训练与评价敏感性。评分员内标准化后聚合可精确重现附件 4 的 {audit['z_reconstruction_exact_cells']}/12500 单元格，仍有差异；因此附件 4 不作为原始均分的简单可逆变换。

## 化学尺度与最低值堆积

73 个化学缺失单元由训练折中位数填补。附件 8 能核实 70 列的尺度，其中 37 列与 log10 对应，33 列保持原值或近似原值；161 列没有充分校准证据。pH、乙醇、色度、CO2 的实际配对支持 log10，不采信网络文案中的笼统原量纲说法。反变换逐字段执行，未知列仅保留附件编码。附件 1 与 8 有个别配对冲突，物理浓度判断需保留来源。

4VG 的 67.2%、CS2 的 83.6%、limonene 的 84.8% 样本位于同一个最小值。它们提示删失、编码地板或检测流程的混合效应，不能直接证明已知检测下限。生成模型学习观测分布中的点质量，不推断地板以下真实浓度。迁移中地板值保持相等也不能证明真实浓度未上升。

2.csv 经名称映射后可与重新计算的 Spearman 矩阵吻合，Pearson 则不吻合。全样本矩阵只用于描述性共表达图；每次预测特征选择重新在训练折内计算。原始相关矩阵 5、6 的数据说明与实际维度差异见前轮审计，本轮不将它们作为独立训练证据。

## 统一实验协议

依据本地官方文件 [3]，协议文件 configs/rebuild_protocol.json 在训练前保存。第一、二问使用固定种子 20261028、20261029 的两次三折划分；每个风格内循环分配，避免把不足折数的类别强行分层。所有同款酒的原始评分整体留在一折。内层从外层训练样本划出 20%，用于 Ridge/PLS 参数或网络训练轮数选择；再在完整外层训练集重训。训练与测试的中位数填补、缩放和选择严格隔离。

第三问每个种子分为生成训练 140、验证 35、独立评价器 40、测试 35 款。生成训练和评价器各保留各风格至少一款，因此极少样本风格的测试覆盖有限。预留 Pils 供迁移，分区文件完整输出。评价器从未参与候选筛选，验证集 SWD 用于选择生成器，测试结果仅报告。重复划分并非独立实验样本；统计推断以啤酒为重采样单位。运行中修正两处实现公平性问题：深度模型外层重训、生成距离共同随机投影。修正记录单列，未扩展模型搜索范围。

证据路径：outputs/rebuild/audit/、outputs/review_20261028/；重复品评的 Kruskal–Wallis 与 Holm 结果仅作描述，跨日期评分员构成不同，不能直接归因于时间漂移。
'''
 q1text=f'''# 问题一：真实面板均分与结构化监督

## 动机与目标

原始评分既提供 0—6 的实际量表，也揭示了评分员差异和细分项继承规则。直接对 50 个目标独立回归会放弃该结构，但共同叶节点的多输出树本身也能保持线性不等式。因此，结构模型的价值必须以误差和个体评分建模证据检验，不能以“存在层次”直接推出深度网络优越。

对酒 i、评分员 r、重复次 t 的记录 y，目标为先在 t 内平均、再在 r 间平均。Ridge、PLS、RF、ET、HGB 对这一面板目标训练；结构网络对逐记录似然训练，权重为同酒评分员数与该评分员重复次数的乘积倒数，使每款酒等权。测试以未见啤酒的面板均分为单位，既不把 2197 条当成 2197 款独立样本，也不把评分员身份用于推断测试酒真实分数。

## 模型与消融

传统基线包括训练均值、Ridge（alpha=10/100）、PLS（5/12 成分）、RF 与 ET [4]（180 棵树，叶最小样本 2，特征比例 0.7）、HGB（每目标 80 轮、7 叶、L2=10）。每个候选的训练与内层验证过程均独立执行。MLP 为 231→64→32→50，ReLU 和 0.15 Dropout；训练至多 160 轮，AdamW 学习率 0.002、权重衰减 0.001。

层次网络采用同一编码器，对 26 个直接强度预测七级概率 p(k|x)，对 24 个细分项预测条件勾选概率 q(k,x)。子项期望为 sum[k p(k|x) q(k,x)]，不能用两个边际期望相乘。子项出现损失只在父项强度大于零的记录上计算。加入评分员的版本在父项七级 logits 上添加中心化评分员效应，预测时使用零中心效应；这一估计对应平均 logits 的参考评分员，而不严格等同于全部评分员概率的平均，是该版本需进一步改进的估计目标差异。

MLP、Hierarchy、HierarchyRater 的可训练参数分别为 {param['MLP']}、{param['Hierarchy']}、{param['HierarchyRater']}。训练曲线与参数配置均保存。结构网络不是原论文的复现模型；未在本轮增加图神经网络，因为现有任务的样本量与机制证据尚不足以支持额外图模块。

## 实际结果

下表为两次完整 OOF 预测各自计算指标后取均值；MAE/RMSE 单位均为原始评分分值，R² 为 50 目标等权平均。

{md(q1g.reset_index())}

观察到的最低 MAE 方法为 {best}，MAE={q1g.loc[best,'MAE']:.4f}，重复划分标准差={q1sd[best]:.4f}。预先规定的内层模型选择流程 InnerSelected 的 MAE={primary.MAE:.4f}、RMSE={primary.RMSE:.4f}、R²={primary.R2:.4f}。观察到某模型排名第一与预先冻结选择流程的测试表现应分别表述，不能用全体 OOF 结果再选模型后声称其评估完全独立。

配对统计以每款酒的两次 OOF 平均预测为基础，与 InnerSelected 比较，用 2000 次啤酒级 bootstrap 计算差值区间，并对 Wilcoxon 检验作 Holm 校正；它评价的是重复 OOF 集成，与上表单次误差均值略有不同。各目标误差见 per_target.csv，不能用总体均值掩盖弱目标。

{md(stats[['comparator','positive_favors_reference','CI_low','CI_high','p_holm']])}

## 评分输出与敏感性

将连续预测裁剪至 [0,6] 后按 floor(score+0.5) 输出整数。准确率衡量的是四舍五入后的真实面板均分等级，而非任一评分员的个体打分正确率。原始连续目标、连续预测与整数等级分别输出，不把近似等级重命名为原始评分。交付表 outputs/rebuild/q1/aroma_flavour_template_with_unobserved.csv 共 250 行 × 53 列、无缺失值：1 列 beer；50 个由感官监督模型预测的 0—6 字段（沿用 3.xlsx 名称与顺序，仅 A_ester_fruity、F_ester_fruity、4vg 三处因历史命名差异而映射）；以及 A_acetaldehyde、F_acetaldehyde 两列阈值锚定推断值（当前均为 0）。后两列不是原始面板评分：发布数据中不存在该字段的 0—6 记录，其取值由附件1 实测浓度经 log10 反变换后与附件8 感官检出阈值 50.0 mg/L 比较得到（250 款酒全部低于阈值，故按 3.xlsx 图例取 0=Not present），判据见 outputs/rebuild/q1_complete/acetaldehyde_grading.json。因此评分卡中 50 个字段来自面板监督、2 个字段来自浓度对阈值的判定，两类证据强度不同，在交付表中仍以独立列呈现而不合并叙述。

50 个字段的 OOF 误差并不均衡，且难度与字段是否带 A_/F_ 前缀无关。30 个带前缀字段的平均 MAE 为 0.5047，20 个不带前缀字段（body、bitternes、co2、aftertaste、alcohol、sweetness、acidity、dms、metallic、X4vg、t2n、diacetyl、stale_hops、orange、coriander、clove、lactic、acetic、barnyard、overall）的平均 MAE 为 0.3486。R² 最高的是 lactic 0.7204、acidity 0.7183、alcohol 0.6653、acetic 0.6629、bitternes 0.5997、body 0.5010；而 dms、metallic、t2n、stale_hops、barnyard 的 R² 接近 0，但 MAE 也小，原因是这些缺陷维度本身变异极小。覆盖 50 个字段并未以精度为代价。

交付评分卡还暴露一个必须披露的局限：12,500 个面板单元与 500 个乙醛单元中，预测等级全部落在 0—4 级，没有任何 5 级或 6 级结果，而真实面板均分取整后确实存在 112 个 5 级与 9 个 6 级单元。这不是取整造成的假象：连续预测的最大值只有 4.34，连四舍五入到 5 都不可能，说明模型本身把高分向中部收缩。按真值档位分组可见系统性偏差：真值 [0,2) 的 8,253 个单元被高估 +0.164，[2,3) 低估 −0.173，[3,4) 低估 −0.461，[4,5) 低估 −1.071，真值 ≥5 的 121 个单元全部被低估、平均低估 −2.074（真值均值 5.07 被压到 3.00）。也就是说，50 列预测在整体 MAE 意义上可用，但尾部不可用：任何依赖识别高强度样品（例如强酒花香或强缺陷）的下游用法都会漏检。本轮不为此重训模型，因为三组补充实验已经封闭且冻结；该偏差作为明确局限记录，定量结果见 outputs/rebuild/q1_complete/grade_range_diagnostic.csv 与 grade_range_summary.json。若要改进，方向是高强度样本的加权或分位数校准，属于模型层面工作，超出本轮范围。

{md(sens.rename('MAE').reset_index())}

额外遮蔽 5%、10%、20% 输入只施加在测试特征，使用训练中位数处理，模拟随机缺失；它不代表真实非随机检测失败。重复评分权重和去除 barnyard 的敏感性结果用于判断数据语义选择的影响。面板均分的区间由评分员重采样获得，是有限评分员测量不确定性，不是模型预测区间。输出的全样本 ET 模型用于后续展示性画像，不能用其对训练数据的拟合分数代替 OOF 成绩。

证据：outputs/rebuild/q1/fold_metrics.csv、repeat_metrics.csv、training_curves.csv、sensitivity.csv、panel_uncertainty.csv。旧网页版的 AF30 MAE=0.5062 对应不同标签权重与五折划分，不与本轮数字直接作超越声明。
'''
 q2text=f'''# 问题二：互补化合物选择与稳定性

## 建图与候选定义

节点为附件 1 的 231 个测量变量，边取附件 2 对应 Spearman |rho|≥0.65，保存符号与绝对边权。图为全样本描述，不进入外层预测。候选因素剔除 13 项理化读数、聚合量和非单一分子池，保留 218 列；乙醇等可识别分子仍允许。pH、颜色、总酯之类可作预测特征，但不作为最终“关键化合物”顶替分子名称。

## 四种预设方法

Spearman 取绝对相关前三；Hybrid2plus1 先取相关前二，再在前二的正则残差空间选第三项；Forward3 每一步均以正则化残差增益选择；Redundancy 用绝对目标相关减去 0.5 倍与已选变量的最大绝对相关。所有方法恰好选择三项，并使用相同 alpha=8 的 Ridge 建立三变量预测模型。残差选择在标准化秩空间进行，正则为 0.02；最终三变量预测使用训练折标准化的原始附件编码。

风格调整版本先在训练折内部减去各风格均值，再计算秩关联与前向选择，最终预测仍以化合物为输入。它是检验风格混杂的敏感性，不等于消除了所有混杂，更不是因果效应估计。风格也可能处于生成过程的中间环节，调整后预测减弱本身不能证明原方法错误。

## 结果与统计证据

{md(q2g.reset_index())}

最低平均折 MAE 出现在 {best2}。完整前向选择并未获得足够证据支持其优于 2+1；因此不把计算步骤更多写成创新优势。最终 top3_50x3.csv 为预先指定 Forward3 的 150 行可复核结果；top3_all_methods.csv 同时提供所有四种方法，方便队员按预测、冗余与稳定性共同审定。没有在测试结果上重新制定加权排名规则。

{md(read('q2/paired_statistics.csv')[['comparator','positive_favors_reference','CI_low','CI_high','p_holm']])}

## 替代组稳定性

100 次 bootstrap 在各风格内抽样，重新填补、标准化和选择。每个因素提供入选频率；同一目标的相邻独立重采样对计算 Jaccard。替代组通过 |rho| 的完全链接聚类构建，组内两两绝对相关至少 0.8，避免单链接长链把弱相关变量并为一组。组稳定性用于表达同源共变的可替代测量，不证明某一成员的独立作用。

{md(sta.reset_index())}

若化合物入选频率低而替代组频率高，说明可稳定识别一组信息，但难以唯一指定分子。若两者均低，应明确 Top-3 不确定。去冗余方法降低共线性不一定提高预测；全样本描述组也不作为验证期特征选择器。

## 化学解释边界

各入选项按附件 8 关联中文/英文名、来源、单位和阈值，见 top3_chemical_annotations.csv。酯与高级醇的共同变化可能反映发酵过程，酸类相关可能携带酸啤酒风格信息。可检验的解释至少同时查看原始相关、风格调整关联、bootstrap 频率和同组替代物。没有元数据或明确实验支持的条目保留“待化学复核”，不以名称联想编造气味机制。

原论文以统计学习提出候选后做真实添加品评实验 [1]；本题现有数据仅支持关联与预测贡献。尤其 barnyard 标签有公式疑点，不能把它与 stale_hops 的相关包装成新发现。50×3 只是任务所需的候选列表，不是 150 个已经证实的因果作用。
'''
 q3text=f'''# 问题三：混合分布生成与受约束迁移

## 从观测语义到生成模型

最低值堆积使连续高斯模型难以重现变量的点质量；231 维对单风格 3—31 款又不足以独立拟合高维分布。本轮将“风格边际分布”和“跨变量依赖”分开处理。各风格经验分布以 n/(n+10) 权重与全体训练经验分布收缩混合，因此稀少风格借用总体信息，同时保留所观测的离散地板值。所有经验分布仅来自生成训练集。

CopulaIndependent 对各边际独立采样，是去掉依赖结构的消融。CopulaJoint 对训练数据中秩转换后的正态分数估计 Ledoit–Wolf 收缩协方差 [5]，归一为相关矩阵；联合正态样本经正态 CDF 与条件经验逆 CDF 输出化合物谱。并列最小值天然对应逆 CDF 的区间，不需要生成后逐列随机回填。它建模观测点质量的联合分布，不是对未知 LOD 以下浓度作 Tobit 参数估计；中秩近似及高斯依赖假设也会限制复杂尾部关系。

传统基线 PCA_Gaussian 将训练标准化特征降至 12 维，风格均值收缩后使用共享协方差采样，再逆变换并限制在训练范围。CVAE 基于变分自编码原理 [6]，输入为 231 特征与 22 类条件，64 维隐层、12 维潜变量，解码器为 (12+22)→64→231；参数量 {summary['cvae_parameters']}，KL 系数 0.05，最多 160 轮，验证重构误差确定检查点。其目的在于检查共享非线性表征能否克服小风格样本不足，不预设深度模型更好。

## 独立评价

三个种子均严格分离生成训练、验证、评价器训练和测试。评价器是 40 款真实酒训练的 ET 分类器/感官回归器；由于 22 类样本非常少，分类准确率上限证据弱，故同时报告真实测试酒分类成绩。评价器输出不用于优化生成样本或选择生成模型。所有模型使用相同固定 48 个随机投影方向计算 SWD，缩放参数来自生成训练集。真实测试值不被裁剪到训练范围。

{md(q3g.reset_index())}

生产生成器依据三次验证 SWD 均值确定为 {summary['production_model']}。验证单次选择可能不同，完整记录在 selected.csv；不能按测试最佳方法回写选择结果。下表记录独立分类器对真实测试样本的能力，帮助判断生成风格准确率的可解释范围。

{md(read('q3/evaluator_real_accuracy.csv'))}

联合 Copula 与独立 Copula 比较对应依赖结构消融；PCA_Gaussian 比较连续低维模型；CVAE 比较非线性生成。不同方法在 SWD、边际误差、相关结构、地板质量和条件一致性上存在取舍，不强行宣布一个模型统治全部指标。最近邻距离与完整样本复制比例在最终输出上重新计算；无完整复制不意味着无隐私风险或无训练记忆，经验边际仍取自训练观测值。

验证集中出现极端训练标准化值，导致个别 SWD 很大。这是固定度量对尾部的敏感性，保留在结果中，未根据验证或测试异常反复改变裁剪规则。极端值清单如下。

{md(ext)}

## 跨风格迁移与风险约束

从独立留出的 Pils/Lager 源谱出发，每源使用 80 个 Tripel 候选及 0.25、0.5、0.75、1.0 四个插值步长，加上保持不变候选。目标函数为到目标候选中心的标准化距离加 0.15 倍改变量。三个版本共用候选预算：不约束、点预测约束、90% bootstrap 预测差分位约束。

DMS、Diacetyl、4VG、trans-2-nonenal 对有观测的源编码施加逐项不增加。stale_hops 不是单一化学列，由仅在生成训练集上训练的 12 个 bootstrap ET 设计预测器评估；点约束要求平均差≤0，风险约束要求 90% 分位差≤0。该分位是模型集成的不确定性启发式，不是经覆盖率校准的置信保证。独立感官评价器只在候选确定后打分。

{md(migg.reset_index())}

啤酒级 bootstrap 先对同款酒跨划分变化求均值，再重采样；目前独立留出迁移覆盖 {mig.beer.nunique()} 款不同 Pils。三个版本的完整逐酒结果同时保存。风险约束即使满足设计预测器，仍可在独立 stale_hops 预测中出现上升；本轮没有证据证明其普遍降低真实缺陷风险，应将这一失败作为模型不确定性未校准与评价器样本小的共同限制。

## 阈值与量纲

DMS 参考均值阈值 17.25 μg/L；Diacetyl 0.29925 mg/L；4VG 0.206333 mg/L。参考均值与中位阈值不同，应明确使用口径。这些是感官检出参考，不是所有风格通用的不可接受界限。trans-2-nonenal 未取得本地附件明确阈值，stale_hops 没有唯一浓度阈值，二者不能计算虚构通过率。4VG 在部分风格中可有正面风味作用，但本题迁移约束仍按不恶化执行。

已校准 log10 列按 10 的幂转换，其余已核实原值列保留；无校准的 161 列不输出伪造物理浓度。缺失源浓度无法验证“不增加”，位于观测地板的源也不能证明真实浓度顺序。约束表把编码不增加、尺度变换、阈值参考和物理不可验证原因分开列出。

最终输出 22 风格各 40 个样本共 880 谱、对应 50 维预测感官画像、已校准量纲表和独立留出迁移实例。生成谱只满足当前统计范围与部分编码约束；缺少物料平衡、工艺动力学、真实化学混合及盲评验证，不能称为可直接酿造配方。
'''

 explanation='# 逐感官关键化合物解释与复核线索\n\n以下逐条列出真实选择结果。数值是全样本描述性关联与100次风格内重采样频率，不是独立因果效应。化学来源取附件8。\n\n'
 ann=read('q2/top3_chemical_annotations.csv')
 for target,group in ann[ann.method=='Forward3'].groupby('target',sort=False):
  explanation+='## '+target+'\n\n'
  for _,v in group.iterrows():
   origin=str(v.get('origin_paper','NA')); explanation+=f"第{int(v['rank'])}项 {v.chemical}：Spearman={v.spearman:.3f}，风格内调整秩关联={v.style_adjusted_rank_association:.3f}，入选频率={v.frequency:.0%}，替代组频率={v.group_frequency:.0%}；附件来源类别为 {origin}。"
   explanation+=('入选率不足一半，应视为可替换的候选而非稳定唯一因素。' if v.frequency<.5 else '在当前重采样中较常入选，仍需独立品评验证。')+'\n\n'
  if target in ['alcohol','body','diacetyl','dms','X4vg','t2n','overall']:
   q2text+='\n'+target+' 的实际候选为 '+ '、'.join(group.chemical)+ '；对应入选频率为 '+ '、'.join(f'{v:.0%}' for v in group.frequency)+'。低频成员不能与高频主因素作同等强度解释。\n'
 (R/'reports/02_q2_compound_explanations.md').write_text(explanation,encoding='utf-8')

 innovation=f'''# 创新动机、消融及证据判断

第一项候选改进是按评分卡观测机制建立条件层次概率模型。困难来自“子项评分=父强度×勾选状态”、重复评分员权重和零父强度下勾选不可观测。它通过条件于强度的子项概率与按酒、评分员均衡权重处理语义；MLP、Hierarchy、HierarchyRater 形成逐步消融。结构模型的误差应与强树模型共同比较，本轮证据不支持以深度模型替代树模型作为主预测器。对平均 logits 的评分员定义还存在目标差异，后续应先修正估计对象，再谈复杂架构。

第二项候选改进是把最低值点质量、联合依赖和独立风险审计连为一个生成—迁移流程。独立边际消融揭示依赖结构的作用；连续高斯和 CVAE 揭示点质量与小样本正则化的重要性；三种约束版本揭示“设计安全”与“独立评价安全”的差别。高斯 Copula、收缩估计和 bootstrap 都是成熟技术，本轮贡献应定位为针对观测语义的组合与验证，不声称创造了这些方法或已完成文献意义上的新算法证明。

{md(q3g[['SWD','correlation_MAE','floor_fraction_MAE']].reset_index())}

Q2 的去冗余与前向选择属于另一项必要对照，而不再包装为第三个核心创新。2+1、完整前向与冗余惩罚的排名表明：去除共线信息、提高预测和增加排序稳定性并非同一目标。替代组报告可以使不确定性可见，但不能把相近变量任意互换为已证实的因果因素。

不支持的结论：结构网络全面胜出、CVAE 优于传统概率生成、完整前向选择必胜、风险约束已保证缺陷真实不恶化、相对原论文性能更优。后续最有价值的工作是补充真实盲评、查明地板生成机制、增加独立风格样本及复核 barnyard 标签，而不是无边界叠加网络模块。
'''
 final=f'''# 本轮结果汇总

本轮实际完成：数据与原始评分审计；9 类第一问模型及内层选择；第二问四方法、风格调整、100 次 bootstrap 与有符号网络；第三问四生成模型、三种迁移约束与独立评价器；重复划分、敏感性、图表与中文参考论文。全部主要结果来自 outputs/rebuild，旧输出不混入新指标。

第一问观察最优 {best}：MAE={q1g.loc[best,'MAE']:.4f}，RMSE={q1g.loc[best,'RMSE']:.4f}，R²={q1g.loc[best,'R2']:.4f}，AF30 MAE={q1g.loc[best,'AF_MAE']:.4f}。内层选择流程 MAE={primary.MAE:.4f}。第二问观察最优 {best2}，平均折 MAE={q2g.loc[best2,'MAE']:.4f}；所有方法保留。第三问按平均验证 SWD 选择 {summary['production_model']}；生成质量与条件一致性存在取舍，逐方法测试表见报告 03。

关键未解决事项：匿名评分行序缺少独立键级证明；barnyard 标签疑点；两项 acetaldehyde 缺少监督；161 项未校准；地板以下浓度未知；独立风格评价器仅 40 款；风险分位没有覆盖率校准；没有真实新酿造/添加/盲评验证。深度模型与完整前向选择没有获得预期优势，已如实保留。

论文为 AI 辅助研究参考。官方规定要求核心建模、算法设计、求解与分析由队员独立完成，审阅签字不能替代独立工作。本轮没有伪造人工修改记录，队员须独立重做和审定关键科学判断，并补充真实记录后决定正式提交内容。

一键复现：powershell -ExecutionPolicy Bypass -File scripts/run_rebuild.ps1。脚本不写死机器路径，python、pandoc、xelatex 依次按 -Python 参数、PATH、常见安装位置解析，也可用 MC_PANDOC、MC_XELATEX 指定。论文构建需要 Pandoc 与 XeLaTeX。
'''
 reports={'00_data_audit.md':audittext,'01_q1_analysis.md':q1text,'02_q2_analysis.md':q2text,'03_q3_analysis.md':q3text,'04_innovation_and_ablation.md':innovation,'05_final_summary.md':final}
 legacy=R/'reports/legacy_reference';legacy.mkdir(exist_ok=True)
 for name,text in reports.items():
  p=R/'reports'/name
  if p.exists() and not (legacy/name).exists():shutil.copy2(p,legacy/name)
  p.write_text(text,encoding='utf-8')
 abstract=f'''针对啤酒风味评分、关键化合物筛选与条件生成中的量表歧义、变量共线及小风格样本问题，本文构建以观测语义和独立评价为主线的求解流程。首先核验 250 款啤酒的 231 项化学特征，并将 2197 条逐评分员记录映射至啤酒，构造评分员等权的真实 0—6 面板均分。评分预测比较线性、树与条件层次网络，在两次三折内部验证中，观察最优 {best} 的 MAE 为 {q1g.loc[best,'MAE']:.4f}，RMSE 为 {q1g.loc[best,'RMSE']:.4f}；结构网络未获得足以替代强树基线的优势。其次，将共表达图、三变量条件选择和风格内 bootstrap 联合用于 50×3 化合物分析，报告单分子与替代组稳定性；{best2} 获得最低平均折误差，但完整前向选择的优势未获支持。最后，以收缩条件边际与 Copula 对最低值点质量建模，对比连续高斯和 CVAE，并用独立 40 款酒训练的评价器检验迁移。结果表明，联合分布误差、风格一致性和缺陷风险存在取舍；满足设计预测器的约束不等于独立评价下缺陷不恶化。本文交付 22 风格共 880 个统计生成谱、感官画像及受约束迁移实例，同时标明量纲、地板浓度与缺失阈值的不可验证边界。该成果为 AI 辅助参考，关键科学判断仍需参赛队独立完成与验证。'''
 intro=r'''# 问题分析与研究路线

啤酒的感官画像既取决于化学组成，也取决于评分卡的观察方式。第一问预测的量表决定第二问的重要性含义，第二问的共变关系又影响第三问是否能生成合理的联合谱；因此三个任务不能简单分别套用回归、排序和生成网络。本文将统一主线确定为：先识别可观测量，再建立预测与分布模型，最后在未参与优化的评价环节检验结果。

原论文 [1] 比较十类预测算法并通过真实添加试验检验候选成分；本赛题新增 0—6 分级与跨风格约束生成，评估对象也不同。本文使用作者公开数据 [2]，不把原文的消费者喜好预测指标与本轮 50 维原始面板均分误差直接比较。网页参考版引入真实评分和概率生成具有参考价值，但原先的记录均值、同训练集评价器与后处理前记忆距离需要重新核查。MC2026A 中部分预处理与代理约束问题也不被继承。

建模假设包括：匿名原始评分按附件物理行序映射；评分员等权面板均分作为可重复监督对象；化学缺失按训练中位数作工作性处理；有限样本的风格经验分布可通过总体收缩改善方差。模型没有假定所有最小值均为同一个已知检出限，也没有假定任意统计生成谱都可由酿造过程实现。上述假设分别通过映射相关、重复权重、随机缺失和生成分布诊断检验。

# 符号与数学定义

设 x_i 为第 i 款酒在附件编码尺度上的 231 维谱，s_i 为风格，y_irt 为评分员 r 在重复 t 的评分。原始监督目标定义为

$$\bar y_i=\frac{1}{R_i}\sum_{r=1}^{R_i}\left(\frac{1}{T_{ir}}\sum_t y_{irt}\right).$$

连续预测的误差按啤酒与目标等权计算。离散输出为 $g_i=\min(6,\max(0,\lfloor\hat y_i+0.5\rfloor))$。模型均分与个体评分是不同随机量，论文中的整数准确率始终针对舍入后的面板均分。

层次模型对强度 I 与勾选 B 分别建模，但组合时使用条件期望：

$$E[IB\mid x]=\sum_{k=0}^{6} k\,P(I=k\mid x)P(B=1\mid I=k,x).$$

这个等式直接对应评分卡运算，并处理父项为零时无法识别勾选状态的问题。化合物替代组表示相近的统计信息，不表示分子可在真实配方中自由替代。

条件经验边际写作 $\widehat F_{js}=w_s\widehat F_{j,s}+(1-w_s)\widehat F_{j,all}$，其中 $w_s=n_s/(n_s+10)$。联合采样为 $z\sim N(0,\widehat R)$，$x_j=\widehat F_{js}^{-1}(\Phi(z_j))$。独立消融令 $\widehat R=I$。其中逆分布的平坦区间保留观测最小值的点质量。

迁移优化在有限候选集中最小化到目标中心的距离与改变量之和；缺陷约束应用于明确观测的化学编码以及设计预测器的 stale_hops 变化。独立评价器只评价已选候选，避免优化器利用同一个评价函数制造表面成功。
'''
 refs='''# 参考文献

[1] Schreurs M, Piampongsant S, Roncoroni M, et al. Predicting and improving complex beer flavor through machine learning. Nature Communications, 15:2368, 2024. DOI: 10.1038/s41467-024-46346-0.

[2] Schreurs M, et al. Data from: Predicting and improving complex beer flavor through machine learning, version v4. Zenodo, https://zenodo.org/records/23624915, 访问日期 2026-10-28.

[3] MathorCup 组委会. 2026 年 MathorCup 大数据挑战赛赛道 A 初赛题、论文格式及提交规范、人工智能工具使用规定（试行）. 本地官方附件, 2026.

[4] Geurts P, Ernst D, Wehenkel L. Extremely randomized trees. Machine Learning, 63:3–42, 2006. DOI: 10.1007/s10994-006-6226-1.

[5] Ledoit O, Wolf M. A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88(2):365–411, 2004. DOI: 10.1016/S0047-259X(03)00096-4.

[6] Kingma D P, Welling M. Auto-Encoding Variational Bayes. arXiv:1312.6114, 2013, https://arxiv.org/abs/1312.6114, 访问日期 2026-10-28.

[7] OpenAI Codex，GPT-6 系列（接口未提供可独立核验的完整模型快照标识），OpenAI，2026-10-28。用于本参考工程的审计、代码、实验、图表与文本辅助。

# AI 使用声明与队员独立工作记录

本参考稿的审计、模型实现、实验运行、图表和文字由 AI 辅助生成，正文各实验与图表均具有该贡献。详细说明见 reports/AI工具使用详情说明.pdf 和 AI_usage_log.md。未记录任何已经完成的队员独立建模、人工修改或盲评，因此不能宣称核心工作由队员独立完成。正式参赛作品应遵守官方规定，由队员独立完成核心设计、求解和分析，并如实记录采纳、修改与弃用内容；本稿不冒充符合该独立性要求的正式提交稿。
'''
 paper='# 观测语义约束下的啤酒风味预测、化合物筛选与条件生成\n\n## 摘要\n\n'+abstract+'\n\n关键词：逐评分员监督；层次评分；稳定选择；混合边际；独立评价\n\n'+intro+'\n'+audittext+figure('01_data','风格样本量及观测最低值堆积；AI 辅助计算绘图')+'\n'+q1text+figure('02_q1_accuracy','真实评分尺度上的模型误差与困难目标；AI 辅助')+figure('03_q1_curves','内层训练与验证曲线，六个外层任务平均；AI 辅助')+'\n'+q2text+figure('04_q2_selection','三因素预测与替代组稳定性；AI 辅助')+figure('05_q2_network','有符号共表达图，未绘孤立节点；AI 辅助')+'\n'+q3text+figure('06_q3_quality','三次划分下生成误差，误差棒为划分标准差；AI 辅助')+figure('07_migration','独立评价器下的迁移变化；AI 辅助')+figure('08_generated_portraits','全样本重拟合后的展示性感官画像，非独立验证；AI 辅助')+'\n'+innovation+figure('09_sensitivity','额外缺失敏感性与 CVAE 重构曲线；AI 辅助')+'\n# 模型评价与结论\n\n'+final.split('\n',1)[1]+'\n'+refs
 p=R/'paper/competition_draft.md';legacy_p=R/'paper/legacy_reference.md'
 if p.exists() and not legacy_p.exists():shutil.copy2(p,legacy_p)
 p.write_text(paper,encoding='utf-8');(R/'paper/rebuild_body.md').write_text(paper.split(intro)[0] if False else paper[paper.index('# 问题分析'):],encoding='utf-8');(R/'paper/rebuild_abstract.txt').write_text(abstract,encoding='utf-8')
 append='# 附录：逐感官 Top-3 与代码索引\n\n以下为预先指定 Forward3 的候选。其他方法及入选概率见 CSV；所有条目为关联候选。\n\n';t=top.pivot(index='target',columns='rank',values='chemical').reset_index();t.columns=['感官','候选1','候选2','候选3'];append+=md(t);(R/'paper/rebuild_appendix.md').write_text(append,encoding='utf-8')
 # Append truthful usage entry once; human cells stay pending.
 ai=R/'reports/AI_usage_log.md';entry='''\n\n## 2026-10-28 独立重构实际记录\n\n工具：OpenAI Codex / GPT-6 系列，完整快照未提供；Python、PyTorch、scikit-learn、Pandas、Matplotlib、Pandoc、XeLaTeX。用户指令：独立审计并实际完成三问，冻结协议，不预设旧结果正确。AI 实际完成数据核验、原始评分目标、模型代码、运行与实现修正、统计、图表及中文参考稿。关键修正：深度模型外层重训、生成度量共享随机方向。训练及结果日志位于 outputs/rebuild/logs；所有新源码头部标明 AI 辅助。\n\n人工修改及核心独立工作：尚无可核实记录，不填写为已完成。采用或弃用决策需队员独立作出。官方规定要求核心建模、算法、求解与分析由队员独立完成，简单审阅或签字不能替代该要求。本产物是研究参考，不能声称其 AI 主导核心已满足该规定。\n'''
 if '## 2026-10-28 独立重构实际记录' not in ai.read_text(encoding='utf-8'):ai.write_text(ai.read_text(encoding='utf-8')+entry,encoding='utf-8')
 csv([dict(decision=d,status='待队员独立完成与复核',reviewer='',date='',evidence='') for d in ['匿名评分映射','面板监督目标','结构模型采用或弃用','Top-3最终规则','缺陷阈值解释','生成及迁移可行性','正式论文独立重写']],'audit/human_decision_record.csv')
 js({'_scope':'LEGACY single-split / single-fold record kept only for provenance. Authoritative closed-experiment numbers live in outputs/rebuild/closed/final_summary.json (22-style x multi-context macro). Do not cross-cite the two.','_authoritative_file':'outputs/rebuild/closed/final_summary.json','q1_observed_best':best,'q1_metrics':q1g.loc[best].to_dict(),'q1_MAE_repeat_sd':q1sd[best],'q1_inner_selected':primary.to_dict(),'q2_observed_best':best2,'q2_metrics':q2g.loc[best2].to_dict(),'q3_production':summary['production_model'],'q3_metrics':q3g.to_dict('index'),'migration_unique_beers':int(mig.beer.nunique()),'migration_summary':migg.to_dict('index')},'summary.json');log('reports and full paper text built')
if __name__=='__main__':main()
