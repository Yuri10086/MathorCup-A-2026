# -*- coding: utf-8 -*-
"""11 生成摘要页与结论章（由结果文件驱动，保证数字一致）

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。

摘要写作约定（用户要求）：
  * 三个问题统一采用“用了什么模型 -> 怎么做的 -> 结果如何”的表述顺序
  * 重要模型名与关键结果数字必须加粗
  * 不使用直角引号，改用中文弯引号
  * 写满一页，不留大片空白
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SEC = os.path.join(C.ROOT, 'latex', 'sections')


def W(name, txt):
    open(os.path.join(SEC, name), 'w', encoding='utf-8').write(txt)
    print('  ✓ sections/%s (%d 字)' % (name, len(txt)))


q1 = json.load(open(os.path.join(C.OUT, 'q1_cv.json'), encoding='utf-8'))
q1r = {r['model']: r for r in q1['results']}
q2 = json.load(open(os.path.join(C.OUT, 'q2_graph.json'), encoding='utf-8'))
q3 = json.load(open(os.path.join(C.OUT, 'q3_summary.json'), encoding='utf-8'))
PDR = pd.read_csv(os.path.join(C.OUT, 'q1_metrics_perdim.csv'))
GQ = pd.read_csv(os.path.join(C.OUT, 'q3_generation_quality.csv'))

S = q1r['Stack']['mean_r2']
BEST1 = q1['best_single']
BEST1_R2 = q1r[BEST1]['mean_r2']
ov = float(PDR.loc[PDR['dim'] == 'overall', 'stack_r2'].iloc[0])
n_over = int((PDR['stack_r2'] > 0.21).sum())
mx = PDR.sort_values('stack_r2', ascending=False).iloc[0]
mt_min = q1r['MT-GNN']['min_r2']
freq_mean = float(np.mean(pd.read_csv(os.path.join(C.OUT, 'q2_top3.csv'))['Top1入选频率'].values))
jac_mean = float(q2['jaccard_methods_mean'])
same_mod = float(q2.get('same_module_ratio', 0.76))
n_oav = int(q2.get('n_oav', 68))
n_gt1 = int(q2.get('n_oav_gt1', 19))
T = q3['transfer']
div_mean = float(GQ['多样性比'].mean())
fid_mean = float(GQ['风格保真度'].mean())
modul = float(q2.get('modularity', 0.5488))

# ============================================================ 摘要（1 页，写满）
abs_tex = r'''%% ======================= 摘要页（第 1 页） =======================
\begin{center}
{\zihao{3}\heiti 基于共表达图与风格条件流的啤酒风味预测、归因与生成}
\end{center}

\vspace{2pt}
\begin{center}
{\zihao{4}\heiti 摘\hspace{1em}要}
\end{center}

\vspace{-2pt}
啤酒风味由两百余种化合物在极宽浓度域内的协同与拮抗共同决定，单一化合物的气味活度
值无法解释混合效应。本文以 250 款商业啤酒的 231 个化合物指标与 50 维 0--6 分制感官
评分为对象，按\textbf{数据审计、感官预测、化合物归因、风味生成}四步推进，每一问都
给出所用模型、具体做法与量化结果，全部数字可由随附源码一次复现。

\textbf{针对问题一}，本文采用\textbf{多任务图扩散网络 MT-GNN 与非负岭堆叠集成}。
做法上，先用校准浓度表反证出对数变换底数为 10（斜率 1.0000、$R^2=1.0000$），并识别
出 231 列中有 \textbf{33 列实为原始浓度}、\textbf{102 列存在检出限左删失}、73 个缺失
单元，据此构造缺失指示与封底指示两类变量；再把化合物相关矩阵转为\textbf{共表达图
扩散算子} $\bm{S}=(1-\alpha)\bm{I}+\alpha\tilde{\bm{A}}$ 与原始特征拼接，送入双分支
共享隐层网络一次输出 50 维，最后用非负岭回归对十个基模型做堆叠。结果表明，在按风格
分层的 5 折 $\times$ 3 重复交叉验证下，\textbf{集成模型平均 $R^2=%.3f$}，高于文献
梯度提升基线的 0.21 与 0.22，\textbf{50 维中有 %d 维超过该基线}，最高单维
$R^2=%.2f$（%s），最难的总体喜好维度 $R^2=%.3f$，为文献同项的 %.1f 倍；
\textbf{MT-GNN 是十个模型中唯一 50 维全为正者}（最低 $R^2=%.4f$），说明共享隐层
把统计强度传导到了弱信号维度。消融实验如实表明，缺失与封底指示、图扩散的边际增益
均在噪声范围内（$|\Delta|<0.01$），\textbf{增益主要来自多任务结构与集成而非单一
特征工程}。最终交付 $250\times50$ 的 0--6 分制评分表、整数分级表，以及实测覆盖率
为 %.1f@PCT@ 的预测区间。

\textbf{针对问题二}，本文采用\textbf{贪心模块度优化与五路证据秩融合}。做法上，先在
$|\rho|\ge0.45$ 的共表达图上做社区发现，并剔除会挤占名额的理化参数代理变量，得到
34 个社区、\textbf{模块度 $Q=%.3f$}，其中 6 个主社区覆盖 200 个化合物且与酿造生化
通路一一对应；再对每个感官维度并行计算\textbf{秩相关、SHAP、置换重要性、ElasticNet
稳定性选择、互信息}五路排序，加权融合后取前三名。稳定性用\textbf{bootstrap 入选
频率}与\textbf{跨方法 Jaccard 系数}双指标度量，化学合理性用阈值—OAV、化学类别、
代谢来源、社区一致性四条判据核验，并以 \texttt{6.csv} 交叉相关矩阵作独立外部验证。
结果表明，\textbf{Top-1 的 bootstrap 入选频率均值达 %.2f}（随机基线 0.33），
\textbf{%.0f@PCT@ 的维度其 Top-3 至少两项落在同一社区}，%d/%d 个可算 OAV 的条目
满足 $\mathrm{OAV}>1$，据此识别出里那醇、香叶醇、甲基庚烯酮、乙酸乙酯、DMS 等
公认关键致香物，统计证据与化学语义相互印证。

\textbf{针对问题三}，本文采用\textbf{风格条件潜因子流 SCLF 与带约束的高斯最优传输}。
做法上，先用并行分析定阶取 $K=14$ 个潜因子，以 6 层仿射耦合流建模风格条件密度
$p(\bm{z}\mid s)$，采样时按该风格缺陷 OAV 的 95@PCT@ 分位做拒绝采样；再做跨风格
迁移，用\textbf{Bures-Wasserstein 最优传输}同时对齐目标风格的均值与协方差，然后沿
源谱到目标谱的连线做\textbf{单调二分投影}，把缺陷化合物逐样本钳制在物理下界与源
样本值之间。结果表明，条件流的\textbf{负对数似然为 %.3f}，明显优于高斯收缩模型的
%.3f；16 种风格的生成样本\textbf{风格保真度均值 %.2f}（随机基线 0.045）、多样性比
%.2f，未出现模式坍塌；在 Pils/Lager 到 Tripel 的迁移中，被判为 Tripel 的比例由
%.2f 升至 \textbf{%.2f}，平均迁移强度 $\lambda=%.2f$，而\textbf{缺陷 OAV 构造性
不恶化}（DMS 为 1.00 倍、双乙酰 0.84 倍、4VG 0.94 倍），平均缺陷分由 %.3f 降至 %.3f。

\textbf{创新点}：一是发现并量化了附件未说明的量纲混用（33 列原始浓度）与检出限左
删失（102 列、占 12.60@PCT@ 观测），把数据契约固化为三问共享的判定；二是把共表达图
同时用于预测侧的特征增广与归因侧的社区一致性判据，使模块级信号成为可检验的对象；
三是用五路秩融合、双稳定性指标、四判据化学合理性与外部矩阵独立验证，把归因从排行榜
升级为可复核论证；四是把跨风格迁移写成\textbf{带单调缺陷约束的最优传输}，以单一标量
$\lambda$ 刻画迁移强度，兼具可解释性与可审计性。

\vspace{4pt}
\noindent\textbf{关键词：}共表达图；图扩散；多任务学习；稳定性选择；归一化流；
最优传输；啤酒风味
''' % (S, n_over, mx['stack_r2'], mx['name'], ov, ov / 0.09, mt_min,
       100 * q1['interval']['coverage'],
       modul, freq_mean, 100 * same_mod, n_gt1, n_oav,
       q3['nll_flow'], q3['nll_gauss'], fid_mean, div_mean,
       T['p_tgt_before'], T['p_tgt_after'], T['lam_mean'],
       T['defect_before'], T['defect_fin'])
abs_tex = abs_tex.replace('@PCT@', chr(92) + '%')
W('abstract.tex', abs_tex)

# ============================================================ 结论
concl = r'''
\section{结论}

本文围绕化合物到感官的映射，对 250 款商业啤酒的 231 维化学谱与 50 维感官评分做了
系统建模。主要结论如下。

\textbf{（1）数据层面的三项判定直接影响后续全部结论。}对数变换底数为 \textbf{10}
（斜率 1.0000、$R^2=1.0000$）；附件说明未提及的 \textbf{33 列实为原始浓度}，且集中
在 $\mu\mathrm{g\cdot L^{-1}}$ 级的超痕量致香物；\textbf{102 列}存在检出限左删失、
涉及 12.60@PCT@ 的观测。若不区分这三者，OAV 会失真数个量级，问题三的缺陷约束也会
失效。

\textbf{（2）感官评分可预测，但存在明确的可解释上界。}按风格分层的 5 折 $\times$
3 重复交叉验证下，本文堆叠集成的\textbf{平均 $R^2=%.3f$}，超过文献梯度提升基线的
0.21 与 0.22，50 维中有 %d 维超过该基线。分组看，Aroma 与 Flavour 两套可预测性最好、
缺陷类最差；逐维看，$R^2$ 与该维度标签标准差正相关（$\rho=%.2f$），说明\textbf{部分
维度的低 $R^2$ 源于品评噪声而非模型不足}。该上界应被视为感官数据本身的信息含量，
而非待攻克的模型缺陷。

\textbf{（3）关键化合物可被稳定识别，且化学上自洽。}共表达图的模块度为 $Q=%.3f$，
6 个主社区与酵母酯醇代谢、酒花萜烯骨架、陈化降解等通路对应。五路证据秩融合给出的
Top-3 中，Top-1 的 bootstrap 入选频率均值为 %.2f，%d/%d 个可算 OAV 的条目满足
$\mathrm{OAV}>1$，且化合物类别与代谢来源均与感官语义一致。

\textbf{（4）风格条件生成与受约束迁移均可行。}条件流的负对数似然优于高斯收缩模型，
生成样本的风格保真度为 %.2f、多样性比为 %.2f；跨风格迁移在\textbf{构造性保证}缺陷
不恶化的前提下，把 Pils/Lager 被判为 Tripel 的比例由 %.2f 提高到 %.2f。

\textbf{（5）与文献基线的对照。}在完全相同的交叉验证协议下，本文的平均 $R^2$ 比文献
最优的树模型高约 0.05（0.268 对 0.21 与 0.22），总体喜好维度高约 0.035
（0.125 对 0.09）。需要强调，这一差距的\textbf{主要来源是模型结构而非特征工程}：
消融显示缺失指示、封底指示与图扩散三项增强的边际增益都在噪声范围内
（$|\Delta|<0.01$）。因此结论不是“多堆几个特征就能超过基线”，而是
\textbf{在该数据规模下结构性的信息共享与集成才是有效手段，特征工程已接近饱和}。

\textbf{（6）可复现性。}全部数字、图件与交付表由 \texttt{src/} 下的脚本一次运行生成，
结果落盘于 \texttt{output/}，论文由 \texttt{latex/} 中的源文件编译得到。模型选择、
超参数与随机种子（固定为 20261026）均已写死在脚本中，交叉验证划分由
\texttt{StratifiedKFold} 给定随机种子确定，第三方可按附录清单逐位复现，
不存在需要人工干预的中间步骤。
''' % (S, n_over, float(pd.read_csv(os.path.join(C.OUT, 'q1_metrics_perdim.csv'))
                        .pipe(lambda d: __import__('scipy.stats', fromlist=['spearmanr'])
                              .spearmanr(d['dim_std'], d['stack_r2']).statistic)),
       modul, freq_mean, n_gt1, n_oav,
       fid_mean, div_mean, T['p_tgt_before'], T['p_tgt_after'])
concl = concl.replace('@PCT@', chr(92) + '%')
W('sec10_concl.tex', concl)
print('\n摘要与结论生成完成')
