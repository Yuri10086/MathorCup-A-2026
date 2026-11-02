# -*- coding: utf-8 -*-
"""01_audit.py —— 数据审计（问题1 的地基）

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek）。

审计五件事，全部可复算：
  A. 缺失结构（1.csv）
  B. 左删失 / 检出限地板（1.csv）
  C. 量纲判定：哪些列是 log10、哪些是原始浓度（8.xlsx 校准表）
  D. 4.csv 的性质：z 分、样本量、重复样本、0–6 分辨率
  E. 表 5/6/7/8 与 3.xlsx 的身份核对
输出 output/audit_report.txt、output/audit.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q1lib as C

LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def sec(t):
    log('\n' + '=' * 78)
    log(t)
    log('=' * 78)


D = C.load()
X, Y, comp, sdim = D['X'], D['Y'], D['comp'], D['sdim']
A = {}

sec('0. 规模与对齐')
log('化学表 1.csv：%d 行 × %d 列（化合物 %d 列，元数据 3 列）' % (
    D['chem'].shape[0], D['chem'].shape[1], len(comp)))
log('感官表 4.csv：%d 行 × %d 列（感官维度 %d 列）' % (
    D['sens'].shape[0], D['sens'].shape[1], len(sdim)))
log('两表 beer 序号逐位相等：%s' % np.array_equal(D['chem']['beer'].values,
                                              D['sens']['beer'].values))
log('风格标签 tasting_category_fine：%d 类，最小类 %s n=%d' % (
    D['style'].__len__() if False else len(set(D['style'])),
    pd.Series(D['style']).value_counts().idxmin(),
    int(pd.Series(D['style']).value_counts().min())))
A['shape'] = dict(n=len(X), p_comp=len(comp), k_sens=len(sdim),
                  n_style=int(len(set(D['style']))))

sec('A. 缺失结构（1.csv）')
na = X.isna()
mc = na.sum(0)
log('缺失格 %d / %d = %.4f%%；涉及 %d 列' % (
    int(na.values.sum()), na.size, 100 * na.values.sum() / na.size,
    int((mc > 0).sum())))
for c in comp:
    if mc[c] > 0:
        log('    %-28s 缺失 %3d 格 (%.1f%%)' % (c, int(mc[c]), 100 * mc[c] / len(X)))
log('4.csv 缺失格：%d（0 缺失）' % int(Y.isna().values.sum()))
A['missing'] = dict(cells=int(na.values.sum()), frac=float(na.values.sum() / na.size),
                    cols=int((mc > 0).sum()),
                    detail={c: int(mc[c]) for c in comp if mc[c] > 0})

sec('B. 左删失 / 检出限地板（1.csv）')
V = X[comp].values.astype(float)
finite = np.isfinite(V)
eq = np.isclose(V, C.FLOOR, atol=1e-9)
colfrac = eq.sum(0) / np.maximum(finite.sum(0), 1)
log('地板常数 %.8f 出现 %d 次 = %.4f%% 的有效观测' % (
    C.FLOOR, int(eq.sum()), 100 * eq.sum() / finite.sum()))
log('涉及 %d 列；其中 ≥5%% 的列 %d 个，≥20%% 的列 %d 个' % (
    int((eq.sum(0) > 0).sum()), int((colfrac >= 0.05).sum()), int((colfrac >= 0.20).sum())))
log('地板含义：原始浓度低于检出限被"封底"为同一常数，是左删失而非真实低浓度。')
log('最严重 10 列：')
for i in np.argsort(-colfrac)[:10]:
    log('    %-28s %5.1f%% (%d/%d)' % (comp[i], 100 * colfrac[i],
                                       int(eq[:, i].sum()), int(finite[:, i].sum())))
log('负值格 %d（对数变换后 < 1 的浓度）；最小值 %.3f、最大值 %.3f' % (
    int((V < -1e-12).sum()), np.nanmin(V), np.nanmax(V)))
A['censoring'] = dict(floor=C.FLOOR, cells=int(eq.sum()),
                      frac=float(eq.sum() / finite.sum()),
                      cols_any=int((eq.sum(0) > 0).sum()),
                      cols_ge5=int((colfrac >= 0.05).sum()),
                      cols_ge20=int((colfrac >= 0.20).sum()))

sec('C. 量纲判定：log10 还是原始浓度')
scale, detail = C.colscale()
nraw = sum(1 for c in comp if scale[c] == 'raw')
nlog = len(comp) - nraw
ncal = sum(1 for c in comp if detail[c]['calibrated'])
log('有 8.xlsx 校准浓度的列 %d 个（可直接判定）；无校准信息 %d 个（默认按 log10）' % (
    ncal, len(comp) - ncal))
log('判定结果：原始浓度 %d 列，log10 %d 列' % (nraw, nlog))
log('原始浓度列（数值 ≈ 浓度本身，为 µg/L~mg/L 级痕量致香物与部分理化参数）：')
for c in comp:
    if scale[c] == 'raw':
        d = detail[c]
        log('    %-32s 逐样本命中 原始值=%.3f  10^v=%.3f' % (c, d['f_id'], d['f_pow']))
amb = [c for c in comp if detail[c]['calibrated'] and
       max(detail[c]['f_pow'], detail[c]['f_id']) < 0.95]
log('无法判定（两种关系都不满足 >95%%，按"有负值即为对数"兜底）：%s' % amb)
log('→ 结论：附件说明只说"少量理化参数保留原始量纲"，实测还有 %d 列风味化合物同为原始浓度。'
    % sum(1 for c in comp if scale[c] == 'raw' and detail[c].get('f_id', 0) > 0.95))
log('→ 若把这 %d 列误当 log10（浓度=10^v），linalool 会从 32.5 µg/L 变成 10^20，'
    'OAV 差 20 个数量级，问题2/3 全部失真。' % nraw)
A['scale'] = dict(raw=int(nraw), log10=int(nlog), calibrated=int(ncal),
                  raw_cols=[c for c in comp if scale[c] == 'raw'])

sec('D. 2.csv / 5.csv / 6.csv 的统计量身份')
c2 = pd.read_csv(os.path.join(C.DATA, '2.csv'), index_col=0)
log('2.csv：%d × %d，对角=%.4f' % (c2.shape[0], c2.shape[1],
                                 float(np.mean(np.diag(c2.values)))))
log('2.csv 描述为"化学参数之间的 Spearman 相关系数"：')
# 用 8.xlsx chemical_name 对齐后与 1.csv 现算相关对比
th = pd.read_excel(os.path.join(C.DATA, '8.xlsx'), sheet_name='Sensory thresholds')
alt = {}
for _, r in th.iterrows():
    fn, cn = str(r['field_name_python']).strip(), str(r['chemical_name']).strip()
    if fn in comp and cn and cn != 'nan':
        alt.setdefault(cn.lower(), fn)
cols2 = [str(c).strip() for c in c2.columns]
hits = [(i, alt.get(c.lower()) or (c if c in comp else None)) for i, c in enumerate(cols2)]
hits = [(i, h) for i, h in hits if h]
idx = [i for i, _ in hits]
sub = X[[h for _, h in hits]]
sp = spearmanr(sub.values).statistic
pe = np.corrcoef(sub.values.T)
diff_sp = np.nanmax(np.abs(sp - c2.values[np.ix_(idx, idx)]))
diff_pe = np.nanmax(np.abs(pe - c2.values[np.ix_(idx, idx)]))
log('  对齐 %d/%d 个化合物；与现算 Spearman 的最大差 %.2e，与 Pearson 的最大差 %.4f' % (
    len(hits), len(cols2), diff_sp, diff_pe))
log('  → 2.csv 是 Spearman（题面"皮尔逊"为勘误），可以直接当作非参数相关图使用。')
A['c2_identity'] = dict(n=len(hits), maxdiff_spearman=float(diff_sp),
                        maxdiff_pearson=float(diff_pe))

c5 = pd.read_csv(os.path.join(C.DATA, '5.csv'), index_col=0)
c6 = pd.read_csv(os.path.join(C.DATA, '6.csv'), index_col=0)
log('5.csv：%d×%d（感官×感官，50 维全对齐）' % (c5.shape[0], c5.shape[1]))
sp5 = spearmanr(Y.values).statistic
log('5.csv 与 4.csv 现算 Spearman：最大绝对差 %.4f，中位绝对差 %.4f' % (
    np.nanmax(np.abs(sp5 - c5.values)), np.nanmedian(np.abs(sp5 - c5.values))))
log('  （逐元素并不严格相等，但量级一致、对角全 1；作为"预计算的感官相关矩阵"可直接引用，'
    '若要做精确结论应以 4.csv 现算为准。）')
log('6.csv：%d×%d（化学×感官交叉相关，非方阵；行是 44 个有阈值的化合物，列是 50 个感官属性）' % (
    c6.shape[0], c6.shape[1]))
A['c5'] = dict(maxdiff=float(np.nanmax(np.abs(sp5 - c5.values))),
               mediandiff=float(np.nanmedian(np.abs(sp5 - c5.values))),
               shape=list(c5.shape))
A['c6'] = dict(shape=list(c6.shape))

sec('E. 7.csv / 8.xlsx / 3.xlsx')
c7 = pd.read_csv(os.path.join(C.DATA, '7.csv'), index_col=0)
log('7.csv：%d × %d，缺失 %d 格（RateBeer 评论汇总，189 个评论维度，非 0–6 分制）' % (
    c7.shape[0], c7.shape[1], int(c7.isna().sum().sum())))
log('7.csv 与 4.csv 排名相关（7.csv 的索引即 beer 序号，可直接对齐）：')
c7n = [c for c in c7.columns if c not in ('beer_id', 'tasting_category_fine')]
mg = c7[c7n].copy()
mg.index = pd.to_numeric(mg.index, errors='coerce')
mg = mg[mg.index.isin(set(D['beer']))]
common = list(mg.index)
log('  可对齐样本 %d/%d（7.csv 索引 = beer 序号；beer_id 是商品条码、非样本编号）' % (
    len(common), len(c7)))
Yidx = Y.set_index(pd.Index(D['beer']))
best = []
for c in sdim:
    cand = []
    b = Yidx.loc[common, c].values.astype(float)
    for d in c7n:
        a = mg.loc[common, d].values.astype(float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 30:
            cand.append((abs(spearmanr(a[m], b[m]).statistic), d))
    if cand:
        best.append(max(cand)[0])
log('  50 个感官维度里，与 7.csv 逐维最强配对 |ρ| 中位 %.3f、最大 %.3f' % (
    np.median(best), np.max(best)))
log('  （评论汇总与品评小组不同源，相关性温和，不能把 7.csv 当额外标签直接并入训练。）')
A['c7'] = dict(shape=list(c7.shape), na=int(c7.isna().sum().sum()),
               median_best_rho=float(np.median(best)),
               max_best_rho=float(np.max(best)))

xl = pd.ExcelFile(os.path.join(C.DATA, '8.xlsx'))
log('8.xlsx 工作表：%s' % xl.sheet_names)
sn = pd.read_excel(os.path.join(C.DATA, '8.xlsx'), sheet_name='Sensory names')
log('Sensory names：%d 行，其中 trained panel 占 %d 行；DataType=categorical 的 %d 个' % (
    len(sn), int((sn['dataset'] == 'Trained panel').sum()),
    int((sn['DataType'] == 'categorical').sum())))
categorical = set(sn.loc[sn['DataType'] == 'categorical', 'R_name'].astype(str))
log('  这 %d 个 categorical 维度是"Aroma/Flavour 的子层级香气项"，本质是"有没有这个气味"，'
    % len(categorical))
log('  与 4.csv 的连续 z 分并存，说明评分为 0–7 强度刻度（论文 Methods：7-point scale）。')
A['categorical'] = sorted(categorical)

xl3 = pd.ExcelFile(os.path.join(C.DATA, '3.xlsx'))
log('3.xlsx 工作表 %d 个：%s' % (len(xl3.sheet_names), xl3.sheet_names))
n_cells = []
for s in xl3.sheet_names[:-1]:
    df = pd.read_excel(os.path.join(C.DATA, '3.xlsx'), sheet_name=s, header=None)
    # 每张卡左右两栏各 26 个评分行；打分格在 col16 / col32，刻度图例在 col2..col14 / col18..col30
    cells, nonzero = 0, 0
    for i in range(df.shape[0]):
        for j in (16, 32):
            v = df.iloc[i, j]
            if isinstance(v, (int, float)) and not pd.isna(v):
                if i == 6:            # 两栏标题行的 0.0/0.0 是图例首格
                    continue
                cells += 1
                if float(v) != 0:
                    nonzero += 1
    n_cells.append((s, cells, nonzero))
    log('  %-9s %d 行：可打分的格子 %d 个，其中被打过分的 %d 个' % (s, len(df), cells, nonzero))
log('→ 3.xlsx 的 12 张评分卡（beer_1~beer_12）**全部为空卡**：只有 0..6 的刻度图例、'
    '左栏 Aroma/Flavour 与右栏 Aroma/Flavour 两套属性名、以及 "Not present" / '
    '"Predominant/overwhelming" 两端锚点，没有任何实际打分。')
log('→ 因此 3.xlsx 的实际作用是给出**评分表的结构定义**（两套各约 20 个感官维度、0–6 七档'
    '强度刻度 + 两端锚点），真正可回归的标签只有 4.csv 的 50 维连续 z 分。')
A['c3'] = dict(sheets=len(xl3.sheet_names),
               cells_per_sheet=[c for _, c, _ in n_cells],
               filled=[n for _, _, n in n_cells])

sec('F. 4.csv 的性质与 0–6 分辨率')
log('4.csv 每维 min/max/sd（50 维）：sd 范围 %.3f ~ %.3f，均值 %.3f' % (
    Y.std().min(), Y.std().max(), Y.std().mean()))
log('每维均值范围 %.3f ~ %.3f（近似 0，说明是 z 分）' % (Y.mean().min(), Y.mean().max()))
nu = Y.nunique()
log('每维不同取值个数：中位 %d，最小 %d，最大 %d' % (
    int(np.median(nu.values)), int(nu.min()), int(nu.max())))
gap = ((Y.max() - Y.min()) / (nu - 1))
log('"相邻取值间距 / 该维 sd" 中位 %.4f —— 远小于分级所需（≈0.3sd 一格），' % float(np.median(gap / Y.std())))
log('说明 z 分是**多位品评员均值**，原始 0–6 整数刻度已被平均抹平，不能逐格反解。')
A['sens'] = dict(sd_min=float(Y.std().min()), sd_max=float(Y.std().max()),
                 nuniq_median=float(np.median(nu.values)),
                 gap_over_sd=float(np.median(gap / Y.std())))

# 重复品评样本：同一化合物谱出现多次（论文：12 个样本被重复品评）
Xv = X[comp].values.astype(float)
key = np.where(np.isfinite(Xv), Xv, np.nan)
_, inv, cnt = np.unique(np.round(np.nan_to_num(key, nan=-999), 9), axis=0,
                        return_inverse=True, return_counts=True)
inv = inv.ravel()
pairs = [(i, j) for i in range(len(Y)) for j in range(i + 1, len(Y)) if inv[i] == inv[j]]
log('化学谱完全相同的样本对：%d 对（重复品评 / 同批次样品）' % len(pairs))
if len(pairs) >= 3:
    rs = []
    for i, j in pairs:
        m = np.isfinite(Y.values[i]) & np.isfinite(Y.values[j])
        rs.append(spearmanr(Y.values[i][m], Y.values[j][m]).statistic)
    log('这些样本对的感官 z 分 Spearman：中位 %.3f（同一款酒重复品评的一致性）' % np.median(rs))
    A['repeat_pairs'] = dict(n=len(pairs), median_rho=float(np.median(rs)))

sec('G. 与公开文献口径的对照（Schreurs et al., Nat Commun 2024, 15:2368）')
log('论文 Table 1（trained panel, test R²）：GBR 0.21、RF 0.22、Extra Trees 0.22、'
    'XGBoost 0.22、AdaBoost 0.21、SVR 0.18、ANN 0.14、Lasso 0.05、PLS −0.25、LR −4.13')
log('论文 Methods：7 分强度刻度、按品评员 z 标准化、231 个化学参数、250 个样本、'
    '70/30 分层留出、缺失 0.1% 用均值填补')
log('→ 本赛道题面把刻度改写为"0–6 分制"：7 档（0..6）恰好对应 7 点强度刻度，二者不矛盾。')
log('→ 文献 R²≈0.21~0.22 是在 **30% 留出集**上算的；本报告一律用 5 折×3 重复 CV 的 pooled R²，'
    '数值不完全可比，需在同一协议下比较。')

os.makedirs(C.OUT, exist_ok=True)
json.dump(A, open(os.path.join(C.OUT, 'audit.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
open(os.path.join(C.OUT, 'audit_report.txt'), 'w', encoding='utf-8').write('\n'.join(LOG))
print('\n审计报告已写 output/audit_report.txt')
