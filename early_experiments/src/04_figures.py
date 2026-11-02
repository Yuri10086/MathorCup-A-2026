# -*- coding: utf-8 -*-
"""04_figures.py —— 问题1 图件（出版级，中文标注）

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek）。

图 1  fig01_audit.png        数据审计三联图：缺失热图 / 检出限地板 / 量纲判定
图 2  fig02_model_cv.png     多模型 R² 对比（箱线 + 均值）+ 逐维热图
图 3  fig03_perdim.png       逐维 R² 排序（按感官分组着色）+ R²-标签离散度关系
图 4  fig04_scatter.png      预测 vs 真值散点（按 R² 分档）
图 5  fig05_calibration.png  0–6 档位混淆矩阵 + 档位分布对比
图 6  fig06_uncertainty.png  预测区间覆盖率与宽度
图 7  fig07_ablation.png     消融实验瀑布图
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q1lib as C

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 130, 'savefig.dpi': 330, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9, 'axes.linewidth': 0.8,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
})

D = C.load()
COMP, SDIM = D['comp'], D['sdim']
Y = D['Y'].values.astype(float)
R = np.load(os.path.join(C.OUT, 'q1_result.npz'))
stack_pred = R['stack_pred']
GRADE, Y06 = R['grade'], R['grade_truth']
SPREAD = R['spread']
CVJ = json.load(open(os.path.join(C.OUT, 'q1_cv.json'), encoding='utf-8'))
GRP = C.GROUP
GC = {'Aroma': '#c26b4a', 'Flavour': '#4a7ab0', 'Mouthfeel': '#5c9e5c',
      'Fault': '#9b5ca8', 'Spice/野生': '#c9a227', 'Overall': '#777777'}


def panel_label(ax, s):
    ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight='bold',
            va='top', ha='left')


# ================================================================ 图 1 审计
fig = plt.figure(figsize=(11.5, 6.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.42, wspace=0.30)

ax = fig.add_subplot(gs[0, :])
miss = D['X'].isna().astype(int).values.T
keep = np.where(miss.sum(1) > 0)[0]
ax.imshow(miss[keep], aspect='auto', cmap=ListedColormap(['#f2f2f2', '#b5342b']),
          interpolation='nearest')
ax.set_yticks(range(len(keep)))
ax.set_yticklabels([COMP[i] for i in keep], fontsize=7)
ax.set_xlabel('啤酒样本序号（按 beer 排序）')
ax.set_title('A  缺失结构：仅 9 列为空，共 73 格（0.126%），无任何样本整行缺失', fontsize=10,
             loc='left')
ax.grid(False)

ax = fig.add_subplot(gs[1, 0])
V = D['X'][COMP].values.astype(float)
eq = np.isclose(V, C.FLOOR, atol=1e-9)
frac = eq.sum(0) / np.maximum(np.isfinite(V).sum(0), 1)
ax.hist(frac * 100, bins=np.arange(0, 105, 5), color='#b5342b', alpha=0.75, edgecolor='w')
ax.axvline(5, ls='--', lw=1, color='k')
ax.text(6, ax.get_ylim()[1] * 0.8, '≥5%%: %d 列' % int((frac >= 0.05).sum()), fontsize=8)
ax.set_xlabel('该列压在地板 −1.8178 的样本比例 (%)')
ax.set_ylabel('列数')
ax.set_title('B  左删失：%d 格（11.80%%）为检出限地板' % int(eq.sum()), fontsize=10, loc='left')

ax = fig.add_subplot(gs[1, 1])
sc, det = C.colscale()
nraw = sum(1 for c in COMP if sc[c] == 'raw')
ncal = sum(1 for c in COMP if det[c]['calibrated'])
ax.bar(['原始浓度\n(逐位等于校准值)', 'log10 变换\n(10^v = 校准值)', '无校准信息\n(默认 log10)'],
       [nraw, ncal - nraw, len(COMP) - ncal],
       color=['#4a7ab0', '#5c9e5c', '#bbbbbb'], edgecolor='w')
for i, v in enumerate([nraw, ncal - nraw, len(COMP) - ncal]):
    ax.text(i, v + 3, str(v), ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('列数')
ax.set_title('C  量纲判定（8.xlsx 校准表逐位比对）', fontsize=10, loc='left')

ax = fig.add_subplot(gs[1, 2])
_v = V[np.isfinite(V)].ravel()
_h, _e = np.histogram(_v, bins=120, range=(-5, 8))
ax.bar((_e[:-1] + _e[1:]) / 2, _h, width=(_e[1] - _e[0]) * 0.9,
       color='#4a7ab0', edgecolor='#2c4d70', lw=0.2)
ax.axvline(C.FLOOR, color='#b5342b', lw=1.2, ls='--')
ax.axvline(0.0, color='#5c9e5c', lw=1.0, ls=':')
ax.set_yscale('log')
ax.set_xlabel('1.csv 数值（统一到对数尺度后）')
ax.set_ylabel('频数（log）')
ax.set_xlim(-5, 8)
ax.text(C.FLOOR - 0.15, ax.get_ylim()[1] * 0.85, '检出限地板\n−1.8178', color='#b5342b',
        fontsize=7.5, va='top', ha='right')
ax.text(0.35, ax.get_ylim()[1] * 0.85, '浓度=1\n(v=0)', color='#4a7b4a', fontsize=7,
        va='top', ha='left')
ax.set_title('D  数值分布：右移长尾 + 左端封底尖峰', fontsize=10, loc='left')
fig.suptitle('图 1  数据审计：缺失、左删失（检出限）与量纲判定', fontsize=11, y=0.99)
fig.savefig(os.path.join(C.FIG, 'fig01_audit.png'))
plt.close(fig)

# ================================================================ 图 2 多模型
OOFM = np.load(os.path.join(C.OUT, 'q1_result.npz'))
names = list(CVJ['stack_weights'][SDIM[0]].keys())
oof_all = {}
PRED_DIR = os.path.join(C.OUT, 'q1_preds')
for n in names:
    oof_all[n] = np.load(os.path.join(PRED_DIR, '%s.npz' % n))['oof']

fig = plt.figure(figsize=(11.5, 5.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.26)

ax = fig.add_subplot(gs[0, 0])
rows = []
for n in names:
    pred = np.nanmean(oof_all[n], axis=0)
    rows.append(C.per_dim_r2(Y, pred))
rows.append(C.per_dim_r2(Y, stack_pred))
labs = names + ['Stack']
order = np.argsort([-np.nanmean(r) for r in rows])
bp = ax.boxplot([rows[i] for i in order], vert=False, widths=0.62, patch_artist=True,
                showfliers=False, medianprops=dict(color='k', lw=1.1))
cols = plt.cm.RdYlBu_r(np.linspace(0.18, 0.9, len(order)))
for p, c in zip(bp['boxes'], cols):
    p.set_facecolor(c)
    p.set_alpha(0.85)
    p.set_edgecolor('k')
    p.set_linewidth(0.6)
ax.set_yticklabels([labs[i] + (' (mean %.3f)' % np.nanmean(rows[i])) for i in order],
                   fontsize=8)
ax.axvline(0, color='k', lw=0.8, ls=':')
ax.set_xlabel('逐维 R²（50 个感官维度）')
ax.set_title('A  7 个模型 + 堆叠的逐维 R² 分布', fontsize=10, loc='left')

ax = fig.add_subplot(gs[0, 1])
M = np.array([rows[i] for i in order])
im = ax.imshow(M, aspect='auto', cmap='RdYlBu_r', vmin=-0.25, vmax=0.8)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([labs[i] for i in order], fontsize=7.5)
ax.set_xticks(range(0, 50, 5))
ax.set_xticklabels([C.SENS_ZH.get(SDIM[i], SDIM[i])[:6] for i in range(0, 50, 5)],
                   fontsize=6.5, rotation=90)
ax.set_xlabel('感官维度（按真值顺序）')
ax.set_title('B  逐维 R² 热图（行=模型，列=感官维度）', fontsize=10, loc='left')
ax.grid(False)
plt.colorbar(im, ax=ax, fraction=0.024, pad=0.01, label='R²')
fig.suptitle('图 2  多模型横向比较（同一 CV 协议：5 折 × 3 重复，按风格分层）', fontsize=11,
             y=1.02)
fig.savefig(os.path.join(C.FIG, 'fig02_model_cv.png'))
plt.close(fig)

# ================================================================ 图 3 逐维
PD = pd.read_csv(os.path.join(C.OUT, 'q1_perdim.csv'))
PD = PD.sort_values('stack_r2', ascending=False).reset_index(drop=True)

fig = plt.figure(figsize=(11.5, 5.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.24)

ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(PD))
ax.bar(x, PD['stack_r2'], color=[GC[g] for g in PD['group']], edgecolor='w', lw=0.4)
ax.plot(x, PD['best_single_r2'], 'k.', ms=3, label='最优单模型')
ax.axhline(np.nanmean(PD['stack_r2']), color='k', ls='--', lw=0.9,
           label='均值 %.3f' % np.nanmean(PD['stack_r2']))
ax.set_xticks(x[::2])
ax.set_xticklabels([C.SENS_ZH.get(d, d)[:8] for d in PD['dim'][::2]], rotation=90,
                   fontsize=6.5)
ax.set_ylabel('R²')
ax.set_title('A  逐维 R²（按分组着色，降序）', fontsize=10, loc='left')
hs = [plt.Line2D([], [], marker='s', ls='', color=GC[g], label=g) for g in GC]
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1 + hs, l1 + list(GC), fontsize=6.5, ncol=2, loc='upper right', framealpha=0.9)

ax = fig.add_subplot(gs[0, 1])
for g in GC:
    m = PD['group'] == g
    ax.scatter(PD.loc[m, 'dim_std'], PD.loc[m, 'stack_r2'], s=26, color=GC[g], label=g,
               edgecolor='k', lw=0.3, alpha=0.85)
z = np.polyfit(PD['dim_std'], PD['stack_r2'], 1)
xx = np.linspace(PD['dim_std'].min(), PD['dim_std'].max(), 20)
ax.plot(xx, np.polyval(z, xx), 'k--', lw=1)
rho = spearmanr(PD['dim_std'], PD['stack_r2']).statistic
ax.set_xlabel('该维标签的标准差（品评员分歧程度）')
ax.set_ylabel('R²')
ax.set_title('B  预测难度与品评分歧：ρ=%.3f' % rho, fontsize=10, loc='left')
ax.legend(fontsize=6.5, loc='upper left')
fig.suptitle('图 3  逐维表现：哪些感官维度可预测、哪些不可预测', fontsize=11, y=1.02)
fig.savefig(os.path.join(C.FIG, 'fig03_perdim.png'))
plt.close(fig)

# ================================================================ 图 4 散点
bins = [(-9, 0), (0, 0.2), (0.2, 0.4), (0.4, 1.0)]
labs = ['R² < 0', '0 ≤ R² < 0.2', '0.2 ≤ R² < 0.4', 'R² ≥ 0.4']
fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1), sharey=True)
for ax, (lo_, hi_), lb in zip(axes, bins, labs):
    m = (PD['stack_r2'] >= lo_) & (PD['stack_r2'] < hi_)
    idx = [SDIM.index(d) for d in PD.loc[m, 'dim']]
    for j in idx:
        ax.scatter(Y[:, j], stack_pred[:, j], s=5, alpha=0.45,
                   color=GC[PD.loc[PD.dim == SDIM[j], 'group'].iloc[0]], lw=0)
    ax.plot([-2, 3], [-2, 3], 'k--', lw=0.8)
    ax.set_xlim(-2.2, 2.6)
    ax.set_ylim(-2.2, 2.6)
    ax.set_xlabel('真值 z 分')
    ax.set_title('%s\n(%d 维)' % (lb, len(idx)), fontsize=9)
    ax.set_aspect('equal', adjustable='box')
axes[0].set_ylabel('预测 z 分')
fig.suptitle('图 4  预测 vs 真值（按逐维 R² 分成四组）', fontsize=11, y=1.03)
fig.savefig(os.path.join(C.FIG, 'fig04_scatter.png'))
plt.close(fig)

# ================================================================ 图 5 档位
fig = plt.figure(figsize=(11.5, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.30)

ax = fig.add_subplot(gs[0, 0])
CM = np.zeros((7, 7))
for a in range(7):
    for b in range(7):
        CM[a, b] = np.mean((GRADE == a) & (Y06 == b))
CMn = CM / CM.sum(1, keepdims=True)
im = ax.imshow(CMn, cmap='Blues', vmin=0, vmax=CMn.max())
ax.set_xticks(range(7))
ax.set_yticks(range(7))
ax.set_xlabel('真值档位（同切点）')
ax.set_ylabel('预测档位')
ax.set_title('A  档位混淆矩阵（行归一）', fontsize=10, loc='left')
for a in range(7):
    for b in range(7):
        if CMn[a, b] > 0.02:
            ax.text(b, a, '%.2f' % CMn[a, b], ha='center', va='center', fontsize=6.2,
                    color='w' if CMn[a, b] > 0.45 * CMn.max() else 'k')
ax.grid(False)

ax = fig.add_subplot(gs[0, 1])
dp = np.bincount(GRADE.ravel(), minlength=7) / GRADE.size
dt = np.bincount(Y06.ravel(), minlength=7) / Y06.size
w = 0.4
ax.bar(np.arange(7) - w / 2, dt, w, label='真值档位', color='#4a7ab0')
ax.bar(np.arange(7) + w / 2, dp, w, label='预测档位', color='#c26b4a')
ax.set_xticks(range(7))
ax.set_xlabel('0–6 档位')
ax.set_ylabel('占比')
ax.set_title('B  档位边缘分布', fontsize=10, loc='left')
ax.legend(fontsize=7.5)

ax = fig.add_subplot(gs[0, 2])
per = np.array([np.mean(GRADE[:, j] == Y06[:, j]) for j in range(50)])
ax.hist(per, bins=np.linspace(0, 0.6, 25), color='#5c9e5c', alpha=0.85, edgecolor='w')
ax.axvline(1 / 7, color='k', ls='--', lw=1, label='随机基线 1/7')
ax.axvline(np.mean(per), color='#b5342b', lw=1.2, label='均值 %.3f' % np.mean(per))
ax.set_xlabel('逐维档位完全命中率')
ax.set_ylabel('维度数')
ax.set_title('C  50 维各自的命中率', fontsize=10, loc='left')
ax.legend(fontsize=7)
fig.suptitle('图 5  0–6 分级交付的档位一致性（完全命中 %.3f，±1 档 %.3f）'
             % (CVJ['grade']['exact_acc'], CVJ['grade']['within1']), fontsize=11, y=1.03)
fig.savefig(os.path.join(C.FIG, 'fig05_calibration.png'))
plt.close(fig)

# ================================================================ 图 6 不确定性
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
ax = axes[0]
inside = ((Y >= stack_pred - 1.645 * SPREAD) & (Y <= stack_pred + 1.645 * SPREAD))
cov_dim = inside.mean(0)
ax.bar(range(50), cov_dim * 100, color=[
    GC[GRP[SDIM[j]]] for j in range(50)], edgecolor='w', lw=0.3)
ax.axhline(90, color='k', ls='--', lw=1)
ax.set_xlabel('感官维度')
ax.set_ylabel('实测覆盖率 (%)')
ax.set_title('A  90%% 名义区间实测覆盖率（总体 %.1f%%）' % (100 * inside.mean()), fontsize=9,
             loc='left')

ax = axes[1]
ax.scatter(SPREAD.ravel(), np.abs(Y - stack_pred).ravel(), s=3, alpha=0.25,
           color='#4a7ab0', lw=0)
lim = np.percentile(np.abs(Y - stack_pred), 99.5)
ax.plot([0, lim], [0, lim], 'k--', lw=1)
ax.set_xlim(0, np.percentile(SPREAD, 99.5))
ax.set_ylim(0, lim)
ax.set_xlabel('重复间预测标准差（当作区间半宽来源）')
ax.set_ylabel('|预测误差|')
r_ = spearmanr(SPREAD.ravel(), np.abs(Y - stack_pred).ravel()).statistic
ax.set_title('B  离散度 vs 真实误差：ρ=%.3f' % r_, fontsize=9, loc='left')

ax = axes[2]
ax.hist(SPREAD.ravel(), bins=60, color='#c9a227', alpha=0.85)
ax.set_xlabel('重复间预测标准差')
ax.set_ylabel('频数')
ax.set_title('C  预测离散度分布（中位 %.3f）' % np.median(SPREAD), fontsize=9, loc='left')
fig.suptitle('图 6  不确定性：为什么 90%% 区间只有 %.1f%% 覆盖率' % (100 * inside.mean()),
             fontsize=11, y=1.04)
fig.savefig(os.path.join(C.FIG, 'fig06_uncertainty.png'))
plt.close(fig)

# ================================================================ 图 7 消融
_fp7 = os.path.join(C.OUT, 'q1_ablation.csv')
if os.path.exists(_fp7):
    AB = pd.read_csv(_fp7)
    fig = plt.figure(figsize=(11.5, 5.6))
    gs = fig.add_gridspec(1, 3, wspace=0.95, width_ratios=[1.0, 0.85, 1.0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    ax = axes[0]
    sel = AB[AB.model == 'RidgeCV'].copy()
    base = float(sel[sel.case.str.startswith('A')]['mean_r2'].iloc[0])
    sel['delta'] = sel['mean_r2'] - base
    sel = sel.sort_values('delta')
    cols = ['#5c9e5c' if c.startswith('A') else ('#b5342b' if c.startswith(('H', 'I')) else
            '#4a7ab0') for c in sel.case]
    ax.barh(range(len(sel)), sel['delta'], color=cols, edgecolor='w')
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([('%s  (%.4f)' % (c[:20], v)) for c, v in zip(sel.case, sel['mean_r2'])],
                       fontsize=6.5)
    ax.axvline(0, color='k', lw=0.9)
    ax.axvspan(-0.006, 0.006, color='#dddddd', alpha=0.6, zorder=0)
    ax.text(-0.006, len(sel) - 0.35, '±0.006 噪声带', fontsize=6.2, ha='right',
            color='#666666', va='center')
    ax.set_xlim(-0.152, 0.040)
    for i, (c, v) in enumerate(zip(sel.case, sel['delta'])):
        fr = ax.transLimits.transform((v, i))[0]
        ax.text(fr + (0.02 if v >= 0 else -0.02), i / max(len(sel) - 1, 1),
                '%+.4f' % v, fontsize=6.2, va='center',
                ha='left' if v >= 0 else 'right',
                transform=ax.transAxes, clip_on=False)
    ax.set_xlabel('ΔR²（相对"完整方案 A"）')
    ax.set_title('A  RidgeCV：各步骤的边际增益（带内=噪声）', fontsize=9, loc='left')

    ax = axes[1]
    sel2 = AB[AB.model == 'Ridge-fixed30'].sort_values('mean_r2')
    cols2 = ['#5c9e5c' if c.startswith('A') else ('#b5342b' if c.startswith(('H', 'I')) else
             '#4a7ab0') for c in sel2.case]
    ax.barh(range(len(sel2)), sel2['mean_r2'], color=cols2, edgecolor='w')
    ax.set_yticks(range(len(sel2)))
    ax.set_yticklabels([('%s (%.3f)' % (c[:20], v)) for c, v in zip(sel2.case, sel2['mean_r2'])],
                       fontsize=6.5)
    ax.axvline(0, color='k', lw=0.9)
    ax.set_xlabel('mean R²')
    ax.set_title('B  Ridge（固定 α=30）：同一批方案的全塌', fontsize=9, loc='left')

    ax = axes[2]
    if (AB.model == 'HistGBR').any():
        sel3 = AB[AB.model == 'HistGBR'].copy()
        b3 = float(sel3[sel3.case.str.startswith('A')]['mean_r2'].iloc[0])
        sel3['delta'] = sel3['mean_r2'] - b3
        sel3 = sel3.sort_values('delta')
        ax.barh(range(len(sel3)), sel3['delta'], color='#4a7ab0', edgecolor='w')
        ax.set_yticks(range(len(sel3)))
        ax.set_yticklabels([('%s (%.4f)' % (c[:20], v))
                            for c, v in zip(sel3.case, sel3['mean_r2'])], fontsize=6.5)
        ax.axvline(0, color='k', lw=0.9)
        ax.axvspan(-0.0005, 0.0005, color='#dddddd', alpha=0.6, zorder=0)
        for i, v in enumerate(sel3['delta']):
            ax.text(v + (0.0002 if v >= 0 else -0.0002), i, '%+.4f' % v,
                    fontsize=6.2, va='center', ha='left' if v >= 0 else 'right')
        ax.set_xlabel('ΔR²（相对"完整方案 A"）')
        ax.set_title('C  HistGBR：几乎完全不变（树对单调变换免疫）', fontsize=9, loc='left')
    fig.suptitle('图 7  消融实验：特征工程几乎不涨分，基线修正才是关键', fontsize=11, y=1.02)
    fig.savefig(os.path.join(C.FIG, 'fig07_ablation.png'))
    plt.close(fig)

print('图件已输出：', sorted(os.listdir(C.FIG)))












