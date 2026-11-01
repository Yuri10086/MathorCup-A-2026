# -*- coding: utf-8 -*-
"""06 图件 · 问题一结果（图4–图7）

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

IMG = os.path.join(C.ROOT, 'latex', 'images')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8,
    'legend.frameon': False, 'figure.dpi': 120, 'savefig.dpi': 400,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
CMAP_DIV = LinearSegmentedColormap.from_list(
    'byr', ['#2166AC', '#4393C3', '#92C5DE', '#F7F7CF', '#FDD07B', '#F4A582', '#B2182B'], N=256)
CMAP_SEQ = LinearSegmentedColormap.from_list(
    'byr_seq', ['#EAF2F8', '#92C5DE', '#F7E7A8', '#FDD07B', '#E07B54', '#B2182B'], N=256)
BLUE, RED, GREEN, PURPLE, GRAY = '#23405C', '#A8452C', '#4A6E5A', '#5C7F9E', '#8C939B'
INK, ACC = '#23405C', '#A8452C'
GCOL = {'Aroma': BLUE, 'Flavour': GREEN, 'Mouthfeel': '#7A6A9B', 'Fault': RED,
        'Spice/野生': '#B08A3E', 'Overall': GRAY}


def panel(ax, tag, title=None, dx=-0.14, dy=1.06):
    ax.text(dx, dy, '(%s)' % tag, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='top', ha='left')
    if title:
        ax.set_title(title, pad=4)


def save(fig, name):
    fig.savefig(os.path.join(IMG, name.replace('.png', '.pdf')))
    fig.savefig(os.path.join(IMG, name))
    plt.close(fig)
    print('  ✓ %s' % name)


D = C.load()
SDIM = D['sdim']
cv = json.load(open(os.path.join(C.OUT, 'q1_cv.json'), encoding='utf-8'))
res = {r['model']: r for r in cv['results']}
PDR = pd.read_csv(os.path.join(C.OUT, 'q1_metrics_perdim.csv'))
NAME_ZH = {'Ridge': 'Ridge', 'ElasticNet': 'ElasticNet', 'PLS': 'PLS',
           'RandomForest': '随机森林', 'GBR': 'GBR（文献口径）',
           'HistGBR': '直方图 GBR', 'RandomForest-B': '随机森林（超参B）',
           'XGBoost': 'XGBoost', 'LightGBM': 'LightGBM',
           '潜因子Ridge-K8': '潜因子 Ridge (K=8)',
           '潜因子Ridge-K15': '潜因子 Ridge (K=15)', 'MT-GNN': 'MT-GNN（本文）',
           'Stack': '堆叠集成（本文）',
           '潜因子+选60-K8': '潜因子Ridge+选60 (K=8)',
           '潜因子+选30-K8': '潜因子Ridge+选30 (K=8)',
           '潜因子+选30-K15': '潜因子Ridge+选30 (K=15)'}
LEVEL = {'Ridge': 'L0', 'ElasticNet': 'L0', 'PLS': 'L0', 'RandomForest': 'L1',
         'GBR': 'L1', 'HistGBR': 'L1', 'RandomForest-B': 'L1', 'XGBoost': 'L1',
         'LightGBM': 'L1', '潜因子Ridge-K8': 'L2', '潜因子Ridge-K15': 'L2',
         'MT-GNN': 'L3', 'Stack': 'L4',
         '潜因子+选60-K8': 'L2', '潜因子+选30-K8': 'L2', '潜因子+选30-K15': 'L2'}

# ============================================================ 图4 模型横向比较
print('图4 模型横向比较 ...')
order = sorted(res, key=lambda k: res[k]['mean_r2'])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={'width_ratios': [1.55, 1]})
ax = axes[0]
yy = np.arange(len(order))
for i, k in enumerate(order):
    r = res[k]
    c = RED if LEVEL[k] in ('L3', 'L4') else (GRAY if LEVEL[k] == 'L1' else BLUE)
    ax.plot([min(0, r['min_r2']), r['max_r2']], [i, i], color='#D8DCE1', lw=1.6,
            zorder=1, solid_capstyle='round')
    ax.scatter([r['mean_r2']], [i], s=42, color=c, zorder=3, edgecolor='white', lw=0.6)
    ax.scatter([r['median_r2']], [i], s=16, color='white', zorder=4, edgecolor=c, lw=0.9)
ax.axvline(0, color='#888888', lw=0.7, ls=':')
ax.axvline(0.21, color=RED, lw=0.9, ls='--')
ax.axvline(0.22, color='#C98A00', lw=0.9, ls='-.')
ax.text(0.214, 6.4, '文献 GBR\n$R^2$=0.21', fontsize=6, color=RED,
        va='bottom', linespacing=1.3)
ax.text(0.224, 4.6, '文献 XGB\n$R^2$=0.22', fontsize=6, color='#C98A00',
        va='bottom', linespacing=1.3)
ax.set_yticks(yy)
ax.set_yticklabels([NAME_ZH.get(k, k) for k in order], fontsize=6.6)
ax.set_xlabel('平均 $R^2$（实心=均值，空心=中位数，浅灰=极差）')
ax.set_xlim(-1.0, 0.42)
ax.set_ylim(-1.15, len(order) - 0.40)
ax.grid(axis='x', ls=':', lw=0.4, color='#E3E6EA')
ax.text(-0.97, -0.92, '蓝：线性与潜因子    灰：树模型    红：本文模型',
        fontsize=6.0, color='#555555', va='bottom', ha='left')
panel(ax, 'a', '十类模型的交叉验证表现')

ax = axes[1]
grp = PDR.groupby('group')['stack_r2'].agg(['mean', 'count']).sort_values('mean')
for i, (g, v) in enumerate(zip(grp.index, grp['mean'].values)):
    ax.plot([0, v], [i, i], color='#E4E8EC', lw=2.2, solid_capstyle='round', zorder=1)
    ax.scatter(v, i, s=42, color=INK, edgecolor='white', lw=0.5, zorder=3)
    ax.text(v + 0.010, i, '%.3f' % v, va='center', fontsize=6.2, color='#333333')
ax.set_yticks(np.arange(len(grp)))
ax.set_yticklabels(['%s(%d)' % (g, n) for g, n in zip(grp.index, grp['count'])],
                   fontsize=6.6)
ax.grid(axis='x', ls=':', lw=0.4, color='#E4E8EC')
ax.set_xlabel('平均 $R^2$（堆叠集成，按感官分组）')
ax.set_xlim(0, max(grp['mean']) * 1.32)
ax.axvline(0, color='#888888', lw=0.6, ls=':')
_m = float(PDR['stack_r2'].mean())
ax.axvline(_m, color=ACC, lw=0.9, ls='--')
ax.text(_m + 0.012, len(grp) - 0.55, '全部 50 维均值 %.3f' % _m, fontsize=5.9,
        color=ACC, va='top')
panel(ax, 'b', '分组成绩（本文最终模型）')
fig.tight_layout(w_pad=1.2)
save(fig, 'fig04_model_compare.png')

# ============================================================ 图5 逐维对比
print('图5 逐维对比 ...')
d = PDR.sort_values('stack_r2').reset_index(drop=True)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), gridspec_kw={'width_ratios': [2.5, 1]})
ax = axes[0]
yy = np.arange(len(d))
cols = [GCOL.get(g, GRAY) for g in d['group']]
ax.hlines(yy, d['best_single_r2'], d['stack_r2'], color='#D0D4D9', lw=1.2, zorder=1)
ax.scatter(d['best_single_r2'], yy, s=16, color='white', edgecolor=GRAY, lw=0.8,
           zorder=2, label='最优单模型')
ax.scatter(d['stack_r2'], yy, s=18, color=cols, zorder=3, label='堆叠集成')
ax.axvline(0, color='#888888', lw=0.7, ls=':')
ax.axvline(0.21, color=RED, lw=0.8, ls='--')
ax.text(0.215, 2, '文献 $R^2$=0.21', color=RED, fontsize=6, rotation=90, va='bottom')
ax.set_yticks(yy)
ax.set_yticklabels(d['name'], fontsize=4.9)
ax.set_xlabel('$R^2$（空心=最优单模型，实心=堆叠集成）')
ax.set_ylim(-1, len(d))
ax.grid(axis='x', ls=':', lw=0.35, color='#E6E9EC')
handles = [Line2D([], [], marker='o', ls='', mfc=GCOL[g], mec='none', ms=4, label=g)
           for g in GCOL]
ax.legend(handles=handles, loc='lower right', fontsize=5.8, ncol=2, handletextpad=0.15,
          columnspacing=0.7)
panel(ax, 'a', '50 个感官维度的 $R^2$', dx=-0.34)

ax = axes[1]
ax.scatter(d['dim_std'], d['stack_r2'], s=18, c=cols, edgecolor='white', lw=0.4)
z = np.polyfit(d['dim_std'], d['stack_r2'], 1)
xs = np.linspace(d['dim_std'].min(), d['dim_std'].max(), 50)
ax.plot(xs, np.polyval(z, xs), color=RED, lw=1.0, ls='--')
from scipy.stats import spearmanr
rho = spearmanr(d['dim_std'], d['stack_r2']).statistic
ax.set_xlabel('该维度标签标准差（区分度）')
ax.set_ylabel('$R^2$')
ax.text(0.04, 0.06, r'$\rho$=%.2f' % rho, transform=ax.transAxes, fontsize=7)
panel(ax, 'b', '难度与区分度', dx=-0.26)
fig.tight_layout(w_pad=1.0)
save(fig, 'fig05_perdim.png')

# ============================================================ 图6 预测散点
print('图6 预测散点 ...')
arr = np.load(os.path.join(C.OUT, 'q1_oof.npz'))
pred = arr['pred']
Y = D['Y'].values
pick = ['A_esters_all', 'F_hops_citrus', 'body', 'overall', 'diacetyl', 'A_malt_bread']
pick = [p for p in pick if p in SDIM]
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
for ax, s in zip(axes.ravel(), pick):
    j = SDIM.index(s)
    y, p = Y[:, j], pred[:, j]
    hb = ax.hexbin(y, p, gridsize=22, cmap=CMAP_SEQ, mincnt=1, linewidths=0.1)
    lim = [min(y.min(), p.min()), max(y.max(), p.max())]
    ax.plot(lim, lim, ls='--', lw=0.8, color=RED)
    ax.set_xlim(lim); ax.set_ylim(lim)
    r2 = PDR.loc[PDR['dim'] == s, 'stack_r2'].values[0]
    rho = PDR.loc[PDR['dim'] == s, 'spearman'].values[0]
    ax.set_title('%s  $R^2$=%.2f  ρ=%.2f' % (C.clean_sens_name(s), r2, rho), fontsize=7)
    ax.set_xlabel('实测（标准化）', fontsize=7)
    ax.set_ylabel('预测', fontsize=7)
    cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=5)
panel(axes[0, 0], 'a', dx=-0.34, dy=1.16)
fig.tight_layout(w_pad=1.0, h_pad=1.2)
save(fig, 'fig06_scatter.png')

# ============================================================ 图7 诊断与消融
print('图7 诊断与消融 ...')
abl = json.load(open(os.path.join(C.OUT, 'q1_ablation.json'), encoding='utf-8'))
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9))
ax = axes[0, 0]
an = list(abl['ablation'])
av = [abl['ablation'][k]['mean_r2'] for k in an]
lab = ['A0 基准', 'A1 +缺失指示', 'A2 +封底指示', 'A3 +图扩散', 'A4 全部']
yy7 = np.arange(len(an))
for i, v in enumerate(av):
    ax.plot([min(av) * 0.985, v], [i, i], color='#E4E8EC', lw=2.0, solid_capstyle='round',
            zorder=1)
    ax.scatter(v, i, s=38, color=INK, edgecolor='white', lw=0.5, zorder=3)
    ax.text(v + (max(av) - min(av)) * 0.04 + 0.001, i, '%.4f' % v, va='center',
            fontsize=6.0, color='#333333')
d0 = av[0]
ax.axvline(d0, color=ACC, lw=0.9, ls='--')
ax.text(d0, -0.52, '基准 A0', fontsize=5.8, color=ACC, ha='center', va='bottom')
ax.set_yticks(yy7)
ax.set_yticklabels(lab, fontsize=6.2)
ax.set_xlabel('平均 $R^2$')
ax.set_ylim(-0.95, len(an) - 0.45)
ax.set_xlim(min(av) * 0.985, max(av) + (max(av) - min(av)) * 0.55 + 0.004)
ax.grid(axis='x', ls=':', lw=0.4, color='#E4E8EC')
panel(ax, 'a', '消融实验')

ax = axes[0, 1]
sa = abl['sens_alpha']
kk = sorted(sa, key=float)
ax.plot([float(x) for x in kk], [sa[x] for x in kk], '-o', color=BLUE, ms=4, lw=1.1)
best_a = max(sa, key=lambda x: sa[x])
ax.axvline(float(best_a), color=RED, ls='--', lw=0.8)
ax.annotate('最优 α=%.2f' % float(best_a), (float(best_a), sa[best_a]),
            textcoords='offset points', xytext=(6, -12), fontsize=6.2, color=RED)
ax.set_xlabel('图扩散强度 α'); ax.set_ylabel('平均 $R^2$')
panel(ax, 'b', '扩散强度敏感性')

ax = axes[1, 0]
st = abl['sens_topk']
kk = sorted(st, key=int)
v7 = [st[x] for x in kk]
for i, v in enumerate(v7):
    ax.plot([i, i], [min(v7) * 0.99, v], color='#E4E8EC', lw=2.0, solid_capstyle='round',
            zorder=1)
    ax.scatter(i, v, s=38, color=INK, edgecolor='white', lw=0.5, zorder=3)
    ax.text(i, v + (max(v7) - min(v7)) * 0.10 + 0.0005, '%.4f' % v, ha='center',
            fontsize=6.0, color='#333333')
ax.set_xticks(np.arange(len(kk)))
ax.set_xticklabels(['k=%s' % x for x in kk], fontsize=6.6)
ax.set_ylim(min(v7) * 0.985, max(v7) + (max(v7) - min(v7)) * 0.55)
ax.set_ylabel('平均 $R^2$')
ax.set_xlim(-0.6, len(kk) - 0.4)
panel(ax, 'c', '邻居数敏感性')

ax = axes[1, 1]
cov = cv['interval']
resid = (pred - Y)
ax.hist(resid.ravel(), bins=60, color=BLUE, alpha=0.85, edgecolor='white', lw=0.2)
ax.axvline(cov['width'] / 2, color=RED, ls='--', lw=0.8)
ax.axvline(-cov['width'] / 2, color=RED, ls='--', lw=0.8)
ax.set_xlabel('残差（标准化尺度）'); ax.set_ylabel('样本数')
ax.text(0.97, 0.95, '90%%名义区间\n实测覆盖 %.1f%%\n平均宽度 %.2f' % (
    100 * cov['coverage'], cov['width']), transform=ax.transAxes, ha='right', va='top',
    fontsize=6.2, linespacing=1.5)
panel(ax, 'd', '残差与区间标定')
fig.tight_layout(w_pad=1.2, h_pad=1.3)
save(fig, 'fig07_diagnostics.png')
print('\n图 4–7 完成')
