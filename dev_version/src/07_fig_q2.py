# -*- coding: utf-8 -*-
"""07 图件 · 问题二结果（图8–图10）

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
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.6,
    'legend.frameon': False, 'figure.dpi': 120, 'savefig.dpi': 400,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
CMAP_DIV = LinearSegmentedColormap.from_list(
    'byr', ['#2166AC', '#4393C3', '#92C5DE', '#F7F7CF', '#FDD07B', '#F4A582', '#B2182B'], N=256)
CMAP_SEQ = LinearSegmentedColormap.from_list(
    'byr_seq', ['#EAF2F8', '#92C5DE', '#F7E7A8', '#FDD07B', '#E07B54', '#B2182B'], N=256)
BLUE, RED, GREEN, PURPLE, GRAY = '#23405C', '#A8452C', '#4A6E5A', '#7A6A9B', '#8C939B'
INK, ACC = '#23405C', '#A8452C'
GCOL = {'Aroma': BLUE, 'Flavour': GREEN, 'Mouthfeel': PURPLE, 'Fault': RED,
        'Spice/野生': '#E08A1E', 'Overall': GRAY}


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
COMP, SDIM = D['comp'], D['sdim']
META = C.build_meta(COMP)
g = json.load(open(os.path.join(C.OUT, 'q2_graph.json'), encoding='utf-8'))
TOP = pd.read_csv(os.path.join(C.OUT, 'q2_top3.csv'))
PL = pd.read_csv(os.path.join(C.OUT, 'q2_plausibility.csv'))


def short(c):
    """化合物短名（用于图内标注）"""
    m = META.loc[c, 'chem_name'] if c in META.index else ''
    if isinstance(m, str) and m.strip():
        return m.strip()
    s = c.replace('X.', '').replace('X', '').replace('.', ' ')
    return s[:22]


# ============================================================ 图8 Top-3 气泡矩阵
print('图8 Top-3 关键化合物矩阵 ...')
dims = list(TOP['感官维度'])
fig, ax = plt.subplots(figsize=(7.2, 5.4))
for i, r in TOP.iterrows():
    for k in (1, 2, 3):
        f = r['Top%d入选频率' % k]
        s = r['Top%d综合分' % k]
        col = GCOL.get(r['分组'], GRAY)
        ax.scatter(k, len(dims) - 1 - i, s=18 + 130 * f, color=col,
                   alpha=0.35 + 0.6 * f, edgecolor='white', lw=0.4, zorder=3)
        ax.text(k + 0.30, len(dims) - 1 - i, short(r['Top%d' % k]), fontsize=4.6,
                va='center', ha='left', color='#222222')
ax.set_xlim(0.4, 3.55)
ax.set_ylim(-1, len(dims))
ax.set_yticks(range(len(dims)))
ax.set_yticklabels(dims[::-1], fontsize=4.9)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Top-1', 'Top-2', 'Top-3'], fontsize=7)
ax.set_xlabel('重要性排名（气泡面积 ∝ bootstrap 入选频率，颜色 = 感官分组）')
ax.grid(axis='y', ls=':', lw=0.3, color='#EBEEF1')
handles = [Line2D([], [], marker='o', ls='', mfc=GCOL[k], mec='none', ms=5, label=k)
           for k in GCOL]
ax.legend(handles=handles, loc='lower right', fontsize=6, ncol=3, handletextpad=0.15,
          columnspacing=0.6, bbox_to_anchor=(1.0, -0.075))
save(fig, 'fig08_top3_matrix.png')

# ============================================================ 图9 稳定性
print('图9 稳定性 ...')
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={'width_ratios': [1, 1.15]})
ax = axes[0]
freq = g['freq']
for k, mk in zip(['Top1', 'Top2', 'Top3'], ['o', 's', '^']):
    vals = []
    for s in SDIM:
        c = TOP.loc[TOP['感官维度'] == C.clean_sens_name(s), k].values
        if len(c) and isinstance(c[0], str):
            vals.append(freq.get(s, {}).get(c[0], 0.0))
    if vals:
        ax.hist(vals, bins=np.linspace(0, 1, 21), alpha=0.55, color=BLUE if k == 'Top1' else
                (GREEN if k == 'Top2' else PURPLE), label=k, edgecolor='white', lw=0.3)
ax.axvline(0.33, ls='--', lw=0.8, color=RED)
ax.text(0.34, ax.get_ylim()[1] * 0.82, '随机基线 1/3', fontsize=6, color=RED)
ax.set_xlabel('bootstrap 入选频率 $\\pi$'); ax.set_ylabel('感官维度个数')
ax.legend(fontsize=6.2)
panel(ax, 'a', 'Top-3 的 bootstrap 稳定性')

ax = axes[1]
jj = pd.DataFrame({'维度': [C.clean_sens_name(s) for s in SDIM],
                   '分组': [C.SENS_GROUP[s] for s in SDIM],
                   'J': [g['jaccard'][s] for s in SDIM]})
order = jj.groupby('分组')['J'].median().sort_values().index
data = [jj.loc[jj['分组'] == gg, 'J'].values for gg in order]
bp = ax.boxplot(data, widths=0.55, patch_artist=True,
                flierprops=dict(marker='o', ms=1.6, mfc=GRAY, mec='none', alpha=0.6))
for b, gg in zip(bp['boxes'], order):
    b.set_facecolor(GCOL.get(gg, GRAY)); b.set_alpha(0.45)
    b.set_edgecolor(GCOL.get(gg, GRAY)); b.set_linewidth(0.8)
for ln in bp['medians']:
    ln.set_color(RED); ln.set_linewidth(1.1)
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels(['%s(%d)' % (gg, (jj['分组'] == gg).sum()) for gg in order],
                   fontsize=6.0, rotation=20, ha='right')
ax.set_ylabel('跨方法 Jaccard 系数')
ax.set_ylim(0, 1)
panel(ax, 'b', '五路证据的方法间一致性')
fig.tight_layout(w_pad=1.4)
save(fig, 'fig09_stability.png')

# ============================================================ 图10 化学合理性
print('图10 化学合理性 ...')
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={'width_ratios': [1.2, 1]})

# 只保留出现次数 >=3 的化学类别，其余并入“其他”，避免图例碎片化
_cnt = PL.groupby('化学类别').size().sort_values(ascending=False)
KEEP = [f for f in _cnt.index if _cnt[f] >= 3]
PL2 = PL.copy()
PL2['族'] = np.where(PL2['化学类别'].isin(KEEP), PL2['化学类别'], '其他')
_order = ([f for f in _cnt.index if f in KEEP] +
          (['其他'] if (PL2['族'] == '其他').any() else []))
_PAL = ['#23405C', '#A8452C', '#4A6E5A', '#7A6A9B', '#B08A3E', '#5C7F9E', '#8C939B']
_famc = {f: _PAL[i % len(_PAL)] for i, f in enumerate(_order)}

ax = axes[0]
pl = PL2.dropna(subset=['OAV中位数']).copy()
pl = pl[pl['OAV中位数'] > 0]
for f in _order:
    sub = pl[pl['族'] == f]
    if len(sub) == 0:
        continue
    ax.scatter(sub['气味阈值'], sub['OAV中位数'], s=28, color=_famc[f],
               edgecolor='white', lw=0.45, label='%s (%d)' % (f, int((PL2['族'] == f).sum())),
               alpha=0.92)
ax.set_xscale('log'); ax.set_yscale('log')
ax.axhline(1, ls='--', lw=0.9, color=ACC)
ax.text(ax.get_xlim()[0] * 1.5, 1.25, 'OAV = 1（感知阈值）', fontsize=6.0, color=ACC)
ax.set_xlabel('气味阈值 $T_c$ / (mg·L$^{-1}$)')
ax.set_ylabel('OAV 中位数')
ax.legend(fontsize=5.6, ncol=2, handletextpad=0.15, columnspacing=0.5,
          loc='lower left', handlelength=0.9)
panel(ax, 'a', 'Top-3 的阈值—OAV 关系')

ax = axes[1]
fam_top = PL2.groupby('族').size().sort_values()
for i, (f, v) in enumerate(fam_top.items()):
    ax.plot([0, v], [i, i], color='#E4E8EC', lw=2.0, solid_capstyle='round', zorder=1)
    ax.scatter(v, i, s=40, color=_famc.get(f, GRAY), edgecolor='white', lw=0.5, zorder=3)
    ax.text(v + max(fam_top) * 0.035, i, str(v), va='center', fontsize=6.0, color='#333333')
ax.set_yticks(np.arange(len(fam_top)))
ax.set_yticklabels(fam_top.index, fontsize=6.4)
ax.grid(axis='x', ls=':', lw=0.4, color='#E4E8EC')
ax.set_xlabel('Top-3 中出现次数（共 %d 个名额）' % len(PL2))
ax.set_xlim(0, fam_top.max() * 1.20)
panel(ax, 'b', 'Top-3 的化学类别分布')
fig.tight_layout(w_pad=1.2)
save(fig, 'fig10_plausibility.png')
print('\n图 8–10 完成')
