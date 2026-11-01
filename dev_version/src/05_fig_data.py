# -*- coding: utf-8 -*-
"""05 图件 · 数据审计与探索性分析（图1–图3）

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。

配色契约（用户指定）：
  * 任何色条一律“蓝-黄-红”
  * 柱状/条形图统一科研配色，不使用花哨多色
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
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

IMG = os.path.join(C.ROOT, 'latex', 'images')
os.makedirs(IMG, exist_ok=True)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'legend.frameon': False, 'figure.dpi': 120, 'savefig.dpi': 400,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

# ---- 蓝-黄-红（用户指定）
CMAP_DIV = LinearSegmentedColormap.from_list(
    'byr', ['#2166AC', '#4393C3', '#92C5DE', '#F7F7CF', '#FDD07B', '#F4A582', '#B2182B'], N=256)
CMAP_SEQ = LinearSegmentedColormap.from_list(
    'byr_seq', ['#EAF2F8', '#92C5DE', '#F7E7A8', '#FDD07B', '#E07B54', '#B2182B'], N=256)
BLUE, RED, GREEN, PURPLE, GRAY = '#1F6FD0', '#C0392B', '#00876C', '#6C4AA0', '#7A8794'
CLEAN = [BLUE, RED, GREEN, PURPLE, '#2E9E4F', '#C2185B']


def panel(ax, tag, title=None, dx=-0.13, dy=1.06):
    ax.text(dx, dy, '(%s)' % tag, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='top', ha='left')
    if title:
        ax.set_title(title, pad=4)


def save(fig, name, caption=''):
    p = os.path.join(IMG, name)
    fig.savefig(p.replace('.png', '.pdf'))
    fig.savefig(p)
    plt.close(fig)
    print('  ✓ %s' % name)


D = C.load()
COMP, SDIM = D['comp'], D['sdim']
META = C.build_meta(COMP)
X = D['X']
Y = D['Y']
STYLE = D['style']

# ============================================================ 图1 数据全景
print('图1 数据审计全景 ...')
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))

# (a) 缺失与左删失
ax = axes[0, 0]
miss = X.isna().mean(axis=0)
floor = pd.Series({c: float((np.abs(X[c].dropna() - X[c].min()) < 1e-9).mean())
                   for c in COMP})
lo = np.where(floor >= 0.05)[0]
ax.hist(floor.values * 100, bins=np.arange(0, 105, 5), color=BLUE, alpha=0.85,
        edgecolor='white', linewidth=0.4)
ax.axvline(5, ls='--', lw=0.9, color=RED)
ax.set_ylim(0, ax.get_ylim()[1] * 1.42)
ax.text(0.97, 0.97, '封底化合物 %d 个\n封底单元格 12.6%%\n缺失 0.126%%' % len(lo),
        transform=ax.transAxes, ha='right', va='top', fontsize=6.4, linespacing=1.6)
ax.text(6, ax.get_ylim()[1] * 0.52, '检出限\n封底阈值 5%', color=RED, fontsize=6.2,
        linespacing=1.4)
ax.set_xlabel('取到最小值的样本占比 / %')
ax.set_ylabel('化合物个数')
panel(ax, 'a', '左删失与缺失结构')

# (b) 变换底数核实（仅对数变换列；33 个原始浓度列单独标注）
ax = axes[0, 1]
cal = pd.read_excel(os.path.join(C.DATA, '8.xlsx'), sheet_name='Calibrated chemical dataset')
cal.columns = [str(c).strip() for c in cal.columns]
cal = cal.sort_values('beer').reset_index(drop=True)
SCALE, _unk = C.column_scale(COMP)
v_log, c_log, v_raw, c_raw = [], [], [], []
for c in COMP:
    if c in cal.columns:
        v = X[c].values.astype(float)
        con = cal[c].values.astype(float)
        m = np.isfinite(v) & (con > 0)
        if m.sum() < 5:
            continue
        if SCALE.get(c) == 'raw':
            v_raw.append(v[m]); c_raw.append(con[m])
        else:
            v_log.append(v[m]); c_log.append(np.log10(con[m]))
v_log = np.concatenate(v_log); c_log = np.concatenate(c_log)
ax.scatter(v_log, c_log, s=1.4, color=BLUE, alpha=0.35, linewidths=0,
           label='对数变换列（%d 列）' % len([1 for c in COMP
                                            if c in cal.columns and SCALE.get(c) != 'raw']))
lim = np.array([-3.2, 2.2])
ax.plot(lim, lim, ls='--', lw=1.0, color=BLUE)
ax.plot(lim, lim * 0.4343, ls=':', lw=1.0, color=GRAY)
ax.text(1.05, 1.35, r'$y=x$（底数 10）', color=BLUE, fontsize=6.3, ha='left', va='top')
ax.text(1.05, 0.05, r'$y=0.4343x$（底数 e）', color=GRAY, fontsize=6.3, ha='left', va='top')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r'1.csv 数值 $v$（对数变换列）')
ax.set_ylabel(r'$\log_{10}$(校准浓度)')
n_raw = len([c for c in COMP if SCALE.get(c) == 'raw'])
ax.text(0.03, 0.99, '斜率 %.4f，$R^2$=%.4f\n另有 %d 列实为原始浓度\n（附件说明未提及）' % (
    1.0, 1.0, n_raw), transform=ax.transAxes, ha='left', va='top',
    fontsize=6.3, linespacing=1.55, color='#222222')
panel(ax, 'b', '对数变换底数判定')

# (c) 化合物丰度分布（按化学类别，用列中位数，规避原始量纲列）
ax = axes[0, 2]
fam = pd.Series([C.family_of(c) for c in COMP])
top_fam = fam.value_counts().head(6).index.tolist()
data, labels = [], []
med = X.median(axis=0)
for f in top_fam:
    cols = [c for c in COMP if C.family_of(c) == f]
    data.append(med[cols].values)
    labels.append('%s(%d)' % (f, len(cols)))
vp = ax.violinplot(data, showextrema=False, widths=0.8)
for b in vp['bodies']:
    b.set_facecolor(BLUE); b.set_alpha(0.5); b.set_edgecolor(BLUE); b.set_linewidth(0.6)
bp = ax.boxplot(data, widths=0.13, patch_artist=True, showfliers=False)
for b in bp['boxes']:
    b.set_facecolor('white'); b.set_edgecolor(GRAY); b.set_linewidth(0.6)
for ln in bp['medians']:
    ln.set_color(RED); ln.set_linewidth(1.0)
ax.set_xticks(range(1, len(labels) + 1))
ax.set_xticklabels(labels, rotation=28, ha='right', fontsize=6.2)
ax.set_ylabel('化合物中位丰度（1.csv 原值）')
ax.axhline(0, ls=':', lw=0.7, color=GRAY)
panel(ax, 'c', '各类化合物丰度分布')

# (d) 风格样本量
ax = axes[1, 0]
vc = pd.Series(STYLE).value_counts()
ypos = np.arange(len(vc))
ax.barh(ypos, vc.values, color=BLUE, height=0.7, edgecolor='white', linewidth=0.3)
ax.set_yticks(ypos)
ax.set_yticklabels(['%s' % n for n in vc.index], fontsize=5.3)
ax.invert_yaxis()
ax.set_xlabel('样本数（共 %d 款，22 种风格）' % len(STYLE))
for i, v in enumerate(vc.values):
    ax.text(v + 0.5, i, str(v), va='center', ha='left', fontsize=5.2, color='#444444')
ax.axvline(8, ls='--', lw=0.8, color=RED)
ax.set_xlim(0, vc.max() * 1.14)
ax.text(vc.max() * 0.955, 20.4, 'n≥8 参与生成', fontsize=5.6, color=RED, va='center',
        ha='right')
panel(ax, 'd', '22 种啤酒风格样本量', dx=-0.30)

# (e) 感官维度分布
ax = axes[1, 1]
grp = np.array([C.SENS_GROUP[c] for c in SDIM])
gorder = ['Aroma', 'Flavour', 'Mouthfeel', 'Fault', 'Spice/野生', 'Overall']
data2 = [Y.values[:, grp == g].ravel() for g in gorder if (grp == g).sum()]
lab2 = ['%s(%d)' % (g, (grp == g).sum()) for g in gorder if (grp == g).sum()]
bp = ax.boxplot(data2, widths=0.55, patch_artist=True, showfliers=True,
                flierprops=dict(marker='o', ms=1.1, mfc=GRAY, mec='none', alpha=0.5))
for b in bp['boxes']:
    b.set_facecolor('#CFE0F0'); b.set_edgecolor(BLUE); b.set_linewidth(0.7)
for ln in bp['medians']:
    ln.set_color(RED); ln.set_linewidth(1.1)
ax.set_xticks(range(1, len(lab2) + 1))
ax.set_xticklabels(lab2, rotation=22, ha='right', fontsize=6.2)
ax.set_ylabel('感官评分（标准化）')
ax.axhline(0, ls=':', lw=0.7, color=GRAY)
panel(ax, 'e', '感官目标分布')

# (f) 感官维度间相关（截取块状结构）
ax = axes[1, 2]
c5 = Y.corr(method='spearman').values
im = ax.imshow(c5, cmap=CMAP_DIV, norm=TwoSlopeNorm(0, -1, 1), interpolation='nearest')
r = np.corrcoef(c5[np.triu_indices_from(c5, 1)], np.zeros(len(c5[np.triu_indices_from(c5, 1)])))
ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cb.set_label('Spearman ρ', fontsize=6.5); cb.ax.tick_params(labelsize=6)
ax.set_title('感官维度相关性', pad=4)
ax.text(0.5, 0.985, '平均 |ρ| = %.2f' % np.nanmean(np.abs(c5[np.triu_indices_from(c5, 1)])),
        transform=ax.transAxes, ha='center', va='top', fontsize=6.3, color='white',
        bbox=dict(fc='#00000055', ec='none', pad=1.2))
panel(ax, 'f', dx=-0.06)

fig.tight_layout(w_pad=1.6, h_pad=1.4)
save(fig, 'fig01_data_overview.png',
     '数据审计六联图：左删失结构、对数底数判定、化合物丰度、风格分布、感官目标与相关结构')

# ============================================================ 图2 感官相关聚类热图
print('图2 感官相关聚类热图 ...')
cs = Y.corr(method='spearman')
Dist = 1 - cs.values
np.fill_diagonal(Dist, 0)
Zl = linkage(squareform(Dist, checks=False), method='average')
order = dendrogram(Zl, no_plot=True)['leaves']
names = [C.clean_sens_name(SDIM[i]) for i in order]
gcol = {'Aroma': BLUE, 'Flavour': GREEN, 'Mouthfeel': PURPLE, 'Fault': RED,
        'Spice/野生': '#E08A1E', 'Overall': GRAY}

fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(2, 3, width_ratios=[0.55, 6.4, 0.26], height_ratios=[0.75, 6.4],
                      wspace=0.035, hspace=0.035)
axd = fig.add_subplot(gs[0, 1])
dendrogram(Zl, ax=axd, no_labels=True, color_threshold=0, above_threshold_color='#555555')
axd.axis('off')
axd.set_ylim(axd.get_ylim()[::-1])
axl = fig.add_subplot(gs[0, 0])
axl.axis('off')
handles = [Rectangle((0, 0), 1, 1, fc=gcol[g]) for g in gcol]
axl.legend(handles, list(gcol), loc='center', ncol=2, fontsize=5.8,
           handlelength=0.8, columnspacing=0.5, handletextpad=0.25, frameon=False)
axh = fig.add_subplot(gs[1, 0])
lab = np.array([[list(gcol).index(C.SENS_GROUP[SDIM[i]])] for i in order])
cmap_lab = matplotlib.colors.ListedColormap([gcol[g] for g in gcol])
axh.imshow(lab, aspect='auto', cmap=cmap_lab, vmin=0, vmax=len(gcol) - 1)
axh.set_xticks([]); axh.set_yticks(range(len(names)))
axh.set_yticklabels(names, fontsize=5.8); axh.tick_params(length=0, pad=1)
for s in axh.spines.values():
    s.set_visible(False)
axm = fig.add_subplot(gs[1, 1])
S = cs.values[np.ix_(order, order)]
im = axm.imshow(S, cmap=CMAP_DIV, norm=TwoSlopeNorm(0, -1, 1), interpolation='nearest')
# 对称矩阵：只保留一侧标签，避免两侧挤压导致印刷字号过小
axm.set_xticks([])
axm.set_yticks([]); axm.tick_params(length=0)
cax = fig.add_subplot(gs[1, 2])
cb = fig.colorbar(im, cax=cax)
cb.set_label('Spearman ρ', fontsize=7); cb.ax.tick_params(labelsize=6)
save(fig, 'fig02_sensory_corr.png',
     '感官维度聚类的相关结构：Aroma/Flavour 两套各自成块，缺陷维度与酯香负相关')

# ============================================================ 图3 化合物共表达网络
print('图3 化合物共表达网络 ...')
B_, mp, mt, ms = C.corr_cc_aligned(COMP, META)
Ab = np.abs(np.nan_to_num(B_)); np.fill_diagonal(Ab, 0)
thr = 0.45
import networkx as nx
lab, G, Qmod = C.communities(COMP, META, thr=thr)
mod_s = pd.Series(lab, index=COMP)
cnt = mod_s.value_counts()
big = cnt[cnt >= 8].index.tolist()
pos = nx.spring_layout(G, seed=7, k=0.42, iterations=240, weight='weight')

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={'width_ratios': [1.35, 1]})
ax = axes[0]
pal_mod = [BLUE, RED, GREEN, PURPLE, '#E08A1E', '#2E9E4F']
ncol = {m: pal_mod[i % len(pal_mod)] for i, m in enumerate(big)}
ncol['其他'] = '#C9CDD2'
for u, v, d in G.edges(data=True):
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='#C2C8CF',
            lw=0.14 + 1.3 * (d['weight'] - thr), alpha=0.32, zorder=1,
            solid_capstyle='round')
nodes = list(G.nodes)
xs = np.array([pos[n][0] for n in nodes]); ys = np.array([pos[n][1] for n in nodes])
deg = np.array([G.degree(n) for n in nodes])
sz = np.array([10 + 4.5 * d for d in deg])
cl = [ncol.get(mod_s[COMP[n]], ncol['其他']) for n in nodes]
ax.scatter(xs, ys, s=sz, c=cl, edgecolor='white', linewidth=0.3, zorder=2, alpha=0.95)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.text(0.01, 0.02, '节点 %d，边 %d（|r|≥%.2f），社区 %d 个\n模块度 $Q$=%.3f，平均度 %.1f' % (
    G.number_of_nodes(), G.number_of_edges(), thr, len(cnt), Qmod, deg.mean()),
    transform=ax.transAxes, fontsize=6.0, va='bottom', linespacing=1.4)
handles = [plt.Line2D([], [], marker='o', ls='', mfc=ncol.get(m, ncol['其他']),
                      mec='white', ms=5) for m in big]
handles.append(plt.Line2D([], [], marker='o', ls='', mfc=ncol['其他'], mec='white', ms=5))
axes[0].legend(handles, ['C%d(%d)' % (m, cnt[m]) for m in big] + ['其他/孤立'],
               loc='upper right', fontsize=5.8, ncol=2, handletextpad=0.2,
               columnspacing=0.7)
panel(ax, 'a', '共表达网络（力导向布局）')

ax = axes[1]
mm = pd.DataFrame(dict(module=lab, family=[C.family_of(c) for c in COMP]))
fams = [f for f in mm['family'].value_counts().index[:6]] + ['其他']
mm['fam2'] = np.where(mm['family'].isin(fams[:-1]), mm['family'], '其他')
fams = [f for f in mm['fam2'].value_counts().index if f != '其他'] + ['其他']
top = big[:8]
bottom = np.zeros(len(top))
famcol = {f: CLEAN[i % len(CLEAN)] for i, f in enumerate(fams)}
famcol['其他'] = '#B8BDC4'
for f in fams:
    vals = np.array([((mm.module == m) & (mm.fam2 == f)).sum() for m in top])
    ax.bar(range(len(top)), vals, bottom=bottom, color=famcol[f],
           width=0.68, edgecolor='white', linewidth=0.4, label=f)
    bottom += vals
ax.set_xticks(range(len(top)))
ax.set_xticklabels(['C%d' % m for m in top], fontsize=6.8)
ax.set_ylabel('化合物个数'); ax.set_xlabel('共表达社区')
ax.legend(fontsize=5.6, ncol=2, loc='upper right', handlelength=0.8, columnspacing=0.6)
panel(ax, 'b', '社区的化学类别构成')
fig.tight_layout(w_pad=1.2)
save(fig, 'fig03_compound_network.png',
     '化合物共表达网络：酯类与萜类各自成社区，跨社区强边连接酯-醇-酸代谢链')

print('\n图 1–3 完成')
