# -*- coding: utf-8 -*-
"""08 图件 · 问题三结果（图11–图14）

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。

配色契约（全图统一，克制的科研风格）：
  主色墨蓝 INK、强调砖红 ACC、中性灰 GRY、浅底 LGT
  —— 不用彩虹色，颜色只承载语义（指标身份 / 样本量 / 基准线）。
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

IMG = os.path.join(C.ROOT, 'latex', 'images')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'font.size': 8, 'axes.titlesize': 8.6, 'axes.labelsize': 7.8,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'legend.fontsize': 6.4,
    'legend.frameon': False, 'figure.dpi': 120, 'savefig.dpi': 450,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
INK = '#23405C'
ACC = '#A8452C'
GRY = '#8C939B'
LGT = '#E4E8EC'
CMAP_DIV = LinearSegmentedColormap.from_list(
    'byr', ['#2166AC', '#4393C3', '#92C5DE', '#F7F7CF', '#FDD07B', '#F4A582', '#B2182B'], N=256)
CMAP_SEQ = LinearSegmentedColormap.from_list(
    'ink', ['#EAF0F5', '#9DB4C8', '#5C7F9E', '#33587A', '#1B3247'], N=256)


def panel(ax, tag, title=None, dx=-0.15, dy=1.05):
    ax.text(dx, dy, '(%s)' % tag, transform=ax.transAxes, fontsize=9.5, fontweight='bold',
            va='top', ha='left', color='#1A1A1A')
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
S = json.load(open(os.path.join(C.OUT, 'q3_summary.json'), encoding='utf-8'))
GQ = pd.read_csv(os.path.join(C.OUT, 'q3_generation_quality.csv'))
TR = pd.read_csv(os.path.join(C.OUT, 'q3_transfer_sensory.csv'))
TC = pd.read_csv(os.path.join(C.OUT, 'q3_transfer_compounds.csv'))
A = np.load(os.path.join(C.OUT, 'q3_arrays.npz'))
gen = pd.read_pickle(os.path.join(C.OUT, 'q3_generated.pkl'))

X = D['X'].values.astype(float)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X)
Xz = sc.transform(SimpleImputer(strategy='median').fit_transform(X))
Zmu, Zsd, V = A['Zmu'], A['Zsd'], A['V']
STYLE = D['style']
Zn = ((Xz - Xz.mean(0)) @ V - Zmu) / Zsd
RAND = 1 / 22.0


def ink_ramp(v, lo, hi):
    """按样本量把颜色映射到墨蓝的明度阶梯（深=样本多）。"""
    t = 0.0 if hi == lo else (v - lo) / (hi - lo)
    base = np.array(to_rgb(INK))
    white = np.array([0.97, 0.97, 0.97])
    return tuple(white + (base - white) * (0.45 + 0.55 * t))


# ============================================================ 图11 潜空间
print('图11 潜空间分布 ...')
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={'width_ratios': [1.25, 1]})
ax = axes[0]
show = [s for s in ['Tripel', 'Kriek', 'Lambic', 'Pils/Lager'] if s in gen]
cols = [INK, ACC, '#5C7F9E', '#8C939B']
for s, c in zip(show, cols):
    m = STYLE == s
    ax.scatter(Zn[m, 0], Zn[m, 1], s=26, color=c, alpha=0.75, edgecolor='white',
               lw=0.35, label='%s 实测' % s, zorder=3)
    Lx = gen[s]
    zz = ((sc.transform(Lx) - Xz.mean(0)) @ V - Zmu) / Zsd
    ax.scatter(zz[:100, 0], zz[:100, 1], s=11, color=c, alpha=0.42, marker='^',
               edgecolor='none', zorder=2)
ax.set_xlabel('潜因子 1'); ax.set_ylabel('潜因子 2')
_xh = 2.6
_n_out = int((Zn[:, 0] > _xh).sum())
for _s in gen:
    _z = ((sc.transform(gen[_s]) - Xz.mean(0)) @ V - Zmu) / Zsd
    _n_out += int((_z[:, 0] > _xh).sum())
ax.set_xlim(-1.9, _xh)
ax.text(0.985, 0.965, '另有 %d 个生成样本落在 $x>%.1f$（未显示）' % (_n_out, _xh),
        transform=ax.transAxes, fontsize=5.6, color='#777777', ha='right', va='top')
ax.legend(fontsize=5.8, ncol=2, handletextpad=0.2, columnspacing=0.5,
          loc='lower left', bbox_to_anchor=(0.0, -0.02))
ax.text(0.985, 0.03, '○ 实测   △ 生成', transform=ax.transAxes, fontsize=6.2,
        ha='right', va='bottom', color='#333333')
panel(ax, 'a', '风格条件生成与实测（潜空间前两维）')

# (b) 各风格 Fréchet —— 横向 lollipop，明度编码实际样本量
ax = axes[1]
g = GQ.sort_values('Fréchet', ascending=True).reset_index(drop=True)
nlo, nhi = g['n_real'].min(), g['n_real'].max()
for i, r in g.iterrows():
    ax.plot([0, r['Fréchet']], [i, i], color=LGT, lw=2.0, solid_capstyle='round', zorder=1)
    ax.scatter(r['Fréchet'], i, s=44, color=ink_ramp(r['n_real'], nlo, nhi),
               edgecolor=INK, linewidth=0.5, zorder=3)
    ax.text(r['Fréchet'] + 4, i, '%.0f' % r['Fréchet'], va='center', fontsize=6.0,
            color='#333333')
ax.set_yticks(range(len(g)))
ax.set_yticklabels(['%s (%d)' % (r['风格'], r['n_real']) for _, r in g.iterrows()],
                   fontsize=6.4)
ax.set_xlabel('Fréchet 距离（越小越接近真实分布）')
ax.set_xlim(0, g['Fréchet'].max() * 1.18)
ax.set_ylim(-0.6, len(g) - 0.4)
ax.grid(axis='x', ls=':', lw=0.4, color=LGT)
panel(ax, 'b', '各风格 Fréchet 距离（点明暗 = 实际样本量）')
fig.tight_layout(w_pad=1.2)
save(fig, 'fig11_latent.png')

# ============================================================ 图12 生成质量（重做）
print('图12 生成质量四联（重做）...')
fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.7))

# ---- (a) 风格保真度 lollipop，含随机基线与分类器上限
ax = axes[0, 0]
g = GQ.sort_values('风格保真度', ascending=True).reset_index(drop=True)
nlo, nhi = g['n_real'].min(), g['n_real'].max()
for i, r in g.iterrows():
    ax.plot([RAND, r['风格保真度']], [i, i], color=LGT, lw=2.0, solid_capstyle='round',
            zorder=1)
    ax.scatter(r['风格保真度'], i, s=46, color=ink_ramp(r['n_real'], nlo, nhi),
               edgecolor=INK, linewidth=0.5, zorder=3)
    ax.text(r['风格保真度'] + 0.02, i, '%.2f' % r['风格保真度'], va='center', fontsize=6.0)
ax.axvline(RAND, color=GRY, lw=0.8, ls='--')
ax.axvline(S['classifier_cv_acc'], color=ACC, lw=0.9, ls='-.')
ax.text(RAND + 0.012, len(g) - 0.55, '随机基线 0.045', fontsize=5.9, color=GRY, va='top')
ax.text(S['classifier_cv_acc'] + 0.015, -0.80,
        '分类器真实数据交叉验证上限 %.3f' % S['classifier_cv_acc'],
        fontsize=5.8, color=ACC, va='bottom', ha='left')
ax.set_yticks(range(len(g)))
ax.set_yticklabels(g['风格'], fontsize=6.4)
ax.set_xlim(0, 1.12)
ax.set_ylim(-1.05, len(g) - 0.35)
ax.set_xlabel('生成样本被判为同风格的比例')
ax.grid(axis='x', ls=':', lw=0.4, color=LGT)
panel(ax, 'a', '风格保真度')

# ---- (b) 保真度—多样性平面
ax = axes[0, 1]
for _, r in GQ.iterrows():
    ax.scatter(r['多样性比'], r['风格保真度'], s=20 + 2.2 * r['n_real'],
               color=ink_ramp(r['n_real'], nlo, nhi), edgecolor=INK, linewidth=0.6,
               zorder=3)
    ax.annotate(r['风格'], (r['多样性比'], r['风格保真度']),
                textcoords='offset points', xytext=(5, 4), fontsize=5.8, color='#333333')
ax.axvline(1.0, color=ACC, lw=0.9, ls='--')
ax.add_patch(Rectangle((0.95, GQ['风格保真度'].mean()), 0.07, 1.16,
                         facecolor='#F3F6F4', edgecolor='none', zorder=0))
ax.axhline(GQ['风格保真度'].mean(), color=GRY, lw=0.7, ls=':')
ax.text(0.985, 0.62, '理想值 1.0', fontsize=5.8, color=ACC, rotation=90, va='bottom',
        ha='right')
ax.set_xlabel('多样性比（生成/真实 标准差）')
ax.set_ylabel('风格保真度')
ax.set_xlim(0.60, 1.02)
ax.set_ylim(0.45, 1.16)
ax.text(0.03, 0.05, '标注点大小 ∝ 实际样本数', transform=ax.transAxes, fontsize=5.8,
        color=GRY)
panel(ax, 'b', '保真度与多样性的权衡')

# ---- (c) 两个独立分布指标的一致性
ax = axes[1, 0]
from scipy.stats import spearmanr
rho = spearmanr(GQ['Fréchet'], GQ['能量距离']).statistic
for _, r in GQ.iterrows():
    ax.scatter(r['Fréchet'], r['能量距离'], s=20 + 2.2 * r['n_real'],
               color=ink_ramp(r['n_real'], nlo, nhi), edgecolor=INK, linewidth=0.6, zorder=3)
    ax.annotate(r['风格'], (r['Fréchet'], r['能量距离']), textcoords='offset points',
                xytext=(5, 3), fontsize=5.8, color='#333333')
z = np.polyfit(GQ['Fréchet'], GQ['能量距离'], 1)
xs = np.linspace(GQ['Fréchet'].min() * 0.95, GQ['Fréchet'].max() * 1.03, 30)
ax.plot(xs, np.polyval(z, xs), color=ACC, lw=1.0, ls='--')
ax.set_xlabel('Fréchet 距离')
ax.set_ylabel('能量距离')
ax.text(0.04, 0.93, 'Spearman $\\rho$=%.2f' % rho, transform=ax.transAxes, fontsize=6.6,
        va='top')
panel(ax, 'c', '两个独立指标是否互相印证')

# ---- (d) 归一化质量点阵（替换原“文字面板”）
ax = axes[1, 1]
Q = GQ.copy()
def norm_lo(x):
    return 1 - (x - x.min()) / (x.max() - x.min() + 1e-12)
fd = norm_lo(Q['Fréchet']); en = norm_lo(Q['能量距离'])
dv = 1 - (Q['多样性比'] - 1).abs() / (Q['多样性比'] - 1).abs().max()
Q['s_fd'], Q['s_en'], Q['s_dv'] = fd, en, dv
Q['mean'] = Q[['s_fd', 's_en', 's_dv']].mean(1)
Q = Q.sort_values('mean').reset_index(drop=True)
MK = dict(s_fd=('o', INK, 'Fréchet'), s_en=('s', ACC, '能量距离'),
          s_dv=('^', '#5C7F9E', '多样性'))
ax.axvspan(0.8, 1.02, color='#F3F6F4', zorder=0)
for i, r in Q.iterrows():
    vals = [r['s_fd'], r['s_en'], r['s_dv']]
    ax.plot([min(vals), max(vals)], [i + 0.16, i + 0.16], color=LGT, lw=1.6, zorder=1)
    for k, (mk, col, lab) in MK.items():
        ax.scatter(r[k], i + 0.16, marker=mk, s=26, color=col, edgecolor='white',
                   linewidth=0.4, zorder=3)
ax.set_yticks(np.arange(len(Q)) + 0.16)
ax.set_yticklabels(Q['风格'], fontsize=6.4)
ax.set_xlim(0, 1.04)
ax.set_ylim(-0.95, len(Q) - 0.15)
ax.set_xlabel('归一化质量得分（1 = 最优，三项指标映射到同一刻度）')
ax.grid(axis='x', ls=':', lw=0.4, color=LGT)
handles = [Line2D([], [], marker=m, ls='', mfc=c, mec='white', ms=4.5, label=l)
           for m, c, l in MK.values()]
ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.015), fontsize=6.0,
          handletextpad=0.2, columnspacing=1.2, ncol=3, framealpha=0)
ax.text(0.81, len(Q) - 0.25, '优秀区', fontsize=5.8, color=GRY, va='top')
panel(ax, 'd', '三指标归一化对照')

fig.tight_layout(w_pad=1.4, h_pad=1.5)
save(fig, 'fig12_quality.png')

# ============================================================ 图13 迁移结果
print('图13 跨风格迁移 ...')
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={'width_ratios': [1.15, 1]})
ax = axes[0]
key = TR['感官维度'].tolist()
xx = np.arange(len(key))
ax.plot(xx, TR['迁移前'], '-o', color=GRY, ms=3.6, lw=1.0, label='Pils/Lager 原样')
ax.plot(xx, TR['最优传输'], '-s', color='#5C7F9E', ms=3.6, lw=1.0,
        label='最优传输（无约束）')
ax.plot(xx, TR['约束后'], '-^', color=ACC, ms=4.2, lw=1.4, label='约束后（本文）')
ax.plot(xx, TR['目标风格均值'], ':', color=INK, lw=1.2, label='Tripel 真实均值')
ax.axhline(0, color='#B0B6BC', lw=0.6, ls=':')
ax.set_xticks(xx)
ax.set_xticklabels(key, rotation=40, ha='right', fontsize=5.8)
ax.set_ylabel('感官预测值（标准化）')
ax.legend(fontsize=6.0, ncol=2, handletextpad=0.3, columnspacing=0.6)
panel(ax, 'a', '迁移前后关键感官维度')

ax = axes[1]
tc = TC.head(8).iloc[::-1]
yy = np.arange(len(tc))
vals = np.log10(tc['倍数变化'].clip(lower=1e-3))
ax.hlines(yy, 0, vals, color=LGT, lw=2.0, zorder=1)
ax.scatter(vals, yy, s=34, color=np.where(vals >= 0, ACC, INK), edgecolor='white',
           linewidth=0.5, zorder=3)
ax.set_yticks(yy)
ax.set_yticklabels([str(s)[:22] for s in tc['化学名']], fontsize=5.8)
ax.axvline(0, color='#7A8188', lw=0.7)
for i, v in enumerate(tc['倍数变化']):
    ax.text(np.log10(max(v, 1e-3)) + (0.12 if np.log10(max(v, 1e-3)) >= 0 else -0.12),
            i, '×%.2f' % v, va='center', ha='left' if np.log10(max(v, 1e-3)) >= 0 else 'right',
            fontsize=5.6)
ax.set_xlabel(r'$\log_{10}$(迁移后 / 迁移前 浓度倍数)')
ax.set_xlim(min(vals.min() * 1.9, -1.0), max(vals.max() * 1.45, 1.0))
panel(ax, 'b', '迁移引起的最大浓度变化')
fig.tight_layout(w_pad=1.2)
save(fig, 'fig13_transfer.png')

# ============================================================ 图14 缺陷约束
print('图14 缺陷约束 ...')
lam = A['lam']
S_b, S_a, S_f = A['S_before'], A['S_after'], A['S_fin']
de = [SDIM.index(c) for c in C.DEFECT_SENS]


def dscore(Sx):
    return Sx[:, de].mean(1)


fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={'width_ratios': [1, 1, 0.85]})
ax = axes[0]
ax.hist(lam, bins=np.linspace(0, 1, 13), color='#5C7F9E', edgecolor='white', lw=0.5)
ax.axvline(lam.mean(), ls='--', lw=0.9, color=ACC)
ax.text(lam.mean() - 0.03, ax.get_ylim()[1] * 0.92, '均值 %.2f' % lam.mean(), fontsize=6.2,
        color=ACC, ha='right')
ax.set_xlabel('迁移强度 $\\lambda$'); ax.set_ylabel('样本数')
panel(ax, 'a', '迁移强度分布', dx=-0.24)

ax = axes[1]
bs = dscore(S_b); as_ = dscore(S_a); fs = dscore(S_f)
ax.scatter(bs, as_, s=18, facecolor='none', edgecolor='#5C7F9E', lw=0.8,
           label='最优传输')
ax.scatter(bs, fs, s=18, color=ACC, alpha=0.85, edgecolor='white', lw=0.3,
           label='约束后')
lim = [min(bs.min(), as_.min(), fs.min()) - 0.05, max(bs.max(), as_.max(), fs.max()) + 0.05]
ax.plot(lim, lim, '--', lw=0.9, color=GRY)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('迁移前缺陷分'); ax.set_ylabel('迁移后缺陷分')
ax.legend(fontsize=5.8, loc='upper left', handletextpad=0.2)
ax.text(lim[0] + 0.02, lim[1] - 0.03, '对角线下方 = 缺陷未恶化', fontsize=5.6, color=GRY,
        va='top')
panel(ax, 'b', '缺陷单调性检验', dx=-0.26)

ax = axes[2]
rows = [('迁移前', bs.mean(), GRY), ('无约束传输', as_.mean(), '#5C7F9E'),
        ('约束后', fs.mean(), ACC)]
yy = np.arange(len(rows))
vals = [r[1] for r in rows]
ax.plot(vals, yy, '-', color=LGT, lw=2.2, zorder=1)
for i, (lab, v, c) in enumerate(rows):
    ax.scatter(v, i, s=46, color=c, edgecolor='white', lw=0.6, zorder=3)
    ax.text(v, i + 0.24, '%.3f' % v, ha='center', fontsize=6.0, color=c)
ax.set_yticks(yy)
ax.set_yticklabels([r[0] for r in rows], fontsize=6.4)
ax.set_ylim(-0.5, len(rows) - 0.4)
ax.set_xlabel('平均缺陷分')
ax.grid(axis='x', ls=':', lw=0.4, color=LGT)
panel(ax, 'c', '缺陷分变化轨迹', dx=-0.30)
fig.tight_layout(w_pad=1.1)
save(fig, 'fig14_defect.png')
print('\n图 11–14 完成')
