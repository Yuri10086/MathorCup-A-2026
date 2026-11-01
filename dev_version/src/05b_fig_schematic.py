# -*- coding: utf-8 -*-
"""05b 图件 · 技术路线图（图0）与黑白算法流程图（图15）

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

IMG = os.path.join(r'E:\dsh_work\MC2026A', 'latex', 'images')
os.makedirs(IMG, exist_ok=True)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'pdf.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03, 'savefig.dpi': 500,
})
INK = '#1A1A1A'
BLUE = '#1F6FD0'
LGRAY = '#F2F4F7'
MGRAY = '#C9CDD2'


def box(ax, x, y, w, h, text, fs=6.4, fc='white', ec=INK, lw=0.9, bold=False, rad=0.02):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.004,rounding_size=%.3f' % rad,
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, zorder=3, linespacing=1.45,
            fontweight='bold' if bold else 'normal', color=INK)
    return p


def arrow(ax, p1, p2, style='-|>', lw=0.8, color=INK, ls='-', rad=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=8, lw=lw, color=color,
                        linestyle=ls, zorder=1,
                        connectionstyle='arc3,rad=%.2f' % rad, shrinkA=1, shrinkB=1)
    ax.add_patch(a)


# =====================================================================
# 图0 技术路线
# =====================================================================
fig, ax = plt.subplots(figsize=(7.0, 5.5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# 第一层：数据审计
box(ax, 0.5, 0.925, 0.94, 0.086,
    '第一层 · 数据审计（统一数据契约）\n'
    '结构对齐（beer 主键）｜对数底数 $10$ 反证｜$33$ 列原始浓度识别｜$102$ 列检出限左删失｜$0$–$6$ 分制保序标定',
    fs=6.2, fc=LGRAY, bold=False)

# 三列
xs = [0.18, 0.50, 0.82]
titles = ['问题一 · 感官评分', '问题二 · 化合物–风味关联', '问题三 · 风味指纹生成']
bodies = [
    '多任务图扩散回归\n\n· 缺失指示 + 稳健缩放\n· 共表达图扩散算子 $\\boldsymbol{S}$\n'
    '· 潜因子多任务（K=8/15）\n· MT-GNN 双分支编码\n· 堆叠集成 + 划分一致性区间',
    '共表达图 → 秩融合归因\n\n· 社区发现（模块度 $Q$）\n· 五路证据（$\\rho$/SHAP/置换/\n'
    '  稳定性选择/互信息）\n· bootstrap 入选频率\n· 跨方法 Jaccard\n· OAV 化学合理性',
    '风格条件潜因子流 + 约束迁移\n\n· 并行分析定 $K$\n· 仿射耦合流 $p(z|s)$\n'
    '· 缺陷约束拒绝采样\n· 高斯最优传输迁移\n· 单调二分投影保缺陷不增',
]
for x, t, b in zip(xs, titles, bodies):
    box(ax, x, 0.735, 0.29, 0.075, t, fs=6.8, fc=BLUE, ec=BLUE, bold=True, lw=1.0)
    ax.texts[-1].set_color('white')
    box(ax, x, 0.505, 0.29, 0.29, b, fs=6.0, fc='white')

for x in xs:
    arrow(ax, (x, 0.882), (x, 0.775), color=BLUE, lw=1.0)

# 三问之间的两条依赖（箭头含义见正文与图注）
arrow(ax, (0.325, 0.63), (0.355, 0.63), color=BLUE, lw=1.2)
arrow(ax, (0.645, 0.63), (0.675, 0.63), color=BLUE, lw=1.2)

# 验证层
box(ax, 0.5, 0.245, 0.94, 0.075,
    '验证层 · 三层检验（同一交叉验证口径）\n'
    '重复分层交叉验证（5 折 × 3 重复）｜消融实验｜参数敏感性｜分布层面生成质量（Fréchet / 能量距离 / 风格判别）',
    fs=6.2, fc=LGRAY)
for x in xs:
    arrow(ax, (x, 0.36), (x, 0.285), color=BLUE, lw=0.9)

# 交付层
box(ax, 0.5, 0.115, 0.94, 0.10,
    '交付层\n'
    '$250\\times50$ 的 $0$–$6$ 分制感官评分表｜$50$ 维 Top-3 关键化合物及稳定性与化学合理性论证\n'
    '风格条件生成器与跨风格迁移方案（缺陷化合物不恶化）',
    fs=6.1, fc='white', ec=INK, lw=1.1)
arrow(ax, (0.5, 0.207), (0.5, 0.162), color=INK, lw=1.0)
ax.text(0.5, 0.985, '全文技术路线', ha='center', va='top', fontsize=8.5, fontweight='bold')
fig.savefig(os.path.join(IMG, 'fig00_roadmap.pdf'))
fig.savefig(os.path.join(IMG, 'fig00_roadmap.png'))
plt.close(fig)
print('  ✓ fig00_roadmap')

# =====================================================================
# 图15 黑白算法流程图（MT-GNN 训练 + 缺陷约束迁移）
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')


def term(ax, x, y, w, h, t, fs=6.2):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle='round,pad=0.004,rounding_size=0.06',
                       fc='white', ec=INK, lw=1.0, zorder=2)
    ax.add_patch(p)
    ax.text(x, y, t, ha='center', va='center', fontsize=fs, zorder=3)


def proc(ax, x, y, w, h, t, fs=6.2):
    box(ax, x, y, w, h, t, fs=fs, fc='white', ec=INK, lw=1.0, rad=0.0)


def dec(ax, x, y, w, h, t, fs=6.2):
    ax.plot([x, x + w / 2, x, x - w / 2, x], [y + h / 2, y, y - h / 2, y, y + h / 2],
            color=INK, lw=1.0, zorder=2)
    ax.fill([x, x + w / 2, x, x - w / 2], [y + h / 2, y, y - h / 2, y], facecolor='white',
            edgecolor='none', zorder=1)
    ax.text(x, y, t, ha='center', va='center', fontsize=fs, zorder=3, linespacing=1.4)


# --- 左：MT-GNN 训练流程
ax = axes[0]
ax.set_title('(a) MT-GNN 训练与预测', fontsize=7.5, pad=3)
term(ax, 0.5, 0.945, 0.40, 0.085, '开始', fs=6.4)
proc(ax, 0.5, 0.815, 0.62, 0.095, '读入 $\\boldsymbol{X},\\boldsymbol{Y}$；中位数插补 + 缺失指示', fs=5.9)
proc(ax, 0.5, 0.685, 0.62, 0.095, '由 2.csv 建图 $\\boldsymbol{A}$ → 扩散算子 $\\boldsymbol{S}$', fs=5.9)
proc(ax, 0.5, 0.555, 0.62, 0.095, '特征增广 $[\\boldsymbol{X},\\ \\boldsymbol{S}\\boldsymbol{X}]$；1%/99% 缩尾 + IQR 稳健化', fs=5.7)
proc(ax, 0.5, 0.425, 0.62, 0.095, '双分支编码 → 拼接 → 多任务头（50 维输出）', fs=5.9)
proc(ax, 0.5, 0.295, 0.62, 0.095, 'SmoothL1 反传 + AdamW + 余弦退火（260 轮）', fs=5.9)
dec(ax, 0.5, 0.165, 0.46, 0.105, '训练轮数用尽？', fs=6.0)
term(ax, 0.5, 0.045, 0.44, 0.08, '输出 50 维感官预测', fs=6.0)
for y1, y2 in [(0.902, 0.863), (0.767, 0.733), (0.637, 0.603), (0.507, 0.473),
               (0.377, 0.343), (0.242, 0.218)]:
    arrow(ax, (0.5, y1), (0.5, y2), lw=0.9)
arrow(ax, (0.73, 0.165), (0.84, 0.165), lw=0.9)
arrow(ax, (0.84, 0.165), (0.84, 0.295), lw=0.9)
arrow(ax, (0.84, 0.295), (0.81, 0.295), lw=0.9)
ax.text(0.755, 0.185, '否', fontsize=5.8)
arrow(ax, (0.5, 0.113), (0.5, 0.086), lw=0.9)
ax.text(0.515, 0.098, '是', fontsize=5.8)

# --- 右：缺陷约束迁移
ax = axes[1]
ax.set_title('(b) 跨风格迁移（缺陷约束）', fontsize=7.5, pad=3)
term(ax, 0.5, 0.945, 0.42, 0.085, '开始（源风格样本）', fs=6.2)
proc(ax, 0.5, 0.825, 0.66, 0.09, '编码到潜空间 $\\boldsymbol{z}_s$；估计 $\\mathcal{N}(\\mu_s,\\Sigma_s)$ 与 $\\mathcal{N}(\\mu_t,\\Sigma_t)$', fs=5.6)
proc(ax, 0.5, 0.705, 0.66, 0.09, '高斯最优传输映射 $\\boldsymbol{z}_{\\mathrm{ot}}=\\mu_t+\\boldsymbol{A}(\\boldsymbol{z}_s-\\mu_s)$', fs=5.8)
proc(ax, 0.5, 0.585, 0.66, 0.09, '预测缺陷感官分；沿 $\\boldsymbol{z}_s\\!\\to\\!\\boldsymbol{z}_{\\mathrm{ot}}$ 单调二分搜索 $\\lambda$', fs=5.6)
dec(ax, 0.5, 0.455, 0.50, 0.10, '缺陷分不增？', fs=6.0)
proc(ax, 0.5, 0.325, 0.60, 0.09, '收缩 $\\lambda$（更保守的迁移强度）', fs=5.9)
term(ax, 0.5, 0.195, 0.60, 0.095, '输出迁移谱 $\\boldsymbol{z}=\\boldsymbol{z}_s+\\lambda(\\boldsymbol{z}_{\\mathrm{ot}}-\\boldsymbol{z}_s)$', fs=5.8)
proc(ax, 0.5, 0.075, 0.62, 0.085, '解码回化合物谱 + 感官画像', fs=5.9)
for y1, y2 in [(0.902, 0.870), (0.780, 0.750), (0.660, 0.630), (0.540, 0.505)]:
    arrow(ax, (0.5, y1), (0.5, y2), lw=0.9)
arrow(ax, (0.5, 0.405), (0.5, 0.370), lw=0.9)
ax.text(0.515, 0.388, '否', fontsize=5.8)
arrow(ax, (0.75, 0.455), (0.86, 0.455), lw=0.9)
arrow(ax, (0.86, 0.455), (0.86, 0.243), lw=0.9)
arrow(ax, (0.86, 0.243), (0.80, 0.243), lw=0.9)
ax.text(0.765, 0.470, '是', fontsize=5.8)
arrow(ax, (0.5, 0.147), (0.5, 0.118), lw=0.9)

fig.tight_layout(w_pad=1.6)
fig.savefig(os.path.join(IMG, 'fig15_algorithm.pdf'))
fig.savefig(os.path.join(IMG, 'fig15_algorithm.png'))
plt.close(fig)
print('  ✓ fig15_algorithm')
