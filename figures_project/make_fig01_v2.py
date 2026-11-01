# -*- coding: utf-8 -*-
"""图1 技术路线（依据队友 draw.io 流程图重绘，2×2 布局 + 三条跨模块传出关系）。

原 SVG 结构（从 C:\\Users\\73173\\Downloads\\流程图.svg 的 24 个文字节点与矩形坐标反推）：
  左列 x≈23–323，右列 x≈444–744；三行 y≈32–308 / 380–656 / 后续
  四个模块各含四个步骤；三条跨列标签：
    统一监督数据   (审计 → 问题一)
    统一化学特征   (审计 → 问题二)
    化合物解释支持 (问题二 → 问题三)
    复用感官评分器 (问题一 → 问题三，图上第四条虚线)

输出：最终图件/fig01_技术路线.png + .pdf，15.2 cm 宽、6–8 pt。
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

CM = 1 / 2.54
FIG_W = 15.2
HERE = Path(os.path.expanduser(r"~\Downloads\论文跑图专用文件夹"))
OUT = HERE / "最终图件"

NAVY = "#18344A"; INK = "#333333"; MUTED = "#6F7C88"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

# 四模块配色（沿用原 draw.io 的语义色）
C_AUDIT = ("#DCE6F1", "#6C8EBF")   # 蓝：审计
C_Q1    = ("#D5E8D4", "#82B366")   # 绿：问题一
C_Q2    = ("#E1D5E7", "#9673A6")   # 紫：问题二
C_Q3    = ("#FFF2CC", "#D6B656")   # 黄：问题三

MODULES = [
    # key, 标题, 四步骤, 配色, 左上角 (x,y)
    ("audit", "数据审计与监督重建",
     ["匿名映射与字段核验", "评分员等权聚合", "尺度及最低值处理", "统一建模数据"], C_AUDIT, (0.020, 0.585)),
    ("q1", "问题一：感官评分预测",
     ["建立 0–6 分监督目标", "多模型比较与验证", "ExtraTrees 回归", "50 维感官评分"], C_Q1, (0.520, 0.585)),
    ("q2", "问题二：关键化合物筛选",
     ["Spearman 有符号关联网络", "互补性与冗余分析", "固定 2+1 筛选及 Bootstrap", "50×3 关键化合物"], C_Q2, (0.020, 0.105)),
    ("q3", "问题三：生成与迁移",
     ["风格条件生成模型", "化学谱与感官画像", "风险约束迁移", "生成谱及迁移审计"], C_Q3, (0.520, 0.105)),
]

BW, BH = 0.450, 0.325     # 模块框宽高
STEP_H = 0.052            # 单步条高
TITLE_H = 0.058


def draw_module(ax, key, title, steps, colors, xy):
    x, y = xy
    face, edge = colors
    ax.add_patch(FancyBboxPatch((x, y), BW, BH,
                                boxstyle="round,pad=0.004,rounding_size=0.014",
                                lw=1.0, edgecolor=edge, facecolor=face, alpha=.55))
    # 标题
    ax.text(x + BW / 2, y + BH - TITLE_H / 2 - 0.008, title,
            ha="center", va="center", fontsize=7.8, color=NAVY, weight="bold")
    # 四步骤条
    top = y + BH - TITLE_H - 0.016
    for i, s in enumerate(steps):
        sy = top - (i + 1) * STEP_H - i * 0.008
        ax.add_patch(FancyBboxPatch((x + 0.020, sy), BW - 0.040, STEP_H,
                                    boxstyle="round,pad=0.002,rounding_size=0.008",
                                    lw=.7, edgecolor=edge, facecolor="white"))
        ax.text(x + BW / 2, sy + STEP_H / 2, f"({i+1}) {s}",
                ha="center", va="center", fontsize=6.6, color=INK)


def arrow(ax, x0, y0, x1, y1, label=None, style="-|>", color="#3A3A3A",
          lw=.9, ls="-", lx=0.5, ly=0.5, lrot=0, dashed=False, z=6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=8, lw=lw, color=color,
                                 shrinkA=2, shrinkB=2, zorder=z,
                                 linestyle=(0, (4, 3)) if dashed else ls))
    if label:
        ax.text(lx, ly, label, ha="center", va="center", fontsize=6.2,
                color=color, rotation=lrot, zorder=z + 1,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=.92))


def main():
    fig = plt.figure(figsize=(FIG_W * CM, 11.2 * CM))
    ax = fig.add_axes([.005, .055, .99, .895])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    for key, title, steps, colors, xy in MODULES:
        draw_module(ax, key, title, steps, colors, xy)

    L = 0.020; R = 0.520          # 两列左边界
    xL2R = L + BW                 # 左列右边缘 0.470
    y_top = 0.585 + BH            # 0.910
    y_bot = 0.105 + BH            # 0.430

    # ── 三条跨模块传出关系（z 高于模块框，避免被遮住） ──
    # 1) 审计 → 问题一（顶部横箭头，"统一监督数据"）
    arrow(ax, xL2R + 0.003, y_top, R - 0.003, y_top,
          "统一监督数据", color="#386A91", lw=1.2, z=8,
          lx=(xL2R + R) / 2, ly=y_top + 0.042)

    # 2) 审计 → 问题二（竖向，"统一化学特征"）
    x_left_mid = L + BW / 2                    # 0.245
    arrow(ax, x_left_mid, 0.585 - 0.003, x_left_mid, y_bot + 0.003,
          "统一化学特征", color="#386A91", lw=1.2, z=8,
          lx=x_left_mid + 0.062, ly=(0.585 + y_bot) / 2, lrot=90)

    # 3) 问题二 → 问题三（底部横箭头，"化合物解释支持"）
    arrow(ax, xL2R + 0.003, y_bot, R - 0.003, y_bot,
          "化合物解释支持", color="#386A91", lw=1.2, z=8,
          lx=(xL2R + R) / 2, ly=y_bot + 0.042)

    # 4) 问题一 → 问题三（虚线，"复用感官评分器"）
    x_right_mid = R + BW / 2                   # 0.745
    arrow(ax, x_right_mid, 0.585 - 0.003, x_right_mid, y_bot + 0.003,
          "复用感官评分器", color="#9673A6", lw=1.1, dashed=True, z=8,
          lx=x_right_mid + 0.062, ly=(0.585 + y_bot) / 2, lrot=90)

    ax.text(.5, .012,
            "实线箭头＝数据与特征契约的传出；虚线箭头＝模型/评价器的复用（问题一冻结评分器供问题三打画像）",
            ha="center", va="bottom", fontsize=6.0, color=MUTED)
    ax.text(.5, 1.012, "图1　四模块技术路线", ha="center", va="bottom",
            fontsize=8.4, color=NAVY)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig01_四模块技术路线.png", dpi=320)
    fig.savefig(OUT / "fig01_四模块技术路线.pdf")
    plt.close(fig)
    print("ok fig01_四模块技术路线")


if __name__ == "__main__":
    main()
