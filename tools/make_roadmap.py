"""AI-assisted DeepSeek Harness, 2026-10-29. Technical-route figure for the v3 manuscript.

Draws the single 图1 referenced by 论文/v3.md §1.3 and §3.4 (both placeholders point to
this figure). All figures quoted inside come from configs/ and outputs/rebuild/ only.

Output: build_v3/figures/16_roadmap.png (+ .pdf)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "build_v3" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False

C_DATA = "#DCE6F1"
C_PROTO = "#F2F2F2"
C_Q1 = "#E2EFDA"
C_Q2 = "#FFF2CC"
C_Q3 = "#FBE5D6"
C_VAL = "#EDEDED"
C_DELIV = "#D9D2E9"
EDGE = "#3A3A3A"


def box(ax, x, y, w, h, text, color, fs=8.8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.04",
                                linewidth=1.0, edgecolor=EDGE, facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.55)


def down(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=10, linewidth=1.0, color=EDGE,
                                 shrinkA=0, shrinkB=0))


def right(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=10, linewidth=1.0, color=EDGE,
                                 shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(11.6, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.10, 1.06)
    ax.axis("off")

    # ── 第一层：数据源与审计 ────────────────────────────────────
    box(ax, 0.04, 0.845, 0.255, 0.105,
        "官方附件 1–8\n250 款 × 231 项化学/理化\n50 项感官汇总 · 22 种风格", C_DATA, 8.6)
    box(ax, 0.335, 0.845, 0.205, 0.105,
        "原研究公开逐评分员记录\n2197 条 · 16 名评分员\n37 个日期", C_DATA, 8.6)
    box(ax, 0.575, 0.845, 0.385, 0.105,
        "数据审计：名称映射与尺度核实（log10 拟合误差 2.9e-10）\n"
        "评分卡父子规则 · 重复品评 · 观测最低值堆积", C_DATA, 8.6)
    right(ax, 0.295, 0.335, 0.8975)
    right(ax, 0.540, 0.575, 0.8975)

    # ── 第二层：统一协议 ────────────────────────────────────────
    down(ax, 0.32, 0.845, 0.775)
    box(ax, 0.04, 0.690, 0.92, 0.085,
        "统一实验协议（训练前冻结）：2 个种子 × 三折 · 同款啤酒整体留于一折\n"
        "内层 20% 选参 · Q3 分区 140/35/40/35 · 填补/标准化/选择仅在训练角色内拟合", C_PROTO, 8.4)

    # ── 第三层：三问并列 ────────────────────────────────────────
    qy, qh = 0.390, 0.235
    box(ax, 0.03, qy, 0.29, qh,
        "问题一　感官评分预测\n\n"
        "等权面板目标（先评分员内、\n再评分员间等权）\n"
        "5 类基线 + MLP + 2 种层次网络\n\n"
        "选用 ExtraTrees\n"
        "MAE 0.4454 · RMSE 0.6305", C_Q1, 8.5)
    box(ax, 0.355, qy, 0.29, qh,
        "问题二　化合物–风味关联\n\n"
        "231 项有符号共表达图\n"
        "四种筛选规则 + 训练内选择\n"
        "固定 2+1：0.4988 / 全流程 0.5024\n\n"
        "交付 50 维 × Top-3 = 150 项\n"
        "Bootstrap 入选频率 · 替代组", C_Q2, 8.5)
    box(ax, 0.68, qy, 0.29, qh,
        "问题三　条件生成与跨风格迁移\n\n"
        "4 类生成器：PCA 高斯 / 独立 Copula /\n"
        "联合 Copula / CVAE；22 风格 × 3 场景\n\n"
        "生成谱 → 冻结评分器 → 50 维画像\n"
        "9 源酒 × 4 策略；越阈 7 → 0 次", C_Q3, 8.5)

    for x in (0.175, 0.50, 0.825):
        down(ax, x, 0.690, qy + qh)

    # ── 第四层：验证 ────────────────────────────────────────────
    vy, vh = 0.215, 0.130
    box(ax, 0.03, vy, 0.29, vh,
        "内部重采样验证\n2 次三折 · 啤酒级 Bootstrap\nWilcoxon + Holm 校正\n（非外部新队列）", C_VAL, 8.4)
    box(ax, 0.355, vy, 0.29, vh,
        "稳定性与敏感性\n100 次 Bootstrap · 替代组 Jaccard\n缺失遮蔽 5/10/20%\n去 barnyard · Z 口径对照", C_VAL, 8.4)
    box(ax, 0.68, vy, 0.29, vh,
        "独立评价器复核\n评价器优化前序列化 · 哈希冻结\n候选选定后才评分\n设计角色与评价角色隔离", C_VAL, 8.4)

    for x in (0.175, 0.50, 0.825):
        down(ax, x, qy, vy + vh)

    # ── 第五层：交付层 ─────────────────────────────────────────
    box(ax, 0.03, 0.045, 0.94, 0.120,
        "交付层\n250 × 53 评分卡（50 个有监督字段 + 2 个阈值锚定列）｜ 50 × 3 = 150 项候选因素\n"
        "22 风格 880 条生成谱与感官画像 ｜ 9 款源酒的四策略迁移审计", C_DELIV, 8.4, bold=True)
    for x in (0.175, 0.50, 0.825):
        down(ax, x, vy, 0.165)

    # ── 标题与边界 ─────────────────────────────────────────────
    ax.text(0.5, 1.015, "图1　三问统一技术路线", ha="center", va="center",
            fontsize=12.5, fontweight="bold")
    ax.text(0.5, -0.035,
            "注：全部数值取自 outputs/rebuild/ 与 configs/ 冻结协议；三问共用 231 项化学编码、"
            "250 款样本与 22 种风格标签。\n所有结论均为同一批啤酒的内部重采样证据，"
            "不构成外部新批次泛化、真实酿造或感官盲评验证。",
            ha="center", va="center", fontsize=8.0, color="#4A4A4A", linespacing=1.6)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"16_roadmap.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("written", OUT / "16_roadmap.png")


if __name__ == "__main__":
    main()
