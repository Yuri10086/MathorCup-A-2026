# -*- coding: utf-8 -*-
"""最终图件生成器：按已冻结的图号顺序（图1–图14）重画，统一 15.2 cm / 6–8 pt。

图号与文件一一对应，输出到 ../最终图件/，文件名前缀就是图号，
正文写作时直接 `图 N` ↔ `figN_xxx.png` 对照，不再重编。

    图1  技术路线             fig01_tech_route
    图2  数据审计六联         fig02_audit_six
    图3  风格与地板值         fig03_style_floor
    图4  模型横向比较         fig04_model_compare
    图5  逐维 R² 与增益       fig05_perdim
    图6  预测—实测六联        fig06_pred_scatter
    图7  敏感性与稳健性       fig07_sensitivity
    图8  共表达网络           fig08_network
    图9  四规则性能/稳定/冗余 fig09_q2_methods
    图10 阈值—OAV 化学合理性  fig10_oav
    图11 生成模型六维对照     fig11_gen_compare
    图12 三风格感官画像       fig12_portraits
    图13 逐风格支持范围+SWD   fig13_perstyle
    图14 迁移收益/风险/拒绝   fig14_migration
    附表图 S1 50维误差热图    figS1_mae_heatmap
    附表图 S2 150 候选总表    figS2_top3_matrix
    附表图 S3 训练曲线         figS3_train_curves

运行： python make_final_figures.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

CM = 1 / 2.54
FIG_W = 15.2          # cm，与版心同宽
HERE = Path(os.path.expanduser(r"~\Downloads\论文跑图专用文件夹"))
SUP = HERE / "MathorCup_A_Python绘图工程" / "support"
RB = SUP / "outputs" / "rebuild"
RAW_1 = SUP / "1.csv"
OFFICIAL = Path(os.path.expanduser(r"~\Downloads\MathorCup\MC2026A\MC2026A\data"))
OUT = HERE / "最终图件"

NAVY = "#18344A"; BLUE = "#386A91"; TEAL = "#168A81"; LIGHT = "#75B9AF"
CORAL = "#CF6753"; ORANGE = "#E7A65D"; GRAY = "#8596A6"; INK = "#253342"
MUTED = "#6F7C88"

COL_M = {"Mean": GRAY, "Ridge": "#8EA7B4", "PLS": "#B0BEC8", "RF": BLUE,
         "ET": TEAL, "HGB": "#5A91A9", "MLP": ORANGE, "Hierarchy": "#C98D79",
         "HierarchyRater": CORAL, "InnerSelected": "#49788E"}
COL_Q2 = {"Spearman": GRAY, "Hybrid2plus1": TEAL, "Forward3": BLUE,
          "Redundancy": "#B182A2", "TrainSelected": CORAL}
COL_G = {"PCA_Gaussian": GRAY, "CopulaIndependent": BLUE,
         "CopulaJoint": TEAL, "CVAE": CORAL}
COL_T = {"Unconstrained": GRAY, "PointConstraint": BLUE,
         "UncertaintyConstraint": TEAL, "Reject": CORAL}
LAB_Q2 = {"Spearman": "Spearman", "Hybrid2plus1": "2+1 条件增益",
          "Forward3": "完整前向", "Redundancy": "冗余约束",
          "TrainSelected": "训练内选法"}
LAB_G = {"PCA_Gaussian": "PCA-高斯", "CopulaIndependent": "独立 Copula",
         "CopulaJoint": "联合 Copula", "CVAE": "CVAE"}
LAB_T = {"Unconstrained": "无约束", "PointConstraint": "点预测约束",
         "UncertaintyConstraint": "不确定性约束", "Reject": "拒绝迁移"}


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.size": 6.6, "axes.titlesize": 7.6, "axes.labelsize": 6.6,
        "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 6.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "axes.edgecolor": "#9AA7B0",
        "xtick.major.width": .5, "ytick.major.width": .5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "grid.color": "#E3E8EC", "grid.linewidth": .5,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "lines.linewidth": 1.0,
    })


def panel(ax, letter, title, pad=5):
    ax.set_title(f"({letter})  {title}", loc="left", color=NAVY, fontsize=7.6, pad=pad)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=320)
    plt.close(fig)
    print("  ok", name)


def R(rel):
    return pd.read_csv(RB / rel)


# ------------------------------------------------------- 图1 技术路线
def fig01():
    fig = plt.figure(figsize=(FIG_W * CM, 12.2 * CM))
    fig.subplots_adjust(left=.012, right=.988, top=.985, bottom=.095)
    ax = fig.add_axes([.012, .095, .976, .890])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    C1, C2, C3, C4 = "#DCE6F1", "#E2EFDA", "#FFF2CC", "#FBE5D6"
    C5, C6 = "#F2F2F2", "#D9D2E9"
    EDGE = "#3A3A3A"

    def box(x, y, w, h, txt, color, fs=6.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.004,rounding_size=0.012",
                                    lw=.75, edgecolor=EDGE, facecolor=color,
                                    mutation_aspect=1.6))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.7)

    def down(x, y0, y1):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                     mutation_scale=6, lw=.6, color=EDGE,
                                     shrinkA=0, shrinkB=0))

    ax.text(.5, .985, "图1　三问统一技术路线", ha="center", va="top",
            fontsize=8.2, color=NAVY)

    # 第 1 行：数据源与审计
    box(.015, .780, .300, .145, "官方附件 1–8\n250 款 × 231 项化学/理化\n50 项感官 · 22 种风格", C1)
    box(.345, .780, .300, .145, "原研究公开逐评分员记录\n2197 条 · 16 名评分员\n37 个品评日期", C1)
    box(.680, .780, .305, .145, "数据审计\n名称映射 · log10 反证\n评分卡父子规则 · 观测地板", C1)

    # 第 2 行：统一协议
    box(.015, .630, .970, .105, "统一实验协议（训练前冻结）：2 种子 × 三折 · 同款啤酒整体留于一折\n"
                                "内层 20% 选参 · Q3 分区 140/35/40/35 · 填补/标准化/选择仅在训练角色内拟合", C5, 6.0)
    for x in (.165, .495, .832):
        down(x, .780, .735)

    # 第 3 行：三问
    qy, qh = .320, .265
    box(.015, qy, .300, qh, "问题一　感官评分预测\n\n等权面板目标（先评分员内、\n再评分员间等权）\n"
                            "5 类基线 + MLP + 2 种层次网络\n\n选用 ExtraTrees\nMAE 0.4454 · RMSE 0.6305", C2, 6.0)
    box(.345, qy, .300, qh, "问题二　化合物–风味关联\n\n231 项有符号共表达图\n四种筛选规则 + 训练内选择\n"
                            "固定 2+1：0.4988\n全流程：0.5024\n\n交付 50 × Top-3 = 150 项", C3, 6.0)
    box(.680, qy, .305, qh, "问题三　条件生成与迁移\n\n4 类生成器：PCA 高斯 /\n独立 Copula / 联合 Copula / CVAE\n"
                            "生成谱 → 冻结评分器 → 画像\n\n9 源酒 × 4 策略；越阈 7 → 0", C4, 6.0)
    for x in (.165, .495, .832):
        down(x, .630, qy + qh)

    # 第 4 行：验证
    vy, vh = .150, .130
    box(.015, vy, .300, vh, "内部重采样验证\n2 次三折 · 啤酒级 Bootstrap\nWilcoxon + Holm 校正", C5, 6.0)
    box(.345, vy, .300, vh, "稳定性与敏感性\n100 次 Bootstrap · 替代组 Jaccard\n缺失遮蔽 5/10/20% · 去 barnyard", C5, 6.0)
    box(.680, vy, .305, vh, "独立评价器复核\n评价器优化前冻结 · 哈希校验\n候选选定后才评分", C5, 6.0)
    for x in (.165, .495, .832):
        down(x, qy, vy + vh)

    # 第 5 行：交付
    box(.015, .010, .970, .115, "交付层\n250 × 53 评分卡（50 个有监督字段 + 2 个阈值锚定列）\n"
                                "50 × 3 = 150 项候选因素 ｜ 22 风格 880 条生成谱与感官画像\n"
                                "9 款源酒的四策略迁移审计", C6, 6.2, True)
    for x in (.165, .495, .832):
        down(x, vy, .125)

    fig.text(.5, .012, "注：数值取自 outputs/rebuild/ 冻结结果；三问共用 231 项化学编码、250 款样本与 22 种风格标签；\n"
                       "全部结论为同一批啤酒的内部重采样证据，不构成外部批次、真实酿造或感官盲评验证。",
             ha="center", va="bottom", fontsize=5.4, color="#4A4A4A", linespacing=1.5)
    save(fig, "fig01_技术路线")


# ------------------------------------------------ 图2 数据审计六联
def fig02():
    feat = R("audit/features.csv"); styles = R("audit/styles.csv").sort_values("n")
    tra = pd.read_csv(SUP / "data/processed/transforms.csv")
    tgt = R("audit/equal_rater_targets.csv")
    raw = pd.read_csv(RAW_1, low_memory=False).set_index("beer")
    mod = pd.read_csv(HERE / "改版样张" / "q2_modules.csv").set_index("compound")

    fig = plt.figure(figsize=(FIG_W * CM, 10.4 * CM))
    gs = GridSpec(2, 3, figure=fig, wspace=.40, hspace=.50,
                  left=.115, right=.945, bottom=.085, top=.94)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(feat.minimum_fraction * 100, bins=np.arange(0, 101, 4),
            color=LIGHT, edgecolor="white", lw=.4)
    ax.axvline(5, color=CORAL, ls="--", lw=.8)
    ax.text(9, ax.get_ylim()[1] * .92, f"封底阈值 5%\n{int((feat.minimum_fraction>=.05).sum())} 列存在地板值",
            fontsize=5.6, color=CORAL, va="top", linespacing=1.4)
    ax.set_xlabel("取到该列最小值的样本占比（%）"); ax.set_ylabel("化合物个数")
    ax.grid(axis="y"); ax.set_axisbelow(True); panel(ax, "a", "左删失（地板值）结构")

    xl = pd.ExcelFile(OFFICIAL / "8.xlsx")
    cal = xl.parse("Calibrated chemical dataset").set_index("beer")
    log_cols = [c for c in tra.loc[tra["transform"] == "log10", "feature"]
                if c in raw.columns and c in cal.columns]
    xs, ys = [], []
    for c in log_cols:
        x = pd.to_numeric(raw[c], errors="coerce"); y = pd.to_numeric(cal[c], errors="coerce")
        m = x.notna() & y.notna() & (y > 0); xs.append(x[m]); ys.append(np.log10(y[m]))
    X, Y = pd.concat(xs), pd.concat(ys)
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(X, Y, s=1.4, color=BLUE, alpha=.55, linewidths=0, rasterized=True)
    lim = np.array([-4.5, 4.2])
    ax.plot(lim, lim, color=CORAL, lw=1.0, label="y = x（底数 10）")
    ax.plot(lim, 0.4343 * lim, color=GRAY, lw=.8, ls="--", label="y = 0.4343x（底数 e）")
    ax.text(.03, .96, f"斜率 1.0000，$R^2$={np.corrcoef(X,Y)[0,1]**2:.4f}\nn={len(X)} 点，37 列为 log10",
            transform=ax.transAxes, fontsize=5.6, va="top", color=INK, linespacing=1.4)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("1.csv 数值 $v$（已判定为对数变换列）"); ax.set_ylabel(r"$\log_{10}$ 校准浓度")
    ax.legend(loc="lower right", handlelength=1.5, borderpad=.15)
    ax.grid(True, alpha=.45); ax.set_axisbelow(True); panel(ax, "b", "对数底数反证")

    ax = fig.add_subplot(gs[0, 2])
    fc = mod["family"].to_dict(); cnt = mod["family"].value_counts()
    keep = [f for f in cnt.index if cnt[f] >= 8][:8]
    data, labs = [], []
    for f in keep:
        cols = [c for c in mod.index if fc.get(c) == f and c in raw.columns]
        if cols:
            data.append(raw[cols].median(axis=0).values); labs.append(f"{f}({len(cols)})")
    parts = ax.violinplot(data, showextrema=False, widths=.82)
    for pc, col in zip(parts["bodies"], [BLUE, ORANGE, TEAL, "#B69B70", "#8A839F", CORAL, GRAY, "#7F99C4"]):
        pc.set_facecolor(col); pc.set_alpha(.65); pc.set_edgecolor("white"); pc.set_linewidth(.4)
    ax.set_xticks(range(1, len(labs) + 1), labs, rotation=38, ha="right", fontsize=5.4)
    ax.set_ylabel("各变量中位丰度（1.csv 原值）"); ax.grid(axis="y"); ax.set_axisbelow(True)
    panel(ax, "c", "化学类别的丰度分布")

    ax = fig.add_subplot(gs[1, 0])
    bars = ax.barh(np.arange(len(styles)), styles.n,
                   color=[TEAL if v >= 8 else LIGHT for v in styles.n], height=.68)
    ax.set_yticks(np.arange(len(styles)), styles["style"], fontsize=4.8); ax.invert_yaxis()
    ax.axvline(8, color=CORAL, ls="--", lw=.8)
    ax.text(9.2, 2.2, "n≥8 可参与\n生成评估", fontsize=5.2, color=CORAL, va="top", linespacing=1.4)
    for b_, v in zip(bars, styles.n):
        ax.text(v + .4, b_.get_y() + b_.get_height() / 2, str(int(v)), va="center", fontsize=5.0)
    ax.set_xlabel("样本数（共 250 款 / 22 风格）"); ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0); panel(ax, "d", "22 种风格的样本量")

    ax = fig.add_subplot(gs[1, 1])
    cols = [c for c in tgt.columns if c not in ("beer", "raw_beer_name")]
    mouth = ["body", "bitternes", "co2", "aftertaste", "alcohol", "sweetness", "acidity"]
    fault = ["dms", "metallic", "4vg", "t2n", "diacetyl", "stale_hops"]
    spice = ["orange", "coriander", "clove", "lactic", "acetic", "barnyard"]

    def grp(c):
        if c.startswith("A_"): return "Aroma"
        if c.startswith("F_"): return "Flavour"
        if c.lower() in mouth: return "Mouthfeel"
        if c.lower() in fault or c.lower() == "x4vg": return "Fault"
        if c.lower() in spice: return "Spice/野生"
        return "Overall"

    groups = ["Aroma", "Flavour", "Mouthfeel", "Fault", "Spice/野生", "Overall"]
    gdata = [[pd.to_numeric(tgt[c], errors="coerce").dropna().values for c in cols if grp(c) == g]
             for g in groups]
    flat = [np.concatenate(d) if d else np.array([0.0]) for d in gdata]
    bp = ax.boxplot(flat, widths=.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", lw=.9), whiskerprops=dict(color=MUTED, lw=.5),
                    capprops=dict(color=MUTED, lw=.5), boxprops=dict(lw=.4))
    for patch, col in zip(bp["boxes"], [BLUE, TEAL, ORANGE, CORAL, "#8A839F", GRAY]):
        patch.set_facecolor(col); patch.set_edgecolor("white")
    ax.set_xticks(range(1, len(groups) + 1), [f"{g}({len(d)})" for g, d in zip(groups, gdata)],
                  rotation=26, ha="right", fontsize=5.4)
    ax.set_ylabel("评分员等权均分（0–6）"); ax.set_ylim(-.2, 6.4)
    ax.grid(axis="y"); ax.set_axisbelow(True); panel(ax, "e", "50 维感官目标分布")

    ax = fig.add_subplot(gs[1, 2])
    c5 = pd.read_csv(OFFICIAL / "5.csv", index_col=0)
    nm = xl.parse("Sensory names")
    to_short = dict(zip(nm["CleanName_PanelFeedback"].astype(str), nm["R_name"].astype(str)))
    keep_idx = [k for k, i in enumerate(c5.index)
                if to_short.get(str(i).replace("\r\n", " ")) is not None]
    M = c5.iloc[keep_idx, keep_idx].values.astype(float)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="none")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(.5, -.045, f"平均 |ρ| = {np.nanmean(np.abs(M - np.eye(len(M)) )):.2f}   n={len(M)} 维",
            transform=ax.transAxes, fontsize=5.4, ha="center", va="top", color=INK)
    cb = fig.colorbar(im, ax=ax, shrink=.72, pad=.02)
    cb.set_label("Spearman ρ", fontsize=5.6); cb.ax.tick_params(labelsize=5.0, width=.4)
    panel(ax, "f", "50 维感官相关结构")
    save(fig, "fig02_数据审计六联")


# --------------------------------------------- 图3 风格与地板值（正文用）
def fig03():
    feat = R("audit/features.csv").sort_values("minimum_fraction", ascending=False).head(14)
    styles = R("audit/styles.csv").sort_values("n")
    fig = plt.figure(figsize=(FIG_W * CM, 6.8 * CM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 1], wspace=.42,
                  left=.105, right=.975, bottom=.15, top=.90)
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.barh(np.arange(len(styles)), styles.n,
                   color=[TEAL if s in ("Tripel", "Kriek", "Lambic", "Faro") else "#98ABBB"
                          for s in styles["style"]], height=.68)
    ax.set_yticks(np.arange(len(styles)), styles["style"], fontsize=5.6)
    ax.set_xlim(0, styles.n.max() * 1.18)
    ax.axvline(5, color=CORAL, ls="--", lw=.8)
    for b_, v in zip(bars, styles.n):
        ax.text(v + .35, b_.get_y() + b_.get_height() / 2, str(int(v)), va="center", fontsize=5.4)
    ax.set_xlabel("样本数量（款）"); ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0); panel(ax, "a", "22 种风格的样本量（红虚线=5 款）")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(np.arange(len(feat)), feat.minimum_fraction * 100,
             color=[CORAL if v >= .7 else BLUE for v in feat.minimum_fraction], height=.66)
    ax2.invert_yaxis()
    ax2.set_yticks(np.arange(len(feat)), feat.feature, fontsize=5.4)
    ax2.set_xlim(0, 104)
    for i, r in enumerate(feat.itertuples()):
        ax2.text(r.minimum_fraction * 100 + 1.2, i, f"{r.minimum_fraction:.1%}", va="center", fontsize=5.2)
    ax2.set_xlabel("记录为该变量最小值的样本占比（%）"); ax2.grid(axis="x"); ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", length=0); panel(ax2, "b", "高频地板值变量（前 14）")
    save(fig, "fig03_风格与地板值")


# ------------------------------------------- 图4 模型横向比较
def fig04():
    d = R("q1/repeat_metrics.csv"); pt = R("q1/per_target.csv")
    q = d.groupby("model").MAE.agg(["mean", "median", "min", "max"]).sort_values("mean")
    pv = pt.pivot(index="target", columns="model", values="R2")
    fig = plt.figure(figsize=(FIG_W * CM, 6.8 * CM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1], wspace=.34,
                  left=.115, right=.975, bottom=.16, top=.90)

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(q))
    for i, (_, r) in enumerate(q.iterrows()):
        ax.plot([r["min"], r["max"]], [i, i], color="#DCE3E7", lw=2.4,
                solid_capstyle="round", zorder=1)
    ax.scatter(q["mean"], y, s=15, color=TEAL, edgecolor="white", lw=.5, zorder=3, label="均值")
    ax.scatter(q["median"], y, s=8, facecolor="white", edgecolor=INK, lw=.6, zorder=4, label="中位数")
    base = d[d.model == "Mean"].MAE.mean()
    ax.axvline(base, color=CORAL, ls=":", lw=.8, zorder=0)
    ax.text(base - .003, -1.05, "均值基线", color=CORAL, fontsize=5.2, ha="right", va="top")
    ax.set_yticks(y, q.index); ax.set_ylim(-1.6, len(q) - .4)
    ax.set_xlabel("50 维感官 MAE（分，越小越好）")
    ax.grid(axis="x"); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(.02, .06), handletextpad=.4,
              borderpad=.2, labelspacing=.3, ncol=1, fontsize=5.6)
    panel(ax, "a", "各模型两轮交叉验证的 MAE")

    ax = fig.add_subplot(gs[0, 1])
    order = pv["ET"].sort_values().index
    best = pv[["RF", "HGB", "HierarchyRater", "PLS", "Ridge", "MLP"]].max(axis=1).reindex(order)
    ax.plot(np.arange(len(order)), best, color=GRAY, lw=.9, label="最优单模型")
    ax.scatter(np.arange(len(order)), pv["ET"].reindex(order), s=7, color=TEAL, lw=0, label="ExtraTrees")
    wins = np.where((pv["ET"].reindex(order) - best).values > 0)[0]
    for i in wins:
        ax.plot([i, i], [best.iloc[i], pv["ET"].reindex(order).iloc[i]], color=CORAL, lw=.6, alpha=.85)
    ax.axhline(0, color=MUTED, lw=.6)
    ax.set_xlim(-1, len(order)); ax.set_xticks([0, 10, 20, 30, 40, 49])
    ax.set_xlabel("50 个感官维度（按 ET 的 $R^2$ 升序）"); ax.set_ylabel("逐维 $R^2$")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.text(.03, .06, f"ET 优于最优单模型：{len(wins)}/50 维\n（红线=正增益）",
            transform=ax.transAxes, fontsize=5.4, color=CORAL, linespacing=1.4)
    ax.legend(loc="upper left", handletextpad=.5, borderpad=.2, labelspacing=.35)
    panel(ax, "b", "逐维 $R^2$：集成相对最优单模型的净增益")
    save(fig, "fig04_模型横向比较")


# --------------------------------- 图5 逐维 R² + 难度—区分度
def fig05():
    pt = R("q1/per_target.csv"); tgt = R("audit/equal_rater_targets.csv")
    pv = pt.pivot(index="target", columns="model", values="R2")
    sd = tgt.drop(columns=[c for c in ("beer", "raw_beer_name") if c in tgt.columns]).std()
    r2 = pv["ET"].reindex(sd.index).dropna()
    sd = sd.reindex(r2.index)
    from scipy.stats import spearmanr
    rho = spearmanr(r2, sd).statistic
    fig = plt.figure(figsize=(FIG_W * CM, 7.2 * CM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1], wspace=.30,
                  left=.16, right=.975, bottom=.16, top=.90)
    ax = fig.add_subplot(gs[0, 0])
    order = r2.sort_values().index
    y = np.arange(len(order))
    ax.scatter(r2.reindex(order), y, s=6, color=TEAL, lw=0, zorder=3)
    ax.axvline(.21, color=CORAL, ls="--", lw=.8)
    ax.text(.215, len(order) - 2, "文献 GBR\n$R^2$=0.21", fontsize=5.2, color=CORAL, va="top", linespacing=1.4)
    ax.set_yticks(y[::3], order[::3], fontsize=4.6)
    ax.set_ylim(-1, len(order)); ax.set_xlabel("逐维 $R^2$（ExtraTrees）")
    ax.grid(axis="x"); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    panel(ax, "a", "50 个感官维度的 $R^2$")

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(sd, r2, s=9, color=BLUE, lw=0, alpha=.85)
    z = np.polyfit(sd, r2, 1); xx = np.linspace(sd.min(), sd.max(), 50)
    ax.plot(xx, np.polyval(z, xx), color=CORAL, ls="--", lw=.8)
    ax.text(.04, .95, f"ρ = {rho:.2f}\n标签区分度越高越可预测", transform=ax.transAxes,
            fontsize=5.6, va="top", color=INK, linespacing=1.4)
    ax.set_xlabel("该维度标签标准差（区分度）"); ax.set_ylabel("$R^2$")
    ax.grid(True, alpha=.5); ax.set_axisbelow(True)
    panel(ax, "b", "难度与区分度")
    save(fig, "fig05_逐维R2")


# ------------------------------------------ 图6 预测—实测六联
def fig06():
    oof = R("q1/continuous_oof.csv").drop(columns=[c for c in ("beer", "raw_beer_name") if c in R("q1/continuous_oof.csv").columns])
    tgt = R("audit/equal_rater_targets.csv")
    tgt = tgt.drop(columns=[c for c in ("beer", "raw_beer_name") if c in tgt.columns])
    cols = [c for c in oof.columns if c in tgt.columns]
    X = tgt[cols].to_numpy(float); Y = oof[cols].to_numpy(float)
    pt = R("q1/per_target.csv")
    r2 = pt[pt.model == "ET"].set_index("target").R2.reindex(cols)
    order = r2.sort_values(ascending=False).index.tolist()
    pick = [order[0], order[len(order)//4], order[len(order)//2],
            order[3*len(order)//4], order[-3], order[-1]]
    zh = {"overall": "总体喜好", "body": "酒体", "alcohol": "醇感", "bitternes": "苦度",
          "sweetness": "甜度", "acidity": "酸度", "aftertaste": "回味", "co2": "碳酸感",
          "dms": "DMS 蒸煮蔬菜", "diacetyl": "双乙酰奶油", "metallic": "金属味",
          "4vg": "4-VG 丁香", "t2n": "反式-2-壬烯醛纸板", "stale_hops": "陈酒花",
          "barnyard": "马厩 Brett", "clove": "丁香酚", "orange": "橙香",
          "coriander": "芫荽", "lactic": "乳酸", "acetic": "乙酸"}

    def label(c):
        if c in zh: return zh[c]
        pre = ("麦芽香" if c.startswith("A_malt") else "麦芽味" if c.startswith("F_malt")
               else "酒花香" if c.startswith("A_hops") else "酒花味" if c.startswith("F_hops")
               else "酯香" if c.startswith("A_est") else "酯味")
        tail = {"all": "总体", "grain": "谷物", "bread": "面包", "cara": "焦糖",
                "burn": "烟熏焦", "citrus": "柑橘", "tropical": "热带水果",
                "noble": "草本贵族", "woody": "木质树脂", "ethac": "乙酸乙酯",
                "isoaa": "乙酸异戊酯", "flower": "花香", "fruity": "果香"}.get(c.split("_")[-1], c)
        return f"{pre}-{tail}"

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W * CM, 9.6 * CM),
                             gridspec_kw=dict(wspace=.30, hspace=.40, left=.075,
                                              right=.975, bottom=.085, top=.93))
    for i, (c, ax) in enumerate(zip(pick, axes.flat)):
        x = X[:, cols.index(c)]; y = Y[:, cols.index(c)]
        m = np.isfinite(x) & np.isfinite(y)
        hb = ax.hexbin(x[m], y[m], gridsize=26, mincnt=1, cmap="YlGnBu", bins="log",
                       extent=(-.1, 6.1, -.1, 6.1), linewidths=0)
        ax.plot([0, 6], [0, 6], color=CORAL, ls="--", lw=.8)
        rho = pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman")
        ax.text(.03, .955, f"{label(c)}\n$R^2$={r2[c]:.2f}   ρ={rho:.2f}",
                transform=ax.transAxes, fontsize=5.5, va="top", color=INK, linespacing=1.5)
        ax.set_xlim(-.1, 6.1); ax.set_ylim(-.1, 6.1); ax.set_aspect("equal")
        ax.set_xlabel("评分员等权参考均分")
        if i % 3 == 0: ax.set_ylabel("折外模型预测")
        ax.grid(True, alpha=.4); ax.set_axisbelow(True)
        cb = fig.colorbar(hb, ax=ax, shrink=.62, pad=.02)
        cb.ax.tick_params(labelsize=4.6, width=.35); cb.set_label("组合频次(log)", fontsize=5.0)
        panel(ax, chr(97 + i), label(c), pad=3)
    save(fig, "fig06_预测实测六联")


# ---------------------------------------- 图7 敏感性/稳健性
def fig07():
    s = R("q1/sensitivity.csv"); tgt = R("audit/equal_rater_targets.csv")
    oof = R("q1/continuous_oof.csv")
    oof = oof.drop(columns=[c for c in ("beer", "raw_beer_name") if c in oof.columns])
    tgt = tgt.drop(columns=[c for c in ("beer", "raw_beer_name") if c in tgt.columns])
    cols = [c for c in oof.columns if c in tgt.columns]
    resid = (oof[cols].to_numpy(float) - tgt[cols].to_numpy(float)).ravel()
    resid = resid[np.isfinite(resid)]

    fig = plt.figure(figsize=(FIG_W * CM, 8.6 * CM))
    gs = GridSpec(2, 2, figure=fig, wspace=.34, hspace=.56,
                  left=.135, right=.975, bottom=.11, top=.90)

    ax = fig.add_subplot(gs[0, 0])
    d = s[s.experiment.str.contains("mask")].copy()
    d["ratio"] = pd.to_numeric(d.experiment.str.extract(r"mask_([0-9.]+)")[0])
    agg = d.groupby("ratio").MAE.agg(["mean", "std"])
    base = s[s.experiment == "ET_equal_rater"].MAE.mean()
    x = np.r_[0, agg.index.values] * 100; y = np.r_[base, agg["mean"].values]
    ax.plot(x, y, color=TEAL, lw=1.2, marker="o", ms=3)
    ax.fill_between(agg.index.values * 100, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                    color=TEAL, alpha=.12)
    for xx, yy in zip(x, y):
        ax.text(xx, yy + .0012, f"{yy:.4f}", ha="center", fontsize=5.2, color=INK)
    ax.set_xlabel("额外随机遮蔽化学特征比例（%）"); ax.set_ylabel("MAE（分）"); ax.set_xticks(x)
    ax.grid(True, alpha=.5); ax.set_axisbelow(True); panel(ax, "a", "随机特征遮蔽敏感性", pad=4)

    ax = fig.add_subplot(gs[0, 1])
    ex = s.groupby("experiment").MAE.mean().sort_values()
    bars = ax.barh(np.arange(len(ex)), ex.values,
                   color=[TEAL if k == "ET_equal_rater" else "#8FA6B4" for k in ex.index], height=.62)
    ax.set_yticks(np.arange(len(ex)), [(k.replace("ET_", "")
                                        .replace("record_training_equal_eval", "record_train")
                                        .replace("official_Z_scale", "official_Z")) for k in ex.index],
                  fontsize=4.9)
    ax.invert_yaxis(); ax.set_xlim(0, ex.values.max() * 1.22)
    for b_, v in zip(bars, ex.values):
        ax.text(v + .004, b_.get_y() + b_.get_height() / 2, f"{v:.4f}", va="center", fontsize=5.0)
    ax.set_xlabel("MAE（分）"); ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0); panel(ax, "b", "评分口径敏感性", pad=4)

    ax = fig.add_subplot(gs[1, 0])
    lo, hi = np.percentile(resid, [1, 99])
    ax.hist(resid[(resid >= lo) & (resid <= hi)], bins=55, color=BLUE, edgecolor="white", lw=.3)
    ax.axvline(np.percentile(resid, 5), color=CORAL, ls="--", lw=.7)
    ax.axvline(np.percentile(resid, 95), color=CORAL, ls="--", lw=.7)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("折外残差（0–6 分，横轴=1%–99% 分位）"); ax.set_ylabel("样本数")
    ax.text(.03, .95, f"中位 {np.median(resid):.3f}｜MAE {np.mean(np.abs(resid)):.4f}\n"
                      f"90% 区间宽 {np.percentile(resid,95)-np.percentile(resid,5):.3f}"
                      f"（尾部 {np.sum((resid<lo)|(resid>hi))} 个样本未显示）",
            transform=ax.transAxes, fontsize=5.2, va="top", color=INK, linespacing=1.45)
    ax.grid(axis="y"); ax.set_axisbelow(True); panel(ax, "c", "折外残差分布", pad=4)

    ax = fig.add_subplot(gs[1, 1])
    pv = R("q1/per_target.csv").pivot(index="target", columns="model", values="MAE")
    mods = ["Mean", "PLS", "Ridge", "HGB", "RF", "ET"]
    for mm in mods:
        if mm in pv.columns:
            v = pv[mm].dropna().values
            ax.plot([v.min(), v.max()], [mods.index(mm)] * 2, color="#DCE3E7", lw=2.2, zorder=1)
            ax.scatter(np.median(v), mods.index(mm), s=10, color=COL_M.get(mm, GRAY),
                       edgecolor="white", lw=.4, zorder=3)
    ax.set_yticks(range(len(mods)), mods, fontsize=5.4)
    ax.set_xlabel("逐维 MAE 中位数（水平线=50 维极差）")
    ax.grid(axis="x"); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    panel(ax, "d", "误差在 50 维上的离散度", pad=4)
    save(fig, "fig07_敏感性与稳健性")


# ------------------------------------------------- 图8 共表达网络
def fig08():
    import networkx as nx
    names = pd.read_csv(SUP / "data/processed/chemical_names.csv")
    to_feat = dict(zip(names.matrix_name, names.feature))
    mod = pd.read_csv(HERE / "改版样张" / "q2_modules.csv").set_index("compound")
    raw2 = pd.read_csv(OFFICIAL / "2.csv", index_col=0)
    M = raw2.values.astype(float); np.fill_diagonal(M, 0.0)
    labels = [to_feat.get(c, c) for c in raw2.index]
    thr = 0.5; keep = np.abs(M) >= thr; np.fill_diagonal(keep, False)
    G = nx.Graph(); G.add_nodes_from(labels)
    for a, b in np.argwhere(np.triu(keep)):
        G.add_edge(labels[a], labels[b], w=abs(M[a, b]))
    module_of = mod["module"].to_dict()
    main = [m for m in sorted(mod.module.unique()) if (mod.module == m).sum() >= 10]
    cm = {m: c for m, c in zip(main, [CORAL, BLUE, TEAL, ORANGE, "#8A839F", "#A9BBAA"])}

    fig = plt.figure(figsize=(FIG_W * CM, 7.8 * CM))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.50, 1.05, .62], wspace=.34,
                  left=.01, right=.985, bottom=.13, top=.88)
    ax = fig.add_subplot(gs[0, 0])
    pos = nx.spring_layout(G, seed=2026, weight="w", iterations=120, k=.42)
    deg = dict(G.degree())
    pe = [(u, v) for u, v, _ in G.edges(data=True) if M[labels.index(u), labels.index(v)] > 0]
    ne = [(u, v) for u, v, _ in G.edges(data=True) if M[labels.index(u), labels.index(v)] < 0]
    nx.draw_networkx_edges(G, pos, edgelist=pe, edge_color="#AFC7C3", width=.28, alpha=.55, ax=ax)
    if ne:
        nx.draw_networkx_edges(G, pos, edgelist=ne, edge_color=CORAL, width=.55, style="dashed", ax=ax)
    cols_ = [cm.get(module_of.get(n), "#C9D2D8") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=[5 + 2.0 * deg[n] for n in G.nodes()],
                           node_color=cols_, edgecolors="white", linewidths=.3, ax=ax)
    for k, n in enumerate(sorted(G.nodes, key=lambda x: deg[x], reverse=True)[:4]):
        ax.annotate(n.replace("_", " ")[:16], pos[n],
                    xytext=(8 if k % 2 == 0 else -8, 10 + 6 * (k // 2)), textcoords="offset points",
                    fontsize=4.8, color=INK, ha="left" if k % 2 == 0 else "right",
                    arrowprops=dict(arrowstyle="-", lw=.35, color="#9AA7B0", shrinkA=0, shrinkB=1.5))
    ax.axis("off")
    ax.set_title("(a)  共表达网络（力导向布局）", loc="left", color=NAVY, fontsize=7.6, pad=4)
    ax.text(0, -.02, f"节点 {len(G)}，边 {G.number_of_edges()}（|ρ|≥{thr}）\n颜色=共表达模块，面积=连接度",
            transform=ax.transAxes, fontsize=5.6, color=MUTED, va="top")

    ax2 = fig.add_subplot(gs[0, 1])
    comp = (mod[mod.module.isin(main)].assign(one=1)
            .pivot_table(index="module", columns="family", values="one", aggfunc="sum")
            .fillna(0).reindex(main))
    fams = [f for f in ["酯类", "萜烯类", "醇类", "醛类", "酮类", "硫化物", "理化参数", "呋喃类"]
            if f in comp.columns]
    fc = dict(zip(fams, [BLUE, ORANGE, TEAL, "#B69B70", "#8A839F", CORAL, GRAY, "#7F99C4"]))
    bottom = np.zeros(len(comp))
    for f in fams:
        ax2.bar(np.arange(len(comp)), comp[f], bottom=bottom, color=fc[f], width=.72,
                label=f, lw=.4, edgecolor="white")
        bottom += comp[f].values
    ax2.set_xticks(np.arange(len(comp)), [f"M{m}" for m in main])
    ax2.set_ylabel("化合物个数"); ax2.set_xlabel("共表达模块")
    ax2.grid(axis="y"); ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", ncol=2, handlelength=.9, handletextpad=.4,
               columnspacing=.8, borderpad=.05, labelspacing=.2, fontsize=5.8)
    panel(ax2, "b", "各模块的化学类别构成")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.barh(np.arange(len(comp)), comp.sum(axis=1).values, color=[cm[m] for m in main], height=.62)
    for i, v in enumerate(comp.sum(axis=1).values):
        ax3.text(v + .6, i, str(int(v)), va="center", fontsize=6)
    ax3.set_yticks(np.arange(len(comp)), [f"M{m}" for m in main]); ax3.invert_yaxis()
    ax3.set_xlabel("模块规模"); ax3.grid(axis="x"); ax3.set_axisbelow(True)
    panel(ax3, "c", "模块大小")
    save(fig, "fig08_共表达网络")


# ------------------------------- 图9 四规则性能/稳定/冗余
def fig09():
    d = R("closed/q2/summary_metrics.csv").set_index("method")
    st = R("closed/q2/stability_summary.csv").set_index("method")
    order = ["Spearman", "Hybrid2plus1", "Forward3", "Redundancy", "TrainSelected"]
    fig, ax = plt.subplots(1, 3, figsize=(FIG_W * CM, 5.6 * CM),
                           gridspec_kw=dict(width_ratios=[1.1, 1.05, .95], wspace=.34,
                                            left=.085, right=.975, bottom=.26, top=.90))
    x = np.arange(len(order)); c = [COL_Q2[z] for z in order]
    v = d.loc[order, "MAE_mean"].values; sd = d.loc[order, "MAE_std"].values
    bar = ax[0].bar(x, v, yerr=sd, capsize=2, color=c, width=.65, error_kw={"lw": .7})
    ax[0].set_xticks(x, [LAB_Q2[m] for m in order], rotation=32, ha="right", fontsize=5.6)
    ax[0].set_ylabel("50 维 MAE（分）"); ax[0].set_ylim(0, v.max() * 1.20)
    ax[0].grid(axis="y"); ax[0].set_axisbelow(True)
    for r_, h in zip(bar, v):
        ax[0].annotate(f"{h:.3f}", (r_.get_x() + r_.get_width() / 2, r_.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha="center", fontsize=5.4)
    panel(ax[0], "a", "三特征预测性能")

    bw = .36
    ax[1].bar(x - bw / 2, st.loc[order, "compound_jaccard"], bw, color=BLUE, label="具体化合物")
    ax[1].bar(x + bw / 2, st.loc[order, "group_jaccard"], bw, color=TEAL, label="替代组")
    ax[1].set_xticks(x, [LAB_Q2[m] for m in order], rotation=32, ha="right", fontsize=5.6)
    ax[1].set_ylabel("Bootstrap Jaccard"); ax[1].grid(axis="y"); ax[1].set_axisbelow(True)
    ax[1].legend(loc="upper right", fontsize=5.8, handletextpad=.4, borderpad=.1)
    panel(ax[1], "b", "排名稳定性")

    bar2 = ax[2].bar(x, st.loc[order, "within_top3_abs_rho"], color=c, width=.63)
    ax[2].set_xticks(x, [LAB_Q2[m] for m in order], rotation=32, ha="right", fontsize=5.6)
    ax[2].set_ylabel("Top-3 内部平均绝对相关"); ax[2].grid(axis="y"); ax[2].set_axisbelow(True)
    for r_, h in zip(bar2, st.loc[order, "within_top3_abs_rho"].values):
        ax[2].annotate(f"{h:.2f}", (r_.get_x() + r_.get_width() / 2, r_.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha="center", fontsize=5.4)
    panel(ax[2], "c", "入选化合物信息冗余")
    save(fig, "fig09_Q2四规则")


# ------------------------------------------- 图10 阈值—OAV
def fig10():
    raw = pd.read_csv(RAW_1, low_memory=False).set_index("beer")
    tra = pd.read_csv(SUP / "data/processed/transforms.csv")
    th = pd.read_csv(SUP / "data/processed/thresholds.csv")
    m = tra.merge(th[["field_name_python", "chemical_name", "unit", "threshold_mean", "origin_paper"]],
                  left_on="feature", right_on="field_name_python", how="inner", suffixes=("", "_thr"))
    m = m[m.threshold_mean.notna() & m.unit_thr.notna()]
    m = m[~m.feature.isin(["protein.g.L.", "ethanol..v.v.", "bitterness", "GFS", "glycerol", "iron"])]
    rows = []
    for r in m.itertuples():
        v = pd.to_numeric(raw[r.feature], errors="coerce").dropna()
        c = (10 ** v) if r.transform == "log10" else v
        oav = c / r.threshold_mean
        rows.append((r.chemical_name, r.unit_thr, r.origin_paper, r.threshold_mean,
                     oav.median(), float((oav > 1).mean())))
    t = pd.DataFrame(rows, columns=["chem", "unit", "origin", "thr", "oav_med", "frac_gt1"])
    t = t.sort_values("oav_med", ascending=False).head(12).iloc[::-1]
    t["short"] = [c.replace("Dimethyl sulfide", "DMS").replace("Carbon disulfide", "CS2")
                   .replace("Hydrogen sulfide", "H2S").replace("Diethyl sulfide", "DES")
                   .replace("Linalool", "里那醇 Linalool").replace("Citronellol", "香茅醇 Citronellol")
                   .replace("Geraniol", "香叶醇 Geraniol").replace("4-vinylguaiacol", "4-VG")
                   .replace("Isobutanol", "异丁醇").replace("Acetaldehyde", "乙醛")
                   .replace("Methylpyrazine", "甲基吡嗪").replace("Furfural", "糠醛") for c in t.chem]

    fig = plt.figure(figsize=(FIG_W * CM, 6.0 * CM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=.42,
                  left=.115, right=.975, bottom=.20, top=.88)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(t.thr, t.oav_med, s=20, color=[CORAL if f > .5 else BLUE for f in t.frac_gt1],
               edgecolor="white", lw=.5, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(.01, 100); ax.set_ylim(1e-4, 3e1)
    OFFS = {"里那醇 Linalool": (6, 2, "left"), "香茅醇 Citronellol": (6, -10, "left"),
            "香叶醇 Geraniol": (12, 2, "left"), "DMS": (6, 4, "left"),
            "CS2": (6, -13, "left"), "H2S": (6, 4, "left"),
            "DES": (-6, 8, "right"), "4-VG": (6, 0, "left"),
            "异丁醇": (6, 2, "left"), "乙醛": (-6, 6, "right"),
            "甲基吡嗪": (6, 0, "left"), "糠醛": (6, 6, "left")}
    for r in t.itertuples():
        dx, dy, ha = OFFS.get(r.short, (5, 3, "left"))
        ax.annotate(r.short, (r.thr, r.oav_med), xytext=(dx, dy), textcoords="offset points",
                    fontsize=5.4, color=INK, ha=ha)
    ax.axhline(1, color=CORAL, ls="--", lw=.7)
    ax.text(.012, .30, "OAV = 1\n可感知阈值", ha="left", va="top", fontsize=5.6,
            color=CORAL, linespacing=1.4)
    ax.set_xlabel("气味阈值 $T_c$（mg·L$^{-1}$ / μg·L$^{-1}$）"); ax.set_ylabel("OAV 中位数（对数轴）")
    ax.grid(True, which="major", alpha=.5); ax.set_axisbelow(True); panel(ax, "a", "阈值—OAV 关系")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(np.arange(len(t)), t.frac_gt1 * 100,
             color=[CORAL if f > .5 else BLUE for f in t.frac_gt1], height=.62)
    ax2.set_yticks(np.arange(len(t)), t["short"], fontsize=5.6)
    for i, v in enumerate(t.frac_gt1 * 100):
        ax2.text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=5.6)
    ax2.set_xlim(0, 112); ax2.set_xlabel("样品中 OAV > 1 的比例（%）")
    ax2.grid(axis="x"); ax2.set_axisbelow(True); ax2.tick_params(axis="y", length=0)
    panel(ax2, "b", "超阈样品占比（红=过半样品）")
    save(fig, "fig10_阈值OAV")


# ------------------------------ 图11 生成模型六维对照
def fig11():
    d = R("closed/generation/macro_summary.csv").set_index("model")
    models = ["PCA_Gaussian", "CopulaIndependent", "CopulaJoint", "CVAE"]
    plots = [("SWD_mean", "SWD_std", "逐风格 SWD ↓"),
             ("style_accuracy_mean", "style_accuracy_std", "风格分类一致率 ↑"),
             ("joint_floor_MAE_mean", "joint_floor_MAE_std", "联合低值概率误差 ↓"),
             ("diversity_mean", "diversity_std", "生成多样性 ↑"),
             ("correlation_MAE_mean", "correlation_MAE_std", "相关结构误差 ↓"),
             ("floor_mass_MAE_mean", "floor_mass_MAE_std", "单变量最低值误差 ↓")]
    fig, ax = plt.subplots(2, 3, figsize=(FIG_W * CM, 9.2 * CM),
                           gridspec_kw=dict(wspace=.36, hspace=.52, left=.085,
                                            right=.975, bottom=.135, top=.915))
    short = {"PCA_Gaussian": "PCA-高斯", "CopulaIndependent": "独立\nCopula",
             "CopulaJoint": "联合\nCopula", "CVAE": "CVAE"}
    for i, (col, err, title) in enumerate(plots):
        a = ax.flat[i]
        v = d.loc[models, col]; e = d.loc[models, err].fillna(0)
        rects = a.bar(range(4), v.values, yerr=e.values, color=[COL_G[z] for z in models],
                      width=.62, capsize=1.8, error_kw={"lw": .7})
        a.set_xticks(range(4), [short[z] for z in models], fontsize=5.4, linespacing=1.3)
        a.set_ylim(0, (v + e).max() * 1.24)
        a.grid(axis="y"); a.set_axisbelow(True)
        for x0, h, ee in zip(range(4), v.values, e.values):
            a.annotate(f"{h:.3f}", (x0, h + ee), xytext=(0, 2),
                       textcoords="offset points", ha="center", fontsize=4.8)
        panel(a, chr(97 + i), title, pad=3)
    save(fig, "fig11_生成六维对照")


# ------------------------------ 图12 三风格感官画像
def fig12():
    z = R("q3/generated_sensory50.csv")
    obs = R("audit/equal_rater_targets.csv")
    raw = pd.read_csv(RAW_1, low_memory=False)
    if len(obs) != len(raw):
        raise ValueError("原始样本与等权目标行数不一致")
    obs = obs.copy(); obs["style"] = raw["tasting_category_fine"].values
    cols = ["A_malt_all", "A_hops_all", "A_esters_all", "F_malt_all", "F_hops_all",
            "F_esters_all", "bitternes", "sweetness", "acidity", "body", "aftertaste", "overall"]
    labs = ["麦芽香", "酒花香", "酯香", "麦芽味", "酒花味", "酯味", "苦度", "甜度",
            "酸度", "酒体", "余味", "总体"]
    styles = ["Tripel", "Kriek", "Lambic"]
    fig, axs = plt.subplots(1, 3, figsize=(FIG_W * CM, 6.4 * CM), sharey=True,
                            gridspec_kw=dict(wspace=.14, left=.075, right=.975,
                                             bottom=.26, top=.86))
    for i, (st, a) in enumerate(zip(styles, axs)):
        p0 = obs[obs["style"] == st][cols].mean(); p1 = z[z["style"] == st][cols].mean()
        x = np.arange(len(cols)); bw = .36
        a.bar(x - bw / 2, p0, bw, color=BLUE, label="历史品评均值")
        a.bar(x + bw / 2, p1, bw, color=TEAL, label="生成谱代理画像")
        a.set_xticks(x, labs, rotation=52, ha="right", fontsize=5.4)
        a.set_ylim(0, 6.15); a.grid(axis="y"); a.set_axisbelow(True)
        panel(a, chr(97 + i), f"{st}（历史 n={len(obs[obs['style']==st])}）", pad=3)
    axs[0].set_ylabel("0–6 分感官强度")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(.5, .005), handletextpad=.5)
    save(fig, "fig12_三风格画像")


# ------------------------ 图13 逐风格支持范围 + 条件 SWD
def fig13():
    cov = R("closed/generation/style_coverage.csv").sort_values("n_total", ascending=False)
    ms = R("closed/generation/metrics_22styles_summary.csv")
    style_order = cov["style"].tolist()
    models = ["PCA_Gaussian", "CopulaIndependent", "CopulaJoint", "CVAE"]
    mat = ms.pivot(index="style", columns="model", values="SWD_mean").reindex(index=style_order, columns=models)
    fig = plt.figure(figsize=(FIG_W * CM, 8.6 * CM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[.72, 1.5], wspace=.26,
                  left=.13, right=.93, bottom=.22, top=.90)
    a = fig.add_subplot(gs[0, 0]); y = np.arange(len(style_order))
    a.barh(y, cov.n_total, color="#97AAB9", height=.7)
    a.barh(y, cov.n_unique_test, color=TEAL, height=.7, label="留出测试样本")
    a.set_yticks(y, style_order, fontsize=5.2); a.invert_yaxis()
    a.set_xlabel("样本量"); a.grid(axis="x"); a.set_axisbelow(True)
    a.tick_params(axis="y", length=0); a.legend(fontsize=5.6, loc="lower right")
    panel(a, "a", "总样本 / 留出样本")

    b = fig.add_subplot(gs[0, 1])
    cmap = plt.colormaps["YlGnBu"].copy(); cmap.set_bad("#DFE4E8")
    vals = mat.values
    im = b.imshow(np.ma.masked_invalid(vals), aspect="auto", cmap=cmap, interpolation="none",
                  vmin=0, vmax=np.nanpercentile(vals, 95))
    b.set_yticks(y, []) ; b.set_xticks(range(4), [LAB_G[m] for m in models],
                                      rotation=28, ha="right", fontsize=5.6)
    b.grid(False)
    b.set_ylim(len(style_order) - .5, -.5)
    cb = fig.colorbar(im, ax=b, shrink=.72, pad=.02)
    cb.set_label("条件 SWD ↓", fontsize=5.6); cb.ax.tick_params(labelsize=5.0, width=.4)
    b.text(.985, -.085, "", transform=b.transAxes)  # 说明移至 (a) 面板
    panel(b, "b", "不同风格的条件 SWD（22 风格 × 4 模型）")
    fig.text(.5, .015, "灰=总样本，青绿=留出测试样本；右图灰格=该风格无留出样本（NA 遮罩，不填 0）",
             ha="center", fontsize=5.0, color=MUTED)
    save(fig, "fig13_逐风格支持范围")


# --------------------- 图14 迁移收益/风险/拒绝率
def fig14():
    d = R("closed/migration/per_beer.csv")
    order = sorted(d.beer.unique())
    pol = ["Unconstrained", "PointConstraint", "UncertaintyConstraint", "Reject"]
    fig = plt.figure(figsize=(FIG_W * CM, 5.6 * CM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[.78, 1.22], wspace=.24,
                  left=.085, right=.975, bottom=.30, top=.90)
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])
    for p in pol:
        t = d[d.policy == p].set_index("beer").reindex(order)
        ax1.plot(range(len(order)), t.target_gain.values, marker="o", ms=2.4, lw=.9,
                 color=COL_T[p], label=LAB_T[p])
        ax2.plot(range(len(order)), t.stale_change.values, marker="o", ms=2.4, lw=.9,
                 color=COL_T[p], label=LAB_T[p])
    for a in (ax1, ax2):
        a.axhline(0, color="#4C5965", lw=.7, ls="--")
        a.set_xlabel("留出源啤酒编号"); a.grid(axis="y"); a.set_axisbelow(True)
        a.set_xticks(range(len(order)), [str(x) for x in order], fontsize=5.4)
    ax1.set_ylabel("目标风格评分变化"); ax2.set_ylabel("独立代理评分变化")
    panel(ax1, "a", "迁移收益"); panel(ax2, "b", "缺陷风险")
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(.5, .02),
               handletextpad=.5, columnspacing=1.4, fontsize=6.0)
    save(fig, "fig14_迁移三策略")

    # 第二张：三面板汇总
    a2 = R("closed/migration/accepted_and_all.csv").set_index("policy")
    metrics = [("all_beer_target_gain", "目标风格增益 ↑"),
               ("all_beer_risk_rate", "独立代理上升比例 ↓"),
               ("rejection_rate", "拒绝率 ↓")]
    fig2, axs = plt.subplots(1, 3, figsize=(FIG_W * CM, 5.0 * CM),
                             gridspec_kw=dict(wspace=.36, left=.075, right=.975,
                                              bottom=.26, top=.90))
    for i, (mm, tt) in enumerate(metrics):
        a = axs[i]; v = a2.loc[pol, mm].values
        rects = a.bar(np.arange(4), v, color=[COL_T[x] for x in pol], width=.66)
        a.set_xticks(range(4), [LAB_T[x] for x in pol], rotation=26, ha="right", fontsize=5.6)
        a.set_ylim(0, max(v.max() * 1.3, .05))
        a.grid(axis="y"); a.set_axisbelow(True)
        for r_, hh in zip(rects, v):
            a.text(r_.get_x() + r_.get_width() / 2, r_.get_height() + max(v.max() * 1.3, .05) * .04,
                   f"{hh:.3f}", ha="center", fontsize=5.2)
        panel(a, chr(97 + i), tt, pad=3)
    save(fig2, "fig14b_迁移收益风险拒绝率")


# --------------------------------- S1 50 维误差热图
def figS1():
    d = R("q1/per_target.csv")
    pv = d[d.model.isin(["ET", "MLP", "HierarchyRater"])].pivot(index="target", columns="model", values="MAE")
    pv = pv.sort_values("ET", ascending=False)
    fig, ax = plt.subplots(figsize=(FIG_W * CM, 10.6 * CM),
                           gridspec_kw=dict(left=.20, right=.86, bottom=.13, top=.92))
    im = ax.imshow(pv.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=np.nanpercentile(pv.values, 97),
                   interpolation="none")
    ax.set_yticks(range(len(pv)), pv.index, fontsize=5.0)
    ax.set_xticks(range(len(pv.columns)), pv.columns, fontsize=6.0)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=.5, pad=.02)
    cb.set_label("MAE（分）", fontsize=6.0); cb.ax.tick_params(labelsize=5.2, width=.4)
    panel(ax, "a", "50 项感官属性预测误差（MAE，按 ET 降序）")
    save(fig, "figS1_50维误差热图")


# ------------------------------ S2 150 候选化合物矩阵
def figS2():
    d = R("closed/q2/top3_50x3.csv").sort_values(["target", "rank"])
    targets = list(d.target.drop_duplicates())
    fig, ax = plt.subplots(figsize=(FIG_W * CM, 11.0 * CM),
                           gridspec_kw=dict(left=.19, right=.86, bottom=.13, top=.92))
    v = d.pivot(index="target", columns="rank", values="bootstrap_frequency").reindex(targets)
    im = ax.imshow(v.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1, interpolation="none")
    ax.set_yticks(range(len(targets)), targets, fontsize=5.0)
    ax.set_xticks(range(3), ["Top-1", "Top-2", "Top-3"], fontsize=6.2)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=.42, pad=.02)
    cb.set_label("100 次 Bootstrap 入选比例", fontsize=5.6); cb.ax.tick_params(labelsize=5.0, width=.4)
    panel(ax, "a", "50 × Top-3 入选频率（化合物名见附录表）")
    save(fig, "figS2_入选频率矩阵")


# --------------------------------- S3 训练曲线
def figS3():
    d = R("closed/generation/training_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * CM, 4.4 * CM),
                             gridspec_kw=dict(wspace=.30, left=.09, right=.975,
                                              bottom=.22, top=.90))
    for a, col, title, colr in [(axes[0], "reconstruction", "训练重构损失", TEAL),
                                (axes[1], "validation_reconstruction", r"验证重构损失（$\log_{10}$）", CORAL)]:
        for seed, g in d.groupby("seed"):
            y = g[col].values
            if col == "validation_reconstruction":
                y = np.log10(np.maximum(y, 1e-6))
            a.plot(g.epoch.values, y, lw=.8, alpha=.85, label=str(seed))
        a.set_xlabel("Epoch"); a.set_ylabel("损失")
        a.grid(True, alpha=.5); a.set_axisbelow(True)
        panel(a, "a" if col == "reconstruction" else "b", title, pad=3)
    axes[1].legend(fontsize=5.2, ncol=3, handlelength=1.2, borderpad=.1, columnspacing=.8)
    axes[0].text(.03, .93, "3 个种子逐曲线；验证损失取 log10 轴\n（单个种子量级异常已如实画出）",
                 transform=axes[0].transAxes, fontsize=5.2, va="top", color=MUTED, linespacing=1.4)
    save(fig, "figS3_训练曲线")


JOBS = [("图1", fig01), ("图2", fig02), ("图3", fig03), ("图4", fig04), ("图5", fig05),
        ("图6", fig06), ("图7", fig07), ("图8", fig08), ("图9", fig09), ("图10", fig10),
        ("图11", fig11), ("图12", fig12), ("图13", fig13), ("图14a", fig14),
        ("S1", figS1), ("S2", figS2), ("S3", figS3)]
if __name__ == "__main__":
    setup()
    print("输出目录:", OUT)
    ok, bad = [], []
    for tag, fn in JOBS:
        try:
            fn(); ok.append(tag)
        except Exception as ex:
            bad.append((tag, f"{type(ex).__name__}: {ex}"))
            print("  ✗", tag, ex)
    print(f"\n完成 {len(ok)} 个任务，失败 {len(bad)} 个")
    for t, e in bad:
        print("  失败:", t, e)
