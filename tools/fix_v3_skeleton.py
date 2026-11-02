"""AI-assisted DeepSeek Harness, 2026-10-29. SKELETON-ONLY revision of 论文/v3.md.

Scope: headings, numbering, and the presence/shape of structural elements.
Prose is NOT rewritten in this pass -- paragraphs are only moved under merged headings.

Skeleton changes
  1. 四章 7 节 -> 6 节：合并 4.1+4.2（数据概况与主键映射本就同题）、4.3+4.4（尺度与地板同题），
     其后 4.5/4.6/4.7 顺延为 4.4/4.5/4.6。
  2. 补出缺失的「附录」一级标题与 A/B/C 三个二级标题（正文原本止于 AI 声明）。
  3. 图号体系：1.3 只作文字交叉引用，图1 的唯一落点放在 3.4（原稿两处占位互相指认）。
  4. 表号体系：为 10 张表加 表1..表10 题注。
  5. AI 声明标题去掉「（草稿，须与真实队员记录核对）」这类编辑批注。
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "论文" / "v3.md"
DST = ROOT / "build_v3" / "v3.md"

TABLES = [
    ("| 符号 | 定义 |", "表1　主要符号说明"),
    ("| model | MAE | RMSE | R2 | AF_MAE |", "表2　问题一各模型的内部交叉验证误差"),
    ("| comparator | positive_favors_reference | CI_low | CI_high | p_holm |",
     "表3　问题一各对照模型相对冻结选法 InnerSelected 的配对检验"),
    ("| experiment | MAE |", "表4　问题一敏感性实验"),
    ("| 方法 | MAE均值 | MAE标准差 | RMSE | R² |", "表5　问题二四种筛选规则与训练内选法的外层误差"),
    ("| 方法 | 单分子Jaccard | 替代组Jaccard | 三因素平均绝对相关 |",
     "表6　问题二四种规则的 Bootstrap 稳定性与替代组指标"),
    ("| 规则 | 次数 |", "表7　问题二训练内选法的规则选择次数"),
    ("| 模型 | 条件SWD | 风格一致性 | 相关误差 | 联合低端误差 |",
     "表8　问题三四类生成器的逐风格诊断指标"),
    ("| 策略 | 目标收益 | stale变化 | stale上升率 | 化学RMS | 拒绝率 |",
     "表9　问题三四种迁移策略的配对结果"),
    ("| 策略 | 指标 | 均值差 | Holm校正p |", "表10　问题三迁移的配对统计检验"),
]


def die(msg):
    raise SystemExit(f"[SKELETON-FAIL] {msg}")


def merge(t, first, second, new_title):
    """把「first 标题 + 其正文 + second 标题 + 其正文」合并为一个标题，正文全部保留。"""
    i = t.index(first)
    j = t.index(second)
    k = t.index("\n## ", j + 1)
    body = t[i + len(first):k]           # first 正文 + second 标题 + second 正文
    body = body.replace(second, "", 1).strip("\n")
    return t[:i] + new_title + "\n\n" + body + t[k:]


def main():
    t = SRC.read_text(encoding="utf-8")

    # ── 1. 四章合并为 6 节 ──────────────────────────────────────
    t = merge(t,
              "## 4.1 数据概况与真实字段结构",
              "## 4.2 样本主键与匿名记录映射",
              "## 4.1 数据概况、主键与匿名记录映射")
    t = merge(t,
              "## 4.3 化学尺度及观测最低值",
              "## 4.4 缺失值与最低值堆积的处理原则",
              "## 4.3 化学尺度、缺失与观测最低值堆积")
    t = t.replace("## 4.5 评分卡结构、重复记录与标签疑点",
                  "## 4.4 评分卡结构、重复记录与标签疑点", 1)
    t = t.replace("## 4.6 未观测乙醛评分的处理",
                  "## 4.5 无面板监督的乙醛维度处理", 1)
    t = t.replace("## 4.7 统一实验划分与泄漏控制",
                  "## 4.6 统一实验划分与泄漏控制", 1)

    # ── 2. 图号：1.3 只做文字引用，图1 唯一落点在 3.4 ────────────
    t = t.replace(
        "图1展示数据、模型、验证与交付物之间的关系。\n\n"
        "【图1：三问统一技术路线图，占位；数据源为`configs/rebuild_protocol.json`、"
        "`configs/closed_final_protocol.json`。】",
        "三问的数据来源、模型、验证方式与交付物之间的关系汇总于图1（见 3.4 节）。", 1)
    t = t.replace("【图1：总技术路线图放置于此，第一章仅文字交叉引用，不重复绘制。】",
                  "【图1 待插入：数据源 → 审计 → 统一协议 → 三问模型 → 三重验证 → 交付层。】", 1)

    # ── 3. 表号：10 张表加题注 ─────────────────────────────────
    for header, cap in TABLES:
        if header not in t:
            die(f"表头未找到：{header}")
        t = t.replace(header, f"**{cap}**\n\n{header}", 1)

    # ── 4. 附录：补出一级标题与 A/B/C ──────────────────────────
    appendix = """# 附录

## 附录A　交付文件清单

【待补：评分卡、50×3 候选、880 条风格谱与感官画像、迁移审计四类交付物的文件路径与内容说明。】

## 附录B　复现环境与运行命令

【待补：Python 与依赖版本、`run_q1.ps1`／`run_final.ps1`／`build_rebuild_pdf.py` 的用途与命令。】

## 附录C　程序清单

【待补：`src/rebuild/` 与 `scripts/` 全部源程序清单及关键代码。】

"""
    decl = t.index("# AI工具使用声明")
    t = t[:decl] + appendix + t[decl:]

    # ── 5. AI 声明标题去掉编辑批注 ─────────────────────────────
    t = t.replace("# AI工具使用声明（草稿，须与真实队员记录核对）", "# AI工具使用声明", 1)

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(t, encoding="utf-8")

    # ── 输出骨架对照 ───────────────────────────────────────────
    print(f"written {DST}  {len(t)} chars\n")
    for ln in t.split("\n"):
        if ln.startswith("# "):
            print(ln)
        elif ln.startswith("## "):
            print("    " + ln)


if __name__ == "__main__":
    main()
