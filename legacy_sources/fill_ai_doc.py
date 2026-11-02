"""填写 AI工具使用详情说明.docx —— 仅记录 DeepSeek 本轮参与部分。

删除策略：按 body 中的连续区间删除「记录1 豆包」小节（从该标题到「记录2 DeepSeek」之前），
避免逐一匹配导致把 DeepSeek 的截图位也误删。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from docx import Document
from docx.text.paragraph import Paragraph

SRC = r"C:\Users\73173\Downloads\AI工具使用详情说明.docx"
d = Document(SRC)
LQ, RQ = "\u300c", "\u300d"


def set_cell(cell, text):
    p = cell.paragraphs[0]
    if p.runs:
        p.runs[0].text = text
        for extra in p.runs[1:]:
            extra.text = ""
    else:
        p.add_run(text)


def set_cell_multi(cell, lines):
    p0 = cell.paragraphs[0]
    first_size = p0.runs[0].font.size if p0.runs else None
    set_cell(cell, lines[0])
    for ln in lines[1:]:
        np = cell.add_paragraph()
        r = np.add_run(ln)
        if first_size is not None:
            r.font.size = first_size


# ---- 1. 先删除「记录1 豆包」连续小节 ----
body = d.element.body
children = list(body.iterchildren())
start_idx = end_idx = None
for idx, child in enumerate(children):
    if child.tag.split("}")[-1] != "p":
        continue
    t = Paragraph(child, d).text.strip()
    if t.startswith("记录1") and start_idx is None:
        start_idx = idx
    elif t.startswith("记录2") and start_idx is not None:
        end_idx = idx
        break
if start_idx is not None and end_idx is not None:
    for idx in range(end_idx - 1, start_idx - 1, -1):
        body.remove(children[idx])
    print("已删除「记录1 豆包」小节，共 %d 个元素" % (end_idx - start_idx))
else:
    print("未找到豆包小节区间，跳过")

# ---- 2. 表0：AI 工具基本信息 ----
t0 = d.tables[0]
set_cell(t0.rows[1].cells[0], "DeepSeek")
set_cell(t0.rows[1].cells[1], "深度求索（DeepSeek）")
set_cell(t0.rows[1].cells[2], "DeepSeek-V4.1-Flash")
set_cell(t0.rows[1].cells[3], "2026-10-30")
if len(t0.rows) > 2:
    t0._tbl.remove(t0.rows[2]._tr)

# ---- 3. 表1：使用目的与应用环节 ----
t1 = d.tables[1]
set_cell(t1.rows[1].cells[0], "DeepSeek")
set_cell_multi(t1.rows[1].cells[1], ["代码审查、缺陷定位与修改建议；工程结构梳理；文档一致性核对"])
set_cell_multi(t1.rows[1].cells[2], [
    "1) 逐文件审读 src/rebuild 下全部源程序，核对论文附录印出的代码与磁盘文件是否一致；",
    "2) 发现 scripts/run_rebuild.ps1 第 2 行硬编码本机解释器路径，建议改为空默认值，交由 run_final.ps1 的 Resolve-Tool 解析；",
    "3) 指出 configs/closed_final_protocol.json 的 SHA-256 冻结哈希会因任何正当修改（含改注释）而中断流程，其防御性质经参赛队判断后移除；",
    "4) 指出 run.py 的 audit() 会静默覆盖原始输入基线，改为输入不同时另写 raw_manifest.rebuilt.csv 并保留原基线；",
    "5) 按一题一文件要求，把分散模块整理为 q1.py / q2.py / q3.py 三个解题文件加 tables.py 共用汇总文件，并同步更新两个 PowerShell 入口；",
    "6) 核对文档中的检查项数目（原 101/101 因移除哈希项降为 65/65），统一更正 README、PAPER_VERSION 与提交说明。",
])
set_cell_multi(t1.rows[1].cells[3], [
    "scripts/run_rebuild.ps1；src/rebuild/*.py（结构整理）",
    "configs/closed_final_protocol.json；outputs/rebuild/audit/protocol_amendment.md",
    "README_REBUILD.md；paper/PAPER_VERSION.md；submission/提交说明与待补项.md",
])
if len(t1.rows) > 3:
    t1._tbl.remove(t1.rows[3]._tr)

# ---- 4. 重新计算表格索引 ----
# 删除后剩余：原0, 原1, 原4(DeepSeek交互), 原5(DeepSeek截图), 原6(采纳核验)
print("删除后表格数:", len(d.tables))
t2 = d.tables[2]   # DeepSeek 关键交互
t3 = d.tables[4]   # 采纳内容、人工修改与核验

set_cell_multi(t2.rows[1].cells[1], [
    "2026-10-30；赛题项目代码的人工检查与工程整理（覆盖问题一至问题三的代码组织，不涉及建模本身）"
])
set_cell_multi(t2.rows[2].cells[1], [
    "（原始提示词由参赛队保存，以下为经标明的关键节选）",
    "1) " + LQ + "这一个对话我打算进行赛题项目代码（放在论文后面附录的）的人工检查，指导我主要看哪些文件" + RQ + "；",
    "2) " + LQ + "把硬编码问题改正" + RQ + "；",
    "3) " + LQ + "现在把123题统一用q1、q2、q3这样的形式作为代码的名字" + RQ + "；",
    "4) " + LQ + "官方说只填国产的模型，所以我就只需要你来填一下你的部分（不要覆盖整个项目文件中有关赛题直接求解的函数之类的，可以说是修改与建议）" + RQ + "。",
])
set_cell_multi(t2.rows[3].cells[1], [
    "AI 给出的主要判断（节选）：",
    "· 附录 E.6 closed_verify.py 比磁盘源文件少 6 行，附录版本在 data/raw 为空时会直接报错；",
    "· scripts/run_rebuild.ps1 把本机 conda 解释器路径写死为默认值，且该行被原样印进论文附录；",
    "· 删除 run.py 头部注释后冻结哈希失配（期望 e3504aac…、实际 1ab8517a…），closed_verify 会因 assert 失败退出；",
    "· 建议将 closed.py 按 Q2/Q3 拆分，并最终收敛为 q1.py / q2.py / q3.py 三个文件。",
])
set_cell_multi(t2.rows[4].cells[1], [
    "采纳：硬编码修正、哈希闸门移除、audit 基线保护、三问单文件化均被采纳并实际执行。",
    "人工修改：参赛队自行决定删除全部源程序头部的 AI 注释；AI 未覆盖任何建模相关函数体，仅做文件组织与工程性修正。",
    "复核依据：改动后实际运行 closed_verify.py 输出 Closed checks 65/65，verify.py 输出 76/76，各分支实跑通过。",
])

set_cell_multi(t3.rows[1].cells[1], ["未介入建模。仅就问题一代码（评分卡组装、乙醛判定、等级诊断）的文件组织与调用关系提出整理建议。"])
set_cell_multi(t3.rows[1].cells[2], ["参赛队独立决定代码结构；AI 未改动任何求解逻辑与数值。"])
set_cell_multi(t3.rows[1].cells[3], ["src/rebuild/q1.py；outputs/rebuild/q1_complete/"])
set_cell_multi(t3.rows[2].cells[1], ["未介入建模。仅就问题二代码（共表达网络、四条筛选规则、bootstrap）的文件组织提出整理建议。"])
set_cell_multi(t3.rows[2].cells[2], ["参赛队独立决定；AI 未改动筛选规则与统计实现。"])
set_cell_multi(t3.rows[2].cells[3], ["src/rebuild/q2.py；outputs/rebuild/q2/"])
set_cell_multi(t3.rows[3].cells[1], ["未介入建模。仅就问题三代码（生成器、22 风格评价、四策略迁移）的文件组织提出整理建议。"])
set_cell_multi(t3.rows[3].cells[2], ["参赛队独立决定；AI 未改动生成与迁移的算法实现。"])
set_cell_multi(t3.rows[3].cells[3], ["src/rebuild/q3.py；outputs/rebuild/q3/"])
set_cell_multi(t3.rows[4].cells[1], ["协助核对论文与支撑材料中的数字口径与文件名一致性，例如 Closed checks 由 101 项更正为 65 项。"])
set_cell_multi(t3.rows[4].cells[2], ["参赛队逐项核对后采纳；AI 未代写正文内容。"])
set_cell_multi(t3.rows[4].cells[3], ["README_REBUILD.md；paper/PAPER_VERSION.md；submission/提交说明与待补项.md"])

# ---- 5. 修正小节标题 ----
for p in d.paragraphs:
    if p.text.strip().startswith("记录2"):
        for r in p.runs:
            r.text = ""
        if p.runs:
            p.runs[0].text = "记录  DeepSeek（本轮代码审查与工程整理）"
        break

d.save(SRC)
print("已保存")
