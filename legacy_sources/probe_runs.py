import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from docx import Document
d = Document(r"C:\Users\73173\Downloads\AI工具使用详情说明.docx")
# 检查单元格内 run 结构，决定填充方式
for ti in (0, 1, 4, 6):
    tb = d.tables[ti]
    c = tb.rows[1].cells[1]
    print(f"表{ti} 单元格 runs: {len(c.paragraphs)} 段, 首段 runs={len(c.paragraphs[0].runs)}")
    for r in c.paragraphs[0].runs[:2]:
        print(f"   run: bold={r.bold} size={r.font.size} name={r.font.name} text={r.text[:40]!r}")
