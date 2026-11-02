import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from docx import Document
d = Document(r"C:\Users\73173\Downloads\AI工具使用详情说明.docx")
t = d.tables[0]
print("=== 表0 实际内容 ===")
for r in t.rows:
    print("  ", " | ".join(c.text.strip() for c in r.cells))
print()
t6 = d.tables[6]
print("=== 表6 第一行核对 ===")
print("  ", [c.text.strip()[:50] for c in t6.rows[1].cells])
