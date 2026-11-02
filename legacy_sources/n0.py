import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from docx import Document
d = Document(r"C:\Users\73173\Downloads\AI工具使用详情说明.docx")
print("恢复后 表格数:", len(d.tables), " 段落数:", len(d.paragraphs))
tb=d.tables[0]
for r in tb.rows: print("  ", [c.text.strip()[:24] for c in r.cells])
