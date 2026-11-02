import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from docx import Document
d = Document(r"C:\Users\73173\Downloads\AI工具使用详情说明.docx")
print("=== 底部表格行数（应无“其他工具”行）===")
t1 = d.tables[1]
for i, r in enumerate(t1.rows):
    print(f"  行{i}: {r.cells[0].text.strip()}")
