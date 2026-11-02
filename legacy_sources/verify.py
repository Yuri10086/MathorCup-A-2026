import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from docx import Document
from docx.text.paragraph import Paragraph
d = Document(r"C:\Users\73173\Downloads\AI工具使用详情说明.docx")
print("表格数:", len(d.tables))
for ti, tb in enumerate(d.tables):
    print(f"\n===== 表{ti}  {len(tb.rows)}行 × {len(tb.columns)}列 =====")
    for ri, row in enumerate(tb.rows):
        vals = [c.text.strip().replace("\n", " ⏎ ")[:120] for c in row.cells]
        print(f"  [{ri}] " + " | ".join(vals))
print("\n=== 残留占位符【】检查 ===")
leftover = sum(1 for tb in d.tables for r in tb.rows for c in r.cells if "【" in c.text)
print("  剩余占位符单元格数:", leftover)
