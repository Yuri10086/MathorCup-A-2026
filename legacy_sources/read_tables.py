import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from docx import Document
d = Document(r"C:\Users\73173\Downloads\AI工具使用详情说明.docx")
for ti, tb in enumerate(d.tables):
    print(f"\n===== 表 {ti}  行={len(tb.rows)} 列={len(tb.columns)} =====")
    for ri, row in enumerate(tb.rows):
        cells = []
        for c in row.cells:
            txt = c.text.strip().replace("\n", " / ")
            cells.append(txt[:46])
        print(f"  [{ri}] " + " | ".join(cells))
