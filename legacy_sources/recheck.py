import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from docx import Document
from docx.text.paragraph import Paragraph
d = Document(r"C:\Users\73173\Downloads\AI工具使用详情说明.docx")
print("表格数:", len(d.tables))
for ti, tb in enumerate(d.tables):
    print(f"表{ti}: {len(tb.rows)}行 × {len(tb.columns)}列  首行={[c.text.strip()[:14] for c in tb.rows[0].cells]}")
print("-" * 60)
for p in d.paragraphs:
    if p.text.strip():
        print(f"[{p.style.name}] {p.text.strip()[:70]}")
