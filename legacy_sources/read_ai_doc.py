import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from docx import Document
d = Document(r"C:\Users\73173\Downloads\AI工具使用详情说明.docx")
print("段落数:", len(d.paragraphs), " 表格数:", len(d.tables))
print("=" * 70)
for i, p in enumerate(d.paragraphs):
    t = p.text.strip()
    if t:
        print(f"[{i}] ({p.style.name}) {t[:160]}")
