import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
d = Document(r"C:\Users\73173\Downloads\AI工具使用详情说明.docx")
body = d.element.body
ti = pi = 0
for child in body.iterchildren():
    tag = child.tag.split("}")[-1]
    if tag == "p":
        p = Paragraph(child, d)
        print(f"P{pi:2d} [{p.style.name}] {p.text.strip()[:90]}")
        pi += 1
    elif tag == "tbl":
        print(f"  TBL{ti}  rows={len(d.tables[ti].rows)}")
        ti += 1
    else:
        print(f"  <{tag}>")
