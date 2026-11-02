import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from docx import Document
d = Document(r"C:\Users\73173\Downloads\AI工具使用详情说明.docx")
print("=== 段落标题 ===")
for p in d.paragraphs:
    if p.text.strip():
        print(f"[{p.style.name}] {p.text.strip()[:80]}")
