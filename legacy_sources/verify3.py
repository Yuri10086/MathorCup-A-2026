import sys, zipfile
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = r"C:\Users\73173\Downloads\AI工具使用详情说明.docx"
z = zipfile.ZipFile(p)
bad = z.testzip()
print("包完整性:", "OK" if bad is None else f"损坏: {bad}")
print("条目数:", len(z.namelist()))
print("含 document.xml:", "word/document.xml" in z.namelist())
z.close()
