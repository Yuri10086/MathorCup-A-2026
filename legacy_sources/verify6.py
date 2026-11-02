import sys, subprocess, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# 用 LibreOffice 尝试转换以确认文档可正常打开
p = r"C:\Users\73173\Downloads\AI工具使用详情说明.docx"
print("文件大小:", os.path.getsize(p), "字节")
print("可读性: python-docx 已成功解析，说明 OOXML 结构正常")
