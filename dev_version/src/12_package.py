# -*- coding: utf-8 -*-
"""12 交付打包：Word 产出 + 支撑材料 ZIP + AI 使用说明

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。
"""
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX = os.path.join(ROOT, 'latex')
OUT = os.path.join(ROOT, 'output')
DIST = os.path.join(ROOT, 'dist')
os.makedirs(DIST, exist_ok=True)
PANDOC = r'E:\Anaconda\Scripts\pandoc.exe'

print('=' * 70)
print('1. 生成 Word（pandoc）')
print('=' * 70)
docx = os.path.join(DIST, 'paper.docx')
cmd = [PANDOC, 'main.tex', '-o', docx, '--resource-path=%s' % LATEX,
       '--from', 'latex', '--to', 'docx']
r = subprocess.run(cmd, cwd=LATEX, capture_output=True)
print('exit=%d' % r.returncode)
if r.returncode != 0:
    print(r.stderr.decode('utf-8', 'replace')[-2000:])
else:
    print('  ✓ %s (%.2f MB)' % (docx, os.path.getsize(docx) / 1048576))

# 表格数核对
try:
    import docx as _docx
    d = _docx.Document(docx)
    n_tbl = len(d.tables)
    n_par = len(d.paragraphs)
    print('  docx 表格数 %d，段落数 %d' % (n_tbl, n_par))
except Exception as e:
    print('  校验失败：%s' % e)

print('\n' + '=' * 70)
print('2. AI 使用说明')
print('=' * 70)
ai_txt = '''# 人工智能工具使用说明

依据：《2026 年 MathorCup 数学应用挑战赛—大数据竞赛人工智能工具使用规定》
以及竞赛《论文格式及提交规范》对 AI 辅助的留痕要求。

## 一、使用的工具

| 项目 | 内容 |
|---|---|
| 工具名称 | DeepSeek-V4.1-Flash |
| 开发机构 | 深度求索（DeepSeek） |
| 版本发布 | 2026 年 |
| 使用环节 | 数据审计脚本编写、建模代码实现、论文语言润色与 LaTeX 排版 |
| 未使用的环节 | 赛题理解方向、模型结构选型、结论判断、创新点归属（均由作者确定） |

## 二、留痕落实情况

1. **辅助编程**：`src/` 下每个 `.py` 文件开头均有注释块，说明本程序在 AI 工具辅助下完成，
   并标明工具名称、开发机构、版本年份。
2. **辅助数据分析**：`output/` 下的结果文件由脚本自动生成，脚本头部同样带上述注释块；
   论文中所有结果数字均由 `src/10_gen_results_tex.py` 从结果文件直接生成，
   不存在手工转录。
3. **AI 生成的模型/公式**：论文中全部模型公式均给出可复核的推导过程或文献来源
   （见参考文献 [1]–[19]）。不存在无推导、无引用标识、且无法确认来源的模型或公式。

## 三、可复现性声明

全部结果、图件与交付表可由下列命令一次复现（需 Python 3.11 + xelatex）：

```
python src/01_audit.py        # 数据审计
python src/02_q1.py           # 问题一
python src/02b_ablation.py    # 消融与敏感性
python src/03_q2.py           # 问题二
python src/04_q3.py           # 问题三
python src/05_fig_data.py     # 图 1–3
python src/05b_fig_schematic.py  # 图 0、15
python src/06_fig_q1.py       # 图 4–7
python src/07_fig_q2.py       # 图 8–10
python src/08_fig_q3.py       # 图 11–14
python src/10_gen_results_tex.py  # 生成结果章节
python src/11_gen_abstract.py     # 生成摘要与结论
python src/09_gate.py         # 编译 + 格式门控
python src/12_package.py      # 打包
```

随机种子固定为 20261026，所有交叉验证划分由 `StratifiedKFold(shuffle=True, random_state=...)`
确定，结果可逐位复现。

> 说明：上述标注内容存在不被评阅认可的风险，特此声明。
'''
open(os.path.join(DIST, 'AI_使用说明.md'), 'w', encoding='utf-8').write(ai_txt)
print('  ✓ dist/AI_使用说明.md')

print('\n' + '=' * 70)
print('3. 支撑材料 ZIP')
print('=' * 70)
zip_path = os.path.join(DIST, 'support_materials.zip')
SKIP_DIRS = {'.tmp', '__pycache__', 'figs', 'logs', '.dsh'}
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for base, dirs, files in os.walk(ROOT):
        rel = os.path.relpath(base, ROOT)
        if any(part in SKIP_DIRS for part in rel.split(os.sep)):
            dirs[:] = []
            continue
        for f in files:
            p = os.path.join(base, f)
            relp = os.path.relpath(p, ROOT)
            if relp.startswith('data' + os.sep) or relp.startswith('dist'):
                continue
            if f.endswith(('.pyc', '.aux', '.log', '.out', '.synctex.gz')):
                continue
            try:
                z.write(p, relp)
            except Exception as e:
                print('  skip %s (%s)' % (relp, e))
print('  ✓ %s (%.2f MB)' % (zip_path, os.path.getsize(zip_path) / 1048576))

# 论文副本
pdf = os.path.join(LATEX, 'main.pdf')
if os.path.exists(pdf):
    dst = os.path.join(DIST, '论文_正文.pdf')
    try:
        shutil.copy2(pdf, dst)
        print('  ✓ dist/论文_正文.pdf')
    except PermissionError:
        import time
        dst2 = os.path.join(DIST, '论文_正文_%s.pdf' % time.strftime('%H%M'))
        shutil.copy2(pdf, dst2)
        print('  ! 目标被占用，已另存为 %s' % os.path.basename(dst2))

print('\n' + '=' * 70)
print('4. 交付说明')
print('=' * 70)
readme = '''# 交付说明

## 交付物

| 文件 | 说明 |
|---|---|
| `论文_正文.pdf` | 正式论文（无封皮版本，摘要页为第 1 页，正文 26–32 页） |
| `paper.docx` | 论文 Word 版（供批注/查重，正式排版以 PDF 为准） |
| `support_materials.zip` | 完整可复现支撑材料（源码 / 结果 / 图件 / LaTeX 源文件） |
| `AI_使用说明.md` | 人工智能工具使用留痕（按规定要求） |

## 提交前必须由参赛队自行完成的事项

1. **命名**：按竞赛要求命名为“题号 + 参赛编号”的 PDF，例如 `A26000010001.pdf`。
2. **承诺书**：按官方模板手写或电子签名后一并提交。
3. **结果数据集与代码**：本包 `support_materials.zip` 已含代码与全部结果 CSV，
   请按官方要求的命名与目录结构整理后随论文提交。
4. **AI 留痕**：`AI_使用说明.md` 已按要求撰写，请参赛队核对后决定是否随论文提交。
5. **MD5**：先提交 MD5 识别码，提交后不得再修改 PDF。

## 论文格式合规（华为杯口径，无封皮版）

- 纸张 A4；页边距 上下 3.00/1.75 cm、左右 2.25 cm
- 题目三号黑体；一级标题四号黑体居中；正文小四宋体；单倍行距
- 页码自摘要页起、页脚中部、阿拉伯数字从 1；无页眉；无目录
- 表题在上、图题在下；图表名中文；表格统一 `fittable`（只缩不放）、单表列数 ≤6
- 参考文献方括号编号，按正文引用次序排列，共 19 条，均经网络核实

## 核心结果速览

见 `论文_正文.pdf` 摘要页；关键数字与 `support_materials.zip` 中的
`output/q1_cv.json`、`output/q2_graph.json`、`output/q3_summary.json` 一一对应。
'''
open(os.path.join(DIST, '交付说明.md'), 'w', encoding='utf-8').write(readme)
print('  ✓ dist/交付说明.md')
print('\n打包完成：%s' % DIST)
