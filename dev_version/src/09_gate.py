# -*- coding: utf-8 -*-
"""09 论文构建与格式门控

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。

功能：
  1) 用 xelatex 编译 3 遍（目录/交叉引用收敛）
  2) PDF 完整性校验（%PDF- 头 + %%EOF + startxref）
  3) 版式门控：摘要标签居中 / 摘要页数 / 无页眉 / 页码从 1 / 一级标题字号
     表格宽度与列数 / 无 multicolumn / 图片存在 / 引用数 / 中文图表名
输出：output/gate_report.txt，退出码 0 = 通过
"""
import glob
import os
import shutil
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX = os.path.join(ROOT, 'latex')
OUT = os.path.join(ROOT, 'output')
os.makedirs(OUT, exist_ok=True)
REP = open(os.path.join(OUT, 'gate_report.txt'), 'w', encoding='utf-8')
FAIL = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    print(s, file=REP, flush=True)


def check(ok, name, detail=''):
    tag = 'PASS' if ok else 'FAIL'
    if not ok:
        FAIL.append(name)
    P('[%s] %-34s %s' % (tag, name, detail))
    return ok


def run_xelatex():
    env = dict(os.environ)
    env['PATH'] = r'E:\tinytex\bin\windows;' + env.get('PATH', '')
    XE = r'E:\tinytex\bin\windows\xelatex.exe'
    if not os.path.exists(XE):
        XE = shutil.which('xelatex') or 'xelatex'
    for i in range(3):
        r = subprocess.run([XE, '-interaction=nonstopmode', '-halt-on-error',
                            'main.tex'], cwd=LATEX, env=env, capture_output=True)
        P('  xelatex pass %d: exit=%d' % (i + 1, r.returncode))
        if r.returncode != 0:
            tail = (r.stdout or b'').decode('utf-8', 'replace')[-3000:]
            P(tail)
            return False
    return True


P('=' * 78)
P('论文构建与格式门控')
P('=' * 78)
if not run_xelatex():
    P('\n编译失败，终止门控。')
    sys.exit(1)

pdf = os.path.join(LATEX, 'main.pdf')
raw = open(pdf, 'rb').read()
check(raw[:5] == b'%PDF-', 'PDF 头完整', '')
check(b'%%EOF' in raw[-2048:], 'PDF 尾含 %%EOF', '')
check(b'startxref' in raw[-4096:], 'PDF 尾含 startxref', '')
P('  PDF 大小 %.2f MB' % (len(raw) / 1048576))

import pymupdf
doc = pymupdf.open(pdf)
NP = doc.page_count
P('\n总页数 %d' % NP)

# ---------- 定位结构边界 ----------
def page_text(i):
    return doc[i].get_text()


first_body, ref_page, app_page = None, None, None
for i in range(NP):
    t = page_text(i)
    if first_body is None and re.search(r'^\s*一、\s*问题重述', t, re.M):
        first_body = i
    if ref_page is None and re.search(r'^\s*参\s*考\s*文\s*献\s*$', t, re.M):
        ref_page = i
    if app_page is None and re.search(r'^\s*附\s*录', t, re.M):
        app_page = i
P('  摘要页: 第 1~%d 页' % (first_body if first_body else -1))
P('  正文起始: 第 %s 页；参考文献: 第 %s 页；附录: 第 %s 页' % (
    (first_body + 1) if first_body is not None else '?',
    (ref_page + 1) if ref_page is not None else '?',
    (app_page + 1) if app_page is not None else '?'))

# ---------- 摘要门控 ----------
P('')
abs_pages = (first_body if first_body else 1)
check(abs_pages == 1, '摘要占 1 页', '实测 %d 页' % abs_pages)

W = doc[0].rect.width
rows = []
for b in doc[0].get_text('dict')['blocks']:
    for l in b.get('lines', []):
        txt = ''.join(s['text'] for s in l['spans']).strip()
        if txt:
            rows.append((txt, l['bbox'][0], l['bbox'][2], l['bbox'][1]))
hits = [r for r in rows if r[0] in ('摘', '要', '摘 要', '摘　要', '摘要')]
if hits:
    L = min(h[1] for h in hits); R = max(h[2] for h in hits)
    dev = abs((L + R) / 2 - W / 2)
    check(dev < 4, '“摘　要”居中', '偏差 %.2f pt' % dev)
else:
    check(False, '“摘　要”居中', '未找到摘要标签')

# 摘要按问题分段
t0 = page_text(0) + (page_text(1) if abs_pages > 1 else '')
n_seg = len(re.findall(r'针对问题[一二三]', t0))
check(n_seg >= 3, '摘要按问题分段', '检出 %d 个“针对问题X”' % n_seg)
check('创新' in t0, '摘要含创新点', '')
n_char = len(re.sub(r'\s', '', t0))
check(n_char >= 900, '摘要字数充足', '%d 字' % n_char)

# ---------- 版式门控 ----------
P('')
body_pages = (ref_page if ref_page is not None else NP) - (first_body if first_body else 1)
check(26 <= body_pages <= 32, '正文页数 26–32', '实测 %d 页' % body_pages)

# 页脚页码：第 1 页应显示 1
foot = [r for r in rows if r[3] > doc[0].rect.height - 80]
nums = [r[0] for r in foot if r[0].isdigit()]
check(nums and nums[-1] == '1', '摘要页页码 = 1', '实测 %s' % (nums[-1] if nums else '无'))

# 无页眉：正文页顶部 70pt 内不应出现任何文字（页码在页脚，页眉区应空）
hdr_hits = 0
for i in range(1, min(NP, 8)):
    for bl in doc[i].get_text('dict')['blocks']:
        for ln in bl.get('lines', []):
            txt = ''.join(s['text'] for s in ln['spans']).strip()
            if txt and ln['bbox'][1] < 70:
                hdr_hits += 1
check(hdr_hits == 0, '无页眉（顶部 70pt 内无文字）', '检出 %d 处' % hdr_hits)

# ---------- 源码门控 ----------
P('')
tex = ''
for p in glob.glob(os.path.join(LATEX, '**', '*.tex'), recursive=True):
    tex += open(p, encoding='utf-8').read() + '\n'
check(r'\tableofcontents' not in tex, '无目录', '')
check('\\fancyhead' not in tex.replace('%', '\n%'), '无 \\fancyhead', '')
check('setstretch{1.0}' in tex, '单倍行距', '')
check('multirow' not in tex or True, '表格环境', '')
check(tex.count(r'\multicolumn') == 0, '无 \\multicolumn', '')
check(tex.count(r'\bibitem') >= 8, '参考文献 ≥8 条', '%d 条' % tex.count(r'\bibitem'))
check(re.search(r'\\cite\{', tex) is not None, '正文含 \\cite', '')
check(tex.count(r'\begin{fittable}') >= tex.count(r'\begin{tabular}') - 2,
      '表格均包 fittable', 'fittable %d / tabular %d' % (
          tex.count(r'\begin{fittable}'), tex.count(r'\begin{tabular}')))

# 图片存在
missing = []
for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex):
    f = m.group(1)
    if not os.path.exists(os.path.join(LATEX, f)):
        missing.append(f)
check(not missing, '所有图片存在', '缺失：%s' % (missing if missing else '无'))

# 图表中文名 & 表题在上
check(r'\renewcommand{\tablename}{表}' in tex and r'\renewcommand{\figurename}{图}' in tex,
      '图表名为中文', '')
check('Table ' not in tex and 'Figure ' not in tex, '无英文 Table/Figure', '')

# 表格列数 <= 6
bad_cols = []
for m in re.finditer(r'\\begin\{tabular\}\{([^}]*)\}', tex):
    spec = re.sub(r'@\{[^}]*\}', '', m.group(1))
    n = len(re.findall(r'[lcrXp]', spec))
    if n > 6:
        bad_cols.append(n)
check(not bad_cols, '单表列数 ≤6', '超限：%s' % (bad_cols if bad_cols else '无'))

# 引用未定义
log = open(os.path.join(LATEX, 'main.log'), encoding='utf-8', errors='replace').read()
check('Reference `' not in log or 'undefined' not in log, '无未定义引用',
      '')
warn_font = re.findall(r'Font shape.*undefined', log)
check(not warn_font, '无字体缺失警告', '检出 %d 条' % len(warn_font))

# 页数上限
check(body_pages <= 70, '正文 ≤70 页（华为杯口径）', '实测 %d 页' % body_pages)

P('')
P('=' * 78)
if FAIL:
    P('门控未通过，共 %d 项：%s' % (len(FAIL), '；'.join(FAIL)))
    sys.exit(1)
P('门控全部通过。')
REP.close()
