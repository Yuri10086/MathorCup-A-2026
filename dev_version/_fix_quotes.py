# -*- coding: utf-8 -*-
"""把论文源文件里的直角引号「」换成中文弯引号“”，并清掉会印出来的星号。"""
import glob
import os

SEC = r'E:\dsh_work\MC2026A\latex\sections'
ROOT = r'E:\dsh_work\MC2026A\latex'

files = glob.glob(os.path.join(SEC, '*.tex')) + [os.path.join(ROOT, 'main.tex')]
tot = 0
for p in files:
    t = open(p, encoding='utf-8').read()
    n1 = t.count('\u300c')
    n2 = t.count('\u300d')
    if n1 or n2:
        t = t.replace('\u300c', '\u201c').replace('\u300d', '\u201d')
        tot += n1 + n2
    # 会印出星号的通配写法
    t = t.replace(r'\texttt{latex/images/*}', r'\texttt{latex/images/}')
    open(p, 'w', encoding='utf-8').write(t)
print('替换直角引号 %d 个' % tot)

# 复查
left = 0
for p in files:
    t = open(p, encoding='utf-8').read()
    left += t.count('\u300c') + t.count('\u300d')
print('剩余直角引号:', left)

# 找出仍会显示星号的位置（排除代码区与合法 LaTeX）
import re
for p in files:
    for i, line in enumerate(open(p, encoding='utf-8'), 1):
        if '*' in line:
            s = line.strip()
            if re.search(r'\\arabic\*|\\alph\*', s):
                continue
            if 'lstlisting' in s:
                continue
            print('  ? %s:%d %s' % (os.path.basename(p), i, s[:100]))
