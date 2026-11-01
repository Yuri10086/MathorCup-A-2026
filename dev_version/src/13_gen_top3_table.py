# -*- coding: utf-8 -*-
"""生成附录全表 tab_top3full.tex（避免 PowerShell 变量插值问题）。"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

T = pd.read_csv(os.path.join(C.OUT, 'q2_top3.csv'))


def esc(s):
    s = str(s)
    for ch in ['_', '%', '&', '#', '$']:
        s = s.replace(ch, '\\' + ch)
    return s


lines = [r'\begin{table}[H]', r'\centering',
         r'\caption{50 个感官维度的 Top-3 关键化合物全表（按规范化学名给出，括号内为 bootstrap 入选频率）}',
         r'\label{tab:top3full}', r'\scriptsize', r'\begin{fittable}',
         r'\begin{tabular}{@{}llllr@{}}', r'\toprule',
         r'感官维度 & Top-1 & Top-2 & Top-3 & 方法 $J$ \\', r'\midrule']
for _, r in T.iterrows():
    def fmt(k):
        nm = r.get('Top%d化学名' % k)
        if not isinstance(nm, str) or not nm.strip():
            nm = r.get('Top%d' % k)
        f = r.get('Top%d入选频率' % k)
        s = esc(nm)
        if pd.notna(f):
            s += '（%.2f）' % f
        return s
    lines.append('%s & %s & %s & %s & %.2f \\\\' % (
        esc(r['感官维度']), fmt(1), fmt(2), fmt(3), r['方法Jaccard']))
lines += [r'\bottomrule', r'\end{tabular}', r'\end{fittable}', r'\end{table}']
p = os.path.join(C.ROOT, 'latex', 'sections', 'tab_top3full.tex')
open(p, 'w', encoding='utf-8').write('\n'.join(lines))
print('写出 %s，%d 行' % (p, len(T)))
print('\n'.join(lines[:12]))
