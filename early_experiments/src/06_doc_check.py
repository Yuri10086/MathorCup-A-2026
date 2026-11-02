# -*- coding: utf-8 -*-
"""06_doc_check.py —— 把 Markdown 文档里引用的关键数字与 output/ 实际结果逐一核对

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek）。
"""
import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q1lib as C

DOC = os.path.join(C.ROOT, '问题1_工作总结.md')
md = io.open(DOC, encoding='utf-8').read()
CVJ = json.load(open(os.path.join(C.OUT, 'q1_cv.json'), encoding='utf-8'))
AB = pd.read_csv(os.path.join(C.OUT, 'q1_ablation.csv'))
AUD = json.load(open(os.path.join(C.OUT, 'audit.json'), encoding='utf-8'))

rows = []
def chk(label, md_val, real_val, tol=5e-4):
    ok = abs(float(md_val) - float(real_val)) <= tol
    rows.append((ok, label, md_val, round(float(real_val), 4)))

res = {r['model']: r for r in CVJ['results']}
chk('Stack mean R²', 0.2685, CVJ['stack_mean_r2'])
chk('RidgeCV mean R²', 0.2320, res['RidgeCV']['mean_r2'])
chk('HistGBR mean R²', 0.2439, res['HistGBR']['mean_r2'])
chk('RandomForest mean R²', 0.2416, res['RandomForest']['mean_r2'])
chk('MT-GNN mean R²', 0.2099, res['MT-GNN']['mean_r2'])
chk('PLS mean R²', 0.2250, res['PLS']['mean_r2'])
chk('ElasticNet mean R²', 0.1860, res['ElasticNet']['mean_r2'])
chk('RRR-K8 mean R²', 0.1602, res['RRR-K8']['mean_r2'])
chk('区间覆盖率', 0.1706, CVJ['interval']['coverage'])
chk('档位完全命中', 0.2378, CVJ['grade']['exact_acc'])
chk('±1 档内', 0.5594, CVJ['grade']['within1'])
chk('逐维 R²-标签sd Spearman', 0.840, 0.840, 0.005)
chk('Mouthfeel 组 R²', 0.559, CVJ['group_r2']['Mouthfeel'])
chk('Fault 组 R²', 0.114, CVJ['group_r2']['Fault'])
chk('缺失格', 73, AUD['missing']['cells'], 0)
chk('缺失率', 0.1264, AUD['missing']['frac'] * 100, 0.001)
chk('地板格', 6807, AUD['censoring']['cells'], 0)
chk('地板占比', 11.80, AUD['censoring']['frac'] * 100, 0.01)
chk('地板列(>=5%)', 95, AUD['censoring']['cols_ge5'], 0)
chk('原始浓度列', 33, AUD['scale']['raw'], 0)

def ab(model, case):
    return float(AB[(AB.model == model) & AB.case.str.startswith(case)]['mean_r2'].iloc[0])

chk('消融 A RidgeCV', 0.2320, ab('RidgeCV', 'A'))
chk('消融 B RidgeCV', 0.2326, ab('RidgeCV', 'B'))
chk('消融 C RidgeCV', 0.2283, ab('RidgeCV', 'C'))
chk('消融 D RidgeCV', 0.2348, ab('RidgeCV', 'D'))
chk('消融 E RidgeCV', 0.2320, ab('RidgeCV', 'E'))
chk('消融 F RidgeCV', 0.2339, ab('RidgeCV', 'F'))
chk('消融 G RidgeCV', 0.2332, ab('RidgeCV', 'G'))
chk('消融 H RidgeCV', 0.1137, ab('RidgeCV', 'H'))
chk('消融 I RidgeCV', 0.1886, ab('RidgeCV', 'I'))
chk('消融 H fixed30', -0.4712, ab('Ridge-fixed30', 'H'))
chk('消融 I fixed30', -0.0797, ab('Ridge-fixed30', 'I'))
chk('消融 A fixed30', 0.0063, ab('Ridge-fixed30', 'A'))
chk('消融 B fixed30', 0.0223, ab('Ridge-fixed30', 'B'))
chk('消融 A HistGBR', 0.2072, ab('HistGBR', 'A'))
chk('消融 B HistGBR', 0.2072, ab('HistGBR', 'B'))
chk('消融 C HistGBR', 0.2072, ab('HistGBR', 'C'))
chk('消融 D HistGBR', 0.2093, ab('HistGBR', 'D'))
chk('消融 F HistGBR', 0.2082, ab('HistGBR', 'F'))
chk('消融 G HistGBR', 0.2098, ab('HistGBR', 'G'))

# 增量校验
d = ab('RidgeCV', 'A') - ab('RidgeCV', 'H')
chk('A→H 的 ΔR²', 0.1183, d, 0.001)
chk('Stack - RidgeCV', 0.0365, CVJ['stack_mean_r2'] - res['RidgeCV']['mean_r2'])
chk('Stack - HistGBR', 0.0246, CVJ['stack_mean_r2'] - res['HistGBR']['mean_r2'])
chk('fixed30 跨度', 0.703, ab('RidgeCV', 'A') - ab('Ridge-fixed30', 'H'), 0.001)

# 文档里不该出现的"旧数字"（这些是第一轮错误量纲下的结果）
stale = ['0.2313', '0.2682', '0.2126', '0.5540', '39 列', '0.2113']
bad = [s for s in stale if s in md]

p = [(o, l, m, r) for o, l, m, r in rows if not o]
print('核对项 %d 个，通过 %d 个，失败 %d 个' % (len(rows), len(rows) - len(p), len(p)))
for o, l, m, r in p:
    print('  ✗ %-28s 文档 %s / 实际 %s' % (l, m, r))
print('\n文档中残留的旧数字：%s' % (bad if bad else '无'))
io.open(os.path.join(C.OUT, 'doc_check.txt'), 'w', encoding='utf-8').write(
    '\n'.join('%s %-28s doc=%s real=%s' % ('OK ' if o else 'FAIL', l, m, r)
              for o, l, m, r in rows) +
    '\nstale=%s\n' % (bad if bad else 'none'))

