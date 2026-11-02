# -*- coding: utf-8 -*-
"""11_diag2.py —— 机制建模可行性：OAV / 7.csv 归属 / 5.csv 先验

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import q1lib as C

D = C.load()
Y = D['Y'].values.astype(float)
comp = D['comp']

print('=' * 78)
print('A. 7.csv 是什么？与 4.csv 的关系')
print('=' * 78)
m7 = pd.read_csv(os.path.join(C.DATA, '7.csv'))
print('7.csv 列名前 8:', list(m7.columns)[:8])
print('是否有 beer 列:', 'beer' in m7.columns, '| beer_id:', 'beer_id' in m7.columns)
c7 = [c for c in m7.columns if c.startswith('aroma_') or c.startswith('flavour_')]
print('感官类列数:', len(c7))
if 'beer' in m7.columns:
    m7s = m7.sort_values('beer').reset_index(drop=True)
    print('beer 对齐后与 4.csv 一致:', np.array_equal(m7s['beer'].values, D['beer']))
# 找出 7.csv 里与 4.csv 同义的列，算相关
pairs = [('aroma_estery', 'A_esters_all'), ('aroma_hops', 'A_hops_all'),
         ('aroma_malty', 'A_malt_all'), ('bitter', 'bitternes'),
         ('sweet', 'sweetness'), ('sour', 'acidity')]
print('\n7.csv 与 4.csv 潜在同义列的相关：')
for a, b in pairs:
    if a in m7.columns and b in D['sens'].columns:
        x = pd.to_numeric(m7[a], errors='coerce').values.astype(float)
        y = D['sens'][b].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 10:
            print('  %-16s vs %-14s r=%.3f (n=%d)' % (a, b, np.corrcoef(x[m], y[m])[0, 1], m.sum()))

print()
print('=' * 78)
print('B. 气味阈值表能覆盖多少 1.csv 化合物？（OAV 建模前提）')
print('=' * 78)
th = pd.read_excel(os.path.join(C.DATA, '8.xlsx'), sheet_name='Sensory thresholds')
th.columns = [str(c).strip() for c in th.columns]
print('threshold_mean 非空:', th['threshold_mean'].notna().sum(), '/', len(th))
print('threshold_median 非空:', th['threshold_median'].notna().sum(), '/', len(th))
# 1.csv 列名 → 8.xlsx field_name_python 的映射
fn = set(str(x).strip() for x in th['field_name_python'])
hit = [c for c in comp if c in fn]
print('1.csv 列名直接命中 field_name_python: %d / %d' % (len(hit), len(comp)))
# 用 chemical_name 别名再试
cn2fn = {}
for _, r in th.iterrows():
    cnn = str(r['chemical_name']).strip().lower()
    f = str(r['field_name_python']).strip()
    if cnn and cnn != 'nan':
        cn2fn.setdefault(cnn, f)
cn_lower = {c.lower().replace('_', ' ').strip(): c for c in comp}
hit2 = []
for k, f in cn2fn.items():
    kk = k.replace('-', ' ').replace('_', ' ')
    for cl, orig in cn_lower.items():
        if kk == cl.replace('-', ' '):
            hit2.append((orig, f))
print('按 chemical_name 别名命中: %d' % len(hit2))
print('示例:', hit2[:10])

# 有阈值的列
has_th = th[th['threshold_mean'].notna() | th['threshold_median'].notna()]
print('有阈值化合物的 field_name_python 示例:', list(has_th['field_name_python'])[:15])
print('这些里能对上 1.csv 的:', [c for c in comp if c in set(has_th['field_name_python'])][:20])

print()
print('=' * 78)
print('C. 5.csv 感官相关矩阵的结构（作为多任务先验）')
print('=' * 78)
m5 = pd.read_csv(os.path.join(C.DATA, '5.csv'), index_col=0)
V = m5.values.astype(float)
ev5 = np.linalg.eigvalsh(V)[::-1]
print('5.csv 前 8 个特征值:', np.round(ev5[:8], 3))
print('负特征值个数:', (ev5 < -1e-8).sum())
# 与 4.csv 实算相关矩阵的一致性
R = np.corrcoef(D['Y'].values.astype(float).T)
iu = np.triu_indices(50, 1)
print('5.csv 与 4.csv 实算相关的 Pearson r = %.4f' %
      np.corrcoef(V[iu], R[iu])[0, 1])
print('→ 若高度一致，5.csv 就是官方给出的感官维度先验相关结构')
print('  对角外均值 5.csv=%.3f  实算=%.3f' % (V[iu].mean(), R[iu].mean()))
