# -*- coding: utf-8 -*-
"""10_diag_diff.py —— 差异化路线的可行性诊断

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek）。

回答四个问题，决定"差异"应该建在哪一层：
  Q1 50 维感官矩阵是不是低秩？（若低秩 → 联合低秩建模优于逐维独立）
  Q2 风格(21 类)本身解释多少方差？（若很高 → 化学信息增量有限，需换评价轴）
  Q3 8.xlsx 的气味阈值能否与 1.csv 化合物对齐、构造 OAV？（机理建模的前提）
  Q4 6.csv 的 44×50 矩阵是什么？（若为"气味物→描述符"先验 → 可做前向机理模型）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import q1lib as C

D = C.load()
Y = D['Y'].values.astype(float)
X = D['X']
comp = D['comp']
sdim = D['sdim']
style = D['style']

print('=' * 78)
print('Q1 感官矩阵的秩结构')
print('=' * 78)
Yc = Y - Y.mean(0)
U, s, Vt = np.linalg.svd(Yc, full_matrices=False)
ev = s ** 2 / np.sum(s ** 2)
print('shape', Y.shape)
print('前 k 个主成分累计解释方差：')
for k in (1, 2, 3, 5, 8, 10, 15, 20, 30):
    print('  k=%-3d  %.4f' % (k, ev[:k].sum()))
print('参与比(有效秩) exp(H) = %.2f' % np.exp(-np.sum(ev * np.log(ev + 1e-12))))
# 若前 5 个主成分就占大部分方差，说明 50 维其实是少数几个"潜感官轴"
print('\n前 3 个主成分负载最大的维度：')
for i in range(3):
    idx = np.argsort(-np.abs(Vt[i]))[:6]
    print('  PC%d: ' % (i + 1) + ', '.join('%s(%.2f)' % (C.SENS_ZH.get(sdim[j], sdim[j]), Vt[i][j])
                                          for j in idx))

print()
print('=' * 78)
print('Q2 风格标签解释的方差（仅用风格均值预测）')
print('=' * 78)
sty = pd.Series(style)
mu = Y.mean(0)
pred_style = np.zeros_like(Y)
for g in sty.unique():
    m = (sty == g).values
    pred_style[m] = Y[m].mean(0)
r_style = C.per_dim_r2(Y, pred_style)
print('风格均值预测 mean R² = %.4f  median = %.4f' % (np.nanmean(r_style), np.nanmedian(r_style)))
print('对照：化学特征 stack 0.2685')
print('→ 风格解释的方差占比 %.1f%%，化学信息必须在此之上才有增量'
      % (100 * np.nanmean(r_style) / 1.0))

print()
print('=' * 78)
print('Q3 气味阈值表')
print('=' * 78)
xl = pd.ExcelFile(os.path.join(C.DATA, '8.xlsx'))
print('8.xlsx 工作表:', xl.sheet_names)
th = pd.read_excel(os.path.join(C.DATA, '8.xlsx'), sheet_name='Sensory thresholds')
print('Sensory thresholds shape:', th.shape)
print('列名:', list(th.columns))
print(th.head(8).to_string()[:1500])

print()
print('=' * 78)
print('Q4 6.csv / 5.csv 结构')
print('=' * 78)
m6 = pd.read_csv(os.path.join(C.DATA, '6.csv'), index_col=0)
print('6.csv shape %s' % (m6.shape,))
print('  行(气味物)示例:', list(m6.index)[:5])
print('  列(描述符)示例:', list(m6.columns)[:3])
print('  取值范围 %.4f ~ %.4f，非零比例 %.3f' % (m6.values.min(), m6.values.max(),
                                                (m6.values != 0).mean()))
# 6.csv 的行名与 1.csv 列名的重合度
cn = {c.lower().strip() for c in comp}
hit = [r for r in m6.index if str(r).lower().strip() in cn]
print('  与 1.csv 列名直接匹配: %d / %d' % (len(hit), len(m6.index)))
if len(hit) < len(m6.index):
    print('  未匹配示例:', [r for r in m6.index if str(r).lower().strip() not in cn][:8])

m5 = pd.read_csv(os.path.join(C.DATA, '5.csv'), index_col=0)
print('\n5.csv shape %s' % (m5.shape,))
print('  行示例:', list(m5.index)[:3])
print('  列示例:', list(m5.columns)[:3])
print('  取值范围 %.4f ~ %.4f，对角均值 %.4f' % (m5.values.min(), m5.values.max(),
                                              np.mean(np.diag(m5.values))))
print('  是否对称: %s' % np.allclose(m5.values, m5.values.T, atol=1e-8))

m7 = pd.read_csv(os.path.join(C.DATA, '7.csv'), index_col=0)
print('\n7.csv shape %s' % (m7.shape,))
print('  列示例:', list(m7.columns)[:6])
print('  缺失总数 %d (%.2f%%)' % (m7.isna().sum().sum(),
                                 100 * m7.isna().mean().mean()))
