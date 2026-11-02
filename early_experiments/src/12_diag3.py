# -*- coding: utf-8 -*-
"""12_diag3.py —— 关键判定：7.csv 身份、噪声天花板、风格基线

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek）。

决定"差异化"能不能立在"可预测性上限"上：
  * 若 7.csv 是同一批啤酒的独立感官测量 → 可直接估测量噪声天花板
  * 若风格标签单独就能达到接近化学模型的 OOF R² → 化学增量必须重新表述
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import q1lib as C

D = C.load()
Y = D['Y'].values.astype(float)
sdim = D['sdim']
style = D['style']

print('=' * 78)
print('A. 7.csv 到底是什么？逐列找最相关的 4.csv 维度')
print('=' * 78)
m7 = pd.read_csv(os.path.join(C.DATA, '7.csv')).sort_values('beer').reset_index(drop=True)
num7 = [c for c in m7.columns if c not in ('beer', 'beer_id', 'tasting_category_fine')]
print('7.csv 数值列数:', len(num7))
best = []
for c in num7:
    v = pd.to_numeric(m7[c], errors='coerce').values.astype(float)
    if not np.isfinite(v).any():
        continue
    cc = []
    for j in range(50):
        y = Y[:, j]
        m = np.isfinite(v) & np.isfinite(y)
        if m.sum() > 30 and np.std(v[m]) > 0:
            cc.append(abs(np.corrcoef(v[m], y[m])[0, 1]))
    if cc:
        best.append((max(cc), c, sdim[int(np.argmax(cc))]))
best.sort(reverse=True)
print('7.csv 各列与 4.csv 的最大 |r| 前 12：')
for r, c, s in best[:12]:
    print('  %-28s vs %-16s |r|=%.3f' % (c, C.SENS_ZH.get(s, s), r))
print('全部列的最大 |r| 中位数 = %.3f' % np.median([b[0] for b in best]))
print('→ 若中位数接近 0，则 7.csv 与 4.csv 描述的不是同一感官体系')

# 再看 7.csv 列名体系
print('\n7.csv 列名（前 20）:', num7[:20])

print()
print('=' * 78)
print('B. 风格标签的 OOF 基线（纯查表，不用任何化学）')
print('=' * 78)
strat = C.strat_labels(style)
oof_style = np.zeros_like(Y)
for rep in range(3):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RS + rep)
    acc = np.zeros_like(Y)
    for tr, te in skf.split(Y, strat):
        for g in np.unique(style[tr]):
            m = style[tr] == g
            t = style[te] == g
            if t.sum():
                acc[te[t]] = Y[tr][m].mean(0)
        # 未见风格 → 全局均值
        unseen = te[~np.isin(style[te], np.unique(style[tr]))]
        if len(unseen):
            acc[unseen] = Y[tr].mean(0)
    oof_style += acc / 3
r_st = C.per_dim_r2(Y, oof_style)
print('风格均值 OOF: mean R²=%.4f  median=%.4f' % (np.nanmean(r_st), np.nanmedian(r_st)))
print('化学 stack OOF: 0.2685')
print('→ 风格单独已达 %.3f，化学相对风格的增量需单独报告' % np.nanmean(r_st))

# 风格 + 化学 的增量：用残差化思路粗估
print()
print('=' * 78)
print('C. 测量噪声天花板：可用重复测量吗？')
print('=' * 78)
print('4.csv 每个 beer 只有一行 → 无同一样本的重复测量。')
print('替代思路：用"相邻啤酒(同风格内最相似化学)"近似重复，估计可复现性。')
# 同风格内，用化学距离最近的邻居的感官，作为"同一样本的另一次测量"代理
from sklearn.preprocessing import StandardScaler
scale, _ = C.colscale()
Xh = C.harmonize(D['X'], D['comp'], scale).values.astype(float)
Xi = np.where(np.isfinite(Xh), Xh, np.nanmedian(Xh, 0))
Xi = StandardScaler().fit_transform(Xi)
icc_like = []
for g in np.unique(style):
    idx = np.where(style == g)[0]
    if len(idx) < 6:
        continue
    Z = Xi[idx]
    d = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nb = np.argmin(d, 1)
    # 邻居的感官向量 vs 自身感官向量：上界参考
    for j in range(50):
        a, b = Y[idx, j], Y[idx[nb], j]
        if np.std(a) > 0 and np.std(b) > 0:
            icc_like.append(np.corrcoef(a, b)[0, 1])
icc_like = np.array(icc_like)
print('同风格化学最近邻的感官相关: mean=%.3f median=%.3f' %
      (np.nanmean(icc_like), np.nanmedian(icc_like)))
print('→ 这是"化学能解释的最大相关"的经验估计（非严格天花板）')
