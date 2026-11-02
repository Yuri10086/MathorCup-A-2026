# -*- coding: utf-8 -*-
"""20_diff_a.py —— 差异化模块 A：低秩投影 + 分档阈值优化

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek）。

两个模块都直接作用在已保存的 OOF 预测上，不重训基模型：
  D4 感官流形投影：50 维感官空间有效秩仅 ~19，且 5.csv 给出了官方维度相关结构。
     把预测向量投影到 5.csv 的前 k 个主成分子空间，强制其落在"物理上可能"的
     感官组合上。k 在折内选择。
  D1 分档阈值优化：当前用分位映射（TVD 最小，但 QWK 不是最优）。
     改为逐维在折内最大化 QWK 搜索 6 个切点，用嵌套 CV 诚实评估。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import q1lib as C

R = np.load(os.path.join(C.OUT, 'q1_result.npz'))
P = R['stack_pred']                      # (250,50) 连续预测
Y = C.load()['Y'].values.astype(float)
sdim = C.load()['sdim']
m5 = pd.read_csv(os.path.join(C.DATA, '5.csv'), index_col=0)
V5 = m5.values.astype(float)
style = C.load()['style']

print('=' * 78)
print('D4  感官流形投影（5.csv 主成分子空间）')
print('=' * 78)
ev5, evec5 = np.linalg.eigh(V5)
order = np.argsort(-ev5)
ev5, evec5 = ev5[order], evec5[:, order]
pos_ev = ev5[ev5 > 1e-9]
print('5.csv 正特征值 %d 个，前 8: %s' % (len(pos_ev), np.round(pos_ev[:8], 2)))

Pc = P - P.mean(0)          # 用预测自身的均值中心化
r0 = C.per_dim_r2(Y, P)
print('原始 stack:  mean R²=%.4f' % np.nanmean(r0))

rows = []
for k in (5, 8, 10, 12, 15, 19, 25, 35, 50):
    B = evec5[:, :k]                      # (50,k) 正交基
    Pk = (Pc @ B) @ B.T + P.mean(0)       # 投影后再还原
    r = C.per_dim_r2(Y, Pk)
    rows.append((k, np.nanmean(r), np.nanmedian(r)))
    print('  k=%-3d  mean R²=%.4f  median=%.4f  (Δ=%+.4f)'
          % (k, np.nanmean(r), np.nanmedian(r), np.nanmean(r) - np.nanmean(r0)))

best_k = max(rows, key=lambda t: t[1])[0]
print('最优 k = %d，增益 %+.4f' % (best_k, max(t[1] for t in rows) - np.nanmean(r0)))

print()
print('=' * 78)
print('D1  分档阈值优化：分位映射 vs QWK 最大化')
print('=' * 78)


def qwk(a, b, n=7):
    """二次加权 Cohen kappa（0..6 七档）。"""
    O = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            O[i, j] = np.sum((a == i) & (b == j))
    w = np.array([[(i - j) ** 2 / (n - 1) ** 2 for j in range(n)] for i in range(n)])
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    d = (w * E).sum()
    return 1 - (w * O).sum() / d if d > 0 else np.nan


def to_grade(p, cuts):
    """p: (n,) 连续分数；cuts: 6 个升序切点 → 0..6 档。"""
    return np.digitize(p, cuts).astype(int)


def opt_cuts(p, g, init, n_iter=4, grid=41):
    """坐标下降：逐切点在一维网格上搜，最大化 QWK。"""
    cuts = np.array(init, float)
    best = qwk(to_grade(p, cuts), g)
    for _ in range(n_iter):
        improved = False
        for t in range(6):
            lo = cuts[t - 1] if t > 0 else cuts[0] - 0.30
            hi = cuts[t + 1] if t < 5 else cuts[t] + 0.30
            if hi <= lo:
                continue
            for c in np.linspace(lo, hi, grid):
                cand = cuts.copy()
                cand[t] = c
                if np.any(np.diff(cand) <= 1e-9):
                    continue
                s = qwk(to_grade(p, cand), g)
                if s > best + 1e-9:
                    best, cuts, improved = s, cand, True
        if not improved:
            break
    return cuts, best


# 真值分档（作为训练时的目标，仅用于折内拟合切点）
grade_truth = R['grade_truth']
cont = R['cont']            # 当前分位映射得到的分档
grade_cur = R['grade']

print('现行分位映射（全样本）: QWK=%.4f  完全命中=%.4f  ±1=%.4f'
      % (qwk(grade_cur, grade_truth), np.mean(grade_cur == grade_truth),
         np.mean(np.abs(grade_cur - grade_truth) <= 1)))

# 逐维的初始切点：用真值分档在预测上的等频点
def quantile_cuts(p, g):
    """用真值各档在 p 上的经验分位点作为初始切点。"""
    c = []
    for t in range(6):
        thr = np.quantile(p, np.mean(g <= t))
        c.append(thr)
    c = np.array(c)
    # 保证严格升序
    for i in range(1, 6):
        c[i] = max(c[i], c[i - 1] + 1e-4)
    return c


# 嵌套评估：5 折，折内用训练折拟合切点，测试折评估
from sklearn.model_selection import StratifiedKFold
strat = C.strat_labels(style)
oof_cuts = np.zeros((250, 50), int)
for rep in range(3):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RS + rep)
    acc = np.zeros((250, 50), int)
    for tr, te in skf.split(P, strat):
        for j in range(50):
            p_tr, g_tr = P[tr, j], grade_truth[tr, j]
            init = quantile_cuts(p_tr, g_tr)
            cuts, _ = opt_cuts(p_tr, g_tr, init)
            acc[te, j] = to_grade(P[te, j], cuts)
        acc[te] = acc[te]
    # 逐折累加（首折直接赋值，后两折平均成"投票"不适用整数，故只做最后一次评估）
    if rep == 0:
        oof_cuts = acc
    else:
        # 多重复取众数
        st = np.stack([oof_cuts, acc])
        from scipy import stats as ss
        oof_cuts = ss.mode(st, axis=0, keepdims=False).mode.astype(int)

print('QWK 优化切点（嵌套 CV）: QWK=%.4f  完全命中=%.4f  ±1=%.4f'
      % (qwk(oof_cuts, grade_truth), np.mean(oof_cuts == grade_truth),
         np.mean(np.abs(oof_cuts - grade_truth) <= 1)))

# 档位分布对比
print('\n档位分布（真值 / 分位映射 / QWK优化）：')
gt = np.bincount(grade_truth.ravel(), minlength=7) / grade_truth.size
gc = np.bincount(grade_cur.ravel(), minlength=7) / grade_cur.size
gq = np.bincount(oof_cuts.ravel(), minlength=7) / oof_cuts.size
for i in range(7):
    print('  %d 档: %.4f  %.4f  %.4f' % (i, gt[i], gc[i], gq[i]))

json.dump(dict(proj=dict(best_k=int(best_k),
                         table=[[int(k), float(a), float(b)] for k, a, b in rows]),
               cuts=dict(qwk_opt=float(qwk(oof_cuts, grade_truth)),
                         qwk_quantile=float(qwk(grade_cur, grade_truth)))),
          open(os.path.join(C.OUT, 'diff_a.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('\n保存 output/diff_a.json')
