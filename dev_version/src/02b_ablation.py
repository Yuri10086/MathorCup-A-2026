# -*- coding: utf-8 -*-
"""02b 问题一 消融实验与敏感性分析

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。

消融项：
  A0 原始特征
  A1 + 缺失指示
  A2 + 封底（左删失）指示
  A3 + 共表达图扩散特征
  A4 A1+A2+A3（= 主模型输入）
敏感性：图扩散强度 α ∈ {0, 0.2, 0.35, 0.5, 0.7}；top-k ∈ {4, 8, 16}
输出：output/q1_ablation.json / q1_sensitivity.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

_TMP = r'E:\dsh_work\MC2026A\.tmp'
os.makedirs(_TMP, exist_ok=True)
for k in ('JOBLIB_TEMP_FOLDER', 'TMP', 'TEMP'):
    os.environ[k] = _TMP
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from sklearn.ensemble import HistGradientBoostingRegressor

RS = 20261026
D = C.load()
COMP, SDIM = D['comp'], D['sdim']
X = D['X'].values.astype(float)
Y = D['Y'].values.astype(float)
STYLE = D['style']
META = C.build_meta(COMP)
R, _mp, _mt, _ms = C.corr_cc_aligned(COMP, META)
Ab = np.abs(np.nan_to_num(R, nan=0.0))
np.fill_diagonal(Ab, 0.0)

# 封底指示矩阵
floor_frac = np.array([float((np.abs(X[:, j] - np.nanmin(X[:, j])) < 1e-9).mean())
                       for j in range(X.shape[1])])
cens_cols = np.where(floor_frac >= 0.05)[0]
IS_CENS = np.zeros_like(X)
for j in cens_cols:
    IS_CENS[:, j] = (np.abs(X[:, j] - np.nanmin(X[:, j])) < 1e-9).astype(float)

vc = pd.Series(STYLE).value_counts()
strat = np.array([s if vc[s] >= 5 else 'Rare' for s in STYLE])
print('封底列 %d 个' % len(cens_cols), flush=True)


def make_S(alpha, topk):
    k = min(topk, Ab.shape[0] - 1)
    idx = np.argpartition(-Ab, k, axis=1)[:, :k]
    M = np.zeros_like(Ab)
    for i in range(Ab.shape[0]):
        M[i, idx[i]] = Ab[i, idx[i]]
    M = np.maximum(M, M.T)
    dg = M.sum(1)
    din = np.where(dg > 0, 1 / np.sqrt(dg), 0.0)
    Ah = (M * din[:, None]) * din[None, :]
    return (1 - alpha) * np.eye(Ah.shape[0]) + alpha * Ah


def prep(Xtr, Xte, cens_tr=None, cens_te=None, S=None, use_miss=False):
    p0 = Xtr.shape[1]                    # 图扩散只作用于原始化合物块
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xtr)
    A, B = imp.transform(Xtr), imp.transform(Xte)
    lo, hi = np.percentile(A, [1, 99], 0)
    A = np.clip(A, lo, hi); B = np.clip(B, lo, hi)
    md = np.median(A, 0)
    iq = np.percentile(A, 75, 0) - np.percentile(A, 25, 0)
    bad = iq <= 0; iq[bad] = np.std(A, 0)[bad] + 1e-9
    A = (A - md) / iq; B = (B - md) / iq
    if use_miss:
        mc = np.where(np.isnan(Xtr).any(axis=0))[0]
        if len(mc):
            A = np.hstack([A, np.isnan(Xtr[:, mc]).astype(float)])
            B = np.hstack([B, np.isnan(Xte[:, mc]).astype(float)])
    if cens_tr is not None:
        A = np.hstack([A, cens_tr]); B = np.hstack([B, cens_te])
    if S is not None:
        A = np.hstack([A, A[:, :p0] @ S.T]); B = np.hstack([B, B[:, :p0] @ S.T])
    return A, B


def run(name, use_miss=False, use_cens=False, alpha=None, topk=8, fold_rep=1):
    t0 = time.time()
    pred = np.zeros_like(Y)
    S = make_S(alpha, topk) if alpha is not None else None
    for rep in range(fold_rep):
        skf = StratifiedKFold(5, shuffle=True, random_state=RS + rep)
        for tr, te in skf.split(X, strat):
            A, B = prep(X[tr], X[te],
                        cens_tr=IS_CENS[tr] if use_cens else None,
                        cens_te=IS_CENS[te] if use_cens else None,
                        S=S, use_miss=use_miss)
            for j in range(len(SDIM)):
                m = HistGradientBoostingRegressor(
                    max_iter=100, learning_rate=0.06, max_depth=3,
                    min_samples_leaf=8, l2_regularization=2.0,
                    max_features=0.6, early_stopping=False,
                    random_state=RS).fit(A, Y[tr, j])
                pred[te, j] = m.predict(B)
    r2 = np.array([r2_score(Y[:, j], pred[:, j]) for j in range(len(SDIM))])
    print('%-34s meanR2=%+.4f med=%+.4f  %.0fs' % (name, r2.mean(), np.median(r2),
                                                   time.time() - t0), flush=True)
    return r2, pred


print('\n=== 消融（XGBoost 骨架，5 折单次）===', flush=True)
abl = {}
for nm, kw in [
    ('A0 原始特征', dict()),
    ('A1 +缺失指示', dict(use_miss=True)),
    ('A2 +封底指示', dict(use_cens=True)),
    ('A3 +图扩散(α=.35,k=8)', dict(alpha=0.35, topk=8)),
    ('A4 全部（主模型输入）', dict(use_miss=True, use_cens=True, alpha=0.35, topk=8)),
]:
    abl[nm] = run(nm, **kw)

print('\n=== 敏感性：扩散强度 α ===', flush=True)
sens_alpha = {}
for a in [0.0, 0.2, 0.35, 0.5]:
    r2, _ = run('α=%.2f' % a, alpha=a, topk=8)
    sens_alpha[str(a)] = float(r2.mean())

print('\n=== 敏感性：top-k ===', flush=True)
sens_topk = {}
for kk in [4, 8, 16]:
    r2, _ = run('topk=%d' % kk, alpha=0.35, topk=kk)
    sens_topk[str(kk)] = float(r2.mean())

json.dump(dict(ablation={k: dict(mean_r2=float(v[0].mean()),
                                 median_r2=float(np.median(v[0])),
                                 per_dim=dict(zip(SDIM, v[0].round(4).tolist())))
                        for k, v in abl.items()},
               sens_alpha=sens_alpha, sens_topk=sens_topk,
               n_censored_cols=int(len(cens_cols))),
          open(os.path.join(C.OUT, 'q1_ablation.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('\n完成：output/q1_ablation.json', flush=True)
