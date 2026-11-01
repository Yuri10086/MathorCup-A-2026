# -*- coding: utf-8 -*-
"""02c 潜因子回归 + 折内特征筛选（补充实验）

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。

动机：无特征筛选时，潜因子 Ridge 在 231 维上仍然过拟合（均值 R²<0）。
本脚本在同一交叉验证协议下，加入“折内 |Spearman| 前 k 个特征”筛选后重做，
用于说明“低秩结构必须与降维结合才有效”这一结论。
结果合并进 output/q1_cv.json（新增模型行）。
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

_TMP = r'E:\dsh_work\MC2026A\.tmp'
os.makedirs(_TMP, exist_ok=True)
for k in ('JOBLIB_TEMP_FOLDER', 'TMP', 'TEMP'):
    os.environ[k] = _TMP
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RS = 20261026
D = C.load()
COMP, SDIM = D['comp'], D['sdim']
X = D['X'].values.astype(float)
Y = D['Y'].values.astype(float)
STYLE = D['style']
vc = pd.Series(STYLE).value_counts()
strat = np.array([s if vc[s] >= 5 else 'Rare' for s in STYLE])


def prep(Xtr, Xte):
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xtr)
    A, B = imp.transform(Xtr), imp.transform(Xte)
    lo, hi = np.percentile(A, [1, 99], 0)
    A = np.clip(A, lo, hi); B = np.clip(B, lo, hi)
    md = np.median(A, 0)
    iq = np.percentile(A, 75, 0) - np.percentile(A, 25, 0)
    bad = iq <= 0; iq[bad] = np.std(A, 0)[bad] + 1e-9
    return (A - md) / iq, (B - md) / iq


def run(name, K=8, nsel=60, alpha=30.0):
    t0 = time.time()
    pred = np.zeros_like(Y)
    for rep in range(3):
        skf = StratifiedKFold(5, shuffle=True, random_state=RS + rep)
        for tr, te in skf.split(X, strat):
            A, B = prep(X[tr], X[te])
            for j in range(len(SDIM)):
                r = np.nan_to_num(np.array([abs(spearmanr(A[:, k], Y[tr, j]).statistic)
                                            for k in range(A.shape[1])]))
                sel = np.argsort(-r)[:nsel]
                Ym = Y[tr] - Y[tr].mean(0)
                _, _, Vt = np.linalg.svd(Ym, full_matrices=False)
                V = Vt[:K].T
                T = Ym @ V
                m = Ridge(alpha=alpha).fit(A[tr][:, sel][:, :] if False else A[:, sel], T)
                pred[te, j] = (m.predict(B[:, sel]) @ V.T + Y[tr].mean(0))[:, j]
    r2 = np.array([r2_score(Y[:, j], pred[:, j]) for j in range(len(SDIM))])
    print('%-28s meanR2=%+.4f med=%+.4f max=%+.4f  %.0fs' % (
        name, r2.mean(), np.median(r2), r2.max(), time.time() - t0), flush=True)
    return r2, pred


print('潜因子回归 + 折内特征筛选', flush=True)
out = {}
for nm, kw in [('潜因子+选60-K8', dict(K=8, nsel=60)),
               ('潜因子+选30-K8', dict(K=8, nsel=30)),
               ('潜因子+选30-K15', dict(K=15, nsel=30))]:
    r2, pred = run(nm, **kw)
    out[nm] = dict(model=nm, mean_r2=float(r2.mean()), median_r2=float(np.median(r2)),
                   max_r2=float(r2.max()), min_r2=float(r2.min()),
                   mean_mae=float('nan'), seconds=0,
                   per_dim=dict(zip(SDIM, r2.round(4).tolist())))

# 合并进 q1_cv.json
p = os.path.join(C.OUT, 'q1_cv.json')
q = json.load(open(p, encoding='utf-8'))
have = {r['model'] for r in q['results']}
for nm, r in out.items():
    if nm not in have:
        q['results'].append(r)
json.dump(q, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('\n已合并 %d 个模型进 q1_cv.json' % len(out), flush=True)

pd.DataFrame({nm: r['per_dim'] for nm, r in out.items()}).to_csv(
    os.path.join(C.OUT, 'q1_perdim_supp.csv'), encoding='utf-8-sig')
print('完成', flush=True)
