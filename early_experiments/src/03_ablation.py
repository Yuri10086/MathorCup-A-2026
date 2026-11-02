# -*- coding: utf-8 -*-
"""03_ablation.py —— 特征工程/预处理消融与"基线是否被做坏"的诊断

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek）。

目的有三：
  1) 交代每个处理步骤（量纲统一、缩尾、标准化、缺失指示、封底指示、图扩散、风格哑变量）
     各自贡献多少 R²；
  2) 复现"把线性基线做坏"的经典做法（IQR 缩放 + 固定 α=30），说明为什么不能用它当基线；
  3) 给出同一 CV 协议下的诚实基线（标准化 + RidgeCV 自动选 α）。

两个模型：RidgeCV（线性，5 折 × 3 重复）与 HistGBR（树，5 折 × 1 重复，n_iter=200）。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q1lib as C

T0 = time.time()
D = C.load()
COMP, SDIM = D['comp'], D['sdim']
Y = D['Y'].values.astype(float)
STYLE = D['style']
SCALE, _ = C.colscale()
XH = C.harmonize(D['X'], COMP, SCALE)
XR = D['X'].copy()                       # 未做量纲统一
STRAT = C.strat_labels(STYLE)
CATS = sorted(set(STYLE))
FLOOR_COLS, _ = C.censoring(D['X'], COMP, 0.05)
LOG = open(os.path.join(C.OUT, 'q1_ablation.log'), 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    print(s, file=LOG, flush=True)


# ---------------------------------------------------------------- 可配置预处理
def prep2(Xtr, Xte, opt):
    """opt: harm / winsor / scale('std'|'iqr') / miss / floor / graph / style"""
    comp = list(Xtr.columns)
    A = Xtr[comp].values.astype(float)
    B = Xte[comp].values.astype(float)
    missA, missB = np.isnan(A), np.isnan(B)
    eqA = np.isclose(A, C.FLOOR, atol=1e-9).astype(float)
    eqB = np.isclose(B, C.FLOOR, atol=1e-9).astype(float)

    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(A)
    A, B = imp.transform(A), imp.transform(B)
    if opt.get('winsor', True):
        lo, hi = np.nanpercentile(A, [1, 99], axis=0)
        A, B = np.clip(A, lo, hi), np.clip(B, lo, hi)
    if opt.get('scale', 'std') == 'iqr':
        med = np.nanmedian(A, 0)
        iqr = np.nanpercentile(A, 75, 0) - np.nanpercentile(A, 25, 0)
        bad = iqr <= 0
        iqr[bad] = np.nanstd(A, 0)[bad] + 1e-9
        A, B = (A - med) / iqr, (B - med) / iqr
    else:
        mu, sd = np.nanmean(A, 0), np.nanstd(A, 0)
        sd = np.where(sd > 0, sd, 1.0)
        A, B = (A - mu) / sd, (B - mu) / sd

    if opt.get('miss', True):
        kc = np.where(missA.any(0))[0]
        if len(kc):
            A = np.hstack([A, missA[:, kc].astype(float)])
            B = np.hstack([B, missB[:, kc].astype(float)])
    if opt.get('floor', True):
        kc = np.where(eqA.mean(0) >= 0.05)[0]
        if len(kc):
            A = np.hstack([A, eqA[:, kc]])
            B = np.hstack([B, eqB[:, kc]])
    if opt.get('graph', False):
        S = opt['graph_op']
        p0 = len(comp)
        A = np.hstack([A, A[:, :p0] @ S.T])
        B = np.hstack([B, B[:, :p0] @ S.T])
    if opt.get('style', False):
        otr = np.zeros((len(A), len(CATS)))
        ote = np.zeros((len(B), len(CATS)))
        si = {c: i for i, c in enumerate(CATS)}
        for i, s in enumerate(opt['style_tr']):
            otr[i, si[s]] = 1.0
        for i, s in enumerate(opt['style_te']):
            ote[i, si[s]] = 1.0
        A = np.hstack([A, otr])
        B = np.hstack([B, ote])
    return A, B


def fit_ridgecv(a, b, c):
    m = MultiOutputRegressor(RidgeCV(alphas=np.logspace(-1, 4, 40)), n_jobs=1).fit(a, b)
    return np.clip(m.predict(c), -6, 8)


def fit_ridge_fixed30(a, b, c):
    return Ridge(alpha=30.0).fit(a, b).predict(c)


def fit_hgb(a, b, c):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_depth=3,
                                      min_samples_leaf=8, l2_regularization=2.0,
                                      max_features=0.6, early_stopping=False,
                                      random_state=C.RS), n_jobs=1).fit(a, b).predict(c)


def run(models, cases, repeats):
    """models: {name: fn}；cases: {case: (Xdf, opt)}；返回 DataFrame。"""
    from sklearn.model_selection import StratifiedKFold
    rows = []
    for mn, fn in models.items():
        for cn, (Xdf, opt0) in cases.items():
            preds = np.zeros((repeats, len(Y), len(SDIM)))
            t0 = time.time()
            for rep in range(repeats):
                skf = StratifiedKFold(n_splits=C.N_SPLITS, shuffle=True,
                                      random_state=C.RS + rep)
                for tr, te in skf.split(Xdf, STRAT):
                    opt = dict(opt0)
                    opt.update(style_tr=STYLE[tr], style_te=STYLE[te])
                    A, B = prep2(Xdf.iloc[tr], Xdf.iloc[te], opt)
                    preds[rep][te] = fn(A, Y[tr], B)
            pred = preds.mean(0)
            r = C.per_dim_r2(Y, pred)
            rows.append(dict(model=mn, case=cn, mean_r2=float(np.nanmean(r)),
                             median_r2=float(np.median(r)),
                             min_r2=float(np.nanmin(r)), n_pos=int((r >= 0).sum()),
                             seconds=round(time.time() - t0, 1)))
            log('%-12s %-22s meanR2=%.4f median=%.4f min=%.4f pos=%2d/50  %5.0fs' % (
                mn, cn, rows[-1]['mean_r2'], rows[-1]['median_r2'],
                rows[-1]['min_r2'], rows[-1]['n_pos'], rows[-1]['seconds']))
    return pd.DataFrame(rows)


SOP = C.build_graph_op(COMP)
CASES = {
    'A 完整方案（本文）':      (XH, dict(harm=True, winsor=True, scale='std', miss=True,
                                       floor=True)),
    'B 无量纲统一(log10)':     (XR, dict(harm=False, winsor=True, scale='std', miss=True,
                                       floor=True)),
    'C 无缩尾':                (XH, dict(harm=True, winsor=False, scale='std', miss=True,
                                       floor=True)),
    'D 无封底指示':            (XH, dict(harm=True, winsor=True, scale='std', miss=True,
                                       floor=False)),
    'E 无缺失指示':            (XH, dict(harm=True, winsor=True, scale='std', miss=False,
                                       floor=True)),
    'F +共表达图扩散':         (XH, dict(harm=True, winsor=True, scale='std', miss=True,
                                       floor=True, graph=True, graph_op=SOP)),
    'G +风格哑变量':           (XH, dict(harm=True, winsor=True, scale='std', miss=True,
                                       floor=True, style=True)),
    'H 复现:IQR+固定α=30':     (XR, dict(harm=False, winsor=True, scale='iqr', miss=True,
                                       floor=False)),
    'I IQR+固定α=30(已统一)':  (XH, dict(harm=True, winsor=True, scale='iqr', miss=True,
                                       floor=False)),
}

log('=' * 78)
log('消融：RidgeCV（5 折 × 3 重复）')
log('=' * 78)
R1 = run({'RidgeCV': fit_ridgecv}, CASES, C.N_REPEATS)
log('\n' + '=' * 78)
log('消融：IQR+固定α=30 的 Ridge（对照组，5 折 × 3 重复）')
log('=' * 78)
R2 = run({'Ridge-fixed30': fit_ridge_fixed30},
         {k: v for k, v in CASES.items() if k.startswith(('A', 'B', 'H', 'I'))},
         C.N_REPEATS)

log('\n' + '=' * 78)
log('消融：HistGBR（5 折 × 1 重复，n_iter=200）')
log('=' * 78)
R3 = run({'HistGBR': fit_hgb}, {k: v for k, v in CASES.items() if k[0] in 'ABCDFG'}, 1)

ALL = pd.concat([R1, R2, R3], ignore_index=True)
ALL.to_csv(os.path.join(C.OUT, 'q1_ablation.csv'), index=False, encoding='utf-8-sig')

log('\n' + '=' * 78)
log('汇总：各步骤相对"完整方案"的边际增益（RidgeCV，mean R²）')
log('=' * 78)
base = float(ALL[(ALL.model == 'RidgeCV') &
                 (ALL.case.str.startswith('A'))]['mean_r2'].iloc[0])
for _, r in ALL[ALL.model == 'RidgeCV'].iterrows():
    log('%-24s %.4f   Δ=%+.4f' % (r['case'], r['mean_r2'], r['mean_r2'] - base))
log('\n→ 若 Δ 全部落在 ±0.01 内，说明特征工程对线性模型的边际贡献有限，')
log('  真正的差距来自"标准化 + 自动选 α"这一基线修正，而不是复杂结构。')

json.dump(dict(base_full=base,
               rows=ALL.to_dict(orient='records')),
          open(os.path.join(C.OUT, 'q1_ablation.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
log('\n完成，总耗时 %.0fs' % (time.time() - T0))
