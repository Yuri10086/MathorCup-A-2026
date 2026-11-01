# -*- coding: utf-8 -*-
"""为填补页面空白补充的可引用数字：感官分块相关结构 + OAV 代表条目 + CV 协议。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r'E:\dsh_work\MC2026A\src')
import common as C

D = C.load()
Y, SDIM = D['Y'], D['sdim']
META = C.build_meta(D['comp'])

# ---------- 1. 感官分块结构：组内 vs 组间平均 |rho|
cs = Y.corr(method='spearman').values
grp = np.array([C.SENS_GROUP[c] for c in SDIM])
iu = np.triu_indices(len(SDIM), 1)
same = grp[iu[0]] == grp[iu[1]]
r = np.abs(cs[iu])
print('=== 感官分块结构 ===')
print('全体 平均|rho| = %.4f' % r.mean())
print('组内 平均|rho| = %.4f（对数 %d）' % (r[same].mean(), same.sum()))
print('组间 平均|rho| = %.4f（对数 %d）' % (r[~same].mean(), (~same).sum()))
print('比值 组内/组间 = %.2f' % (r[same].mean() / r[~same].mean()))
for g in ['Aroma', 'Flavour', 'Mouthfeel', 'Fault']:
    idx = np.where(grp == g)[0]
    if len(idx) < 2:
        continue
    sub = np.abs(cs[np.ix_(idx, idx)])
    m = np.triu_indices(len(idx), 1)
    print('  %-10s 组内平均|rho| = %.3f（%d 维，%d 对）' % (g, sub[m].mean(), len(idx), sub[m].size))

# 最强的一对
k = int(np.argmax(r))
print('最强一对: %s <-> %s  rho=%.3f' % (C.clean_sens_name(SDIM[iu[0][k]]),
                                        C.clean_sens_name(SDIM[iu[1][k]]), cs[iu][k]))

# ---------- 2. OAV 代表条目
X = D['X'].values.astype(float)
OAV, UNIT = C.oav_matrix(X, D['comp'], META)
tab = pd.DataFrame({
    'name': [META.loc[c, 'chem_name'] or c for c in OAV.columns],
    'unit': [META.loc[c, 'unit'] for c in OAV.columns],
    'thr': [META.loc[c, 'threshold'] for c in OAV.columns],
    'src': [META.loc[c, 'origin'] for c in OAV.columns],
    'med': OAV.median().values,
    'gt1': (OAV > 1).mean().values,
}, index=OAV.columns).sort_values('med', ascending=False)
print('\n=== OAV 代表条目（前 12）===')
print(tab.head(12).to_string(float_format=lambda x: '%.3f' % x))
print('\n化合物总数 %d；OAV>1 占比 >50%% 的有 %d 个' % (
    len(tab), int((tab['gt1'] > 0.5).sum())))

# ---------- 3. CV 协议数字
q1 = pd.read_csv(os.path.join(C.OUT, 'q1_metrics_perdim.csv'))
print('\n=== CV 协议 ===')
print('折数 5 × 重复 3 = 15 次训练；每次 200 训练/50 验证')
print('50 维平均 R2 = %.4f，标准差 = %.4f' % (q1['stack_r2'].mean(), q1['stack_r2'].std()))
rep = np.load(os.path.join(C.OUT, 'q1_oof.npz'))
print('重复间预测标准差：中位 %.4f，90%% 分位 %.4f' % (
    np.median(rep['spread']), np.percentile(rep['spread'], 90)))
print('残留 37 个生成样本')
