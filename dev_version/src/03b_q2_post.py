# -*- coding: utf-8 -*-
"""03b 问题二 后处理：共表达图 + 秩融合 + 稳定性 + 化学合理性

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。

⚠ 本轮修正了一个方向性 bug：rankdata 使“越重要 → 秩越大”，而此前排序时又取了
负号，导致把**最不重要**的化合物当成 Top-3（表现为 Top-3 的化学构成与随机分布一致、
bootstrap 入选频率为 0）。修正后 Top-3 恢复了化学语义。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RS = 20261026
rng = np.random.default_rng(RS)
D = C.load()
COMP, SDIM = D['comp'], D['sdim']
META = C.build_meta(COMP)
X = SimpleImputer(strategy='median').fit_transform(D['X'].values.astype(float))
Y = D['Y'].values
Xz = StandardScaler().fit_transform(X)
K = len(COMP)

# ============================================================ 1 共表达图
print('1. 共表达图与社区发现', flush=True)
mods, G, Qmod = C.communities(COMP, META, thr=0.45)
deg = np.array([G.degree(i) for i in range(len(COMP))])
wdeg = np.array([sum(d['weight'] for _, _, d in G.edges(i, data=True))
                 for i in range(len(COMP))])
n_mod = len(set(mods))
print('   节点 %d，边 %d，社区 %d，模块度 Q=%.4f，平均度 %.1f' % (
    G.number_of_nodes(), G.number_of_edges(), n_mod, Qmod, deg.mean()))

mod_df = pd.DataFrame(dict(compound=COMP, module=mods, degree=deg, wdegree=wdeg,
                           family=[C.family_of(c) for c in COMP]))
mod_df.to_csv(os.path.join(C.OUT, 'q2_modules.csv'), index=False, encoding='utf-8-sig')
mod_summary = mod_df.groupby('module').agg(
    n=('compound', 'size'), mean_degree=('degree', 'mean'),
    top_family=('family', lambda s: s.mode().iat[0] if len(s.mode()) else ''),
    families=('family', lambda s: '、'.join(pd.Series(s).value_counts().head(3).index)),
    members=('compound', lambda s: '、'.join(s.head(8))),
).sort_values('n', ascending=False)
mod_summary.to_csv(os.path.join(C.OUT, 'q2_module_summary.csv'), encoding='utf-8-sig')
print(mod_summary.head(8)[['n', 'families', 'members']].to_string())

# ============================================================ 2 秩融合
print('\n2. 五路证据秩融合', flush=True)
imp = np.load(os.path.join(C.OUT, 'q2_importance.npz'))
imp_sp, imp_shap, imp_rf, imp_en, imp_mi = (imp['sp'], imp['shap'], imp['rf'],
                                            imp['en'], imp['mi'])

# ⚠ 候选集排除“理化参数”（pH/色度/CO2/乙醇/糖/蛋白/铁/BHT 等 15 列）：
#   实测若纳入，它们会占据 35% 的 Top-3 名额（如 CO2.PSI. 成为“酒花香-柑橘”的
#   首位），原因是它们与风格/工艺强共线，是风格代理变量而非风味驱动物。
#   问题一（预测）仍全部使用；问题二（归因）只保留风味化合物。
IS_FLAVOR = np.array([C.family_of(c) != '理化参数' for c in COMP])
print('   候选集：风味化合物 %d 个（排除理化参数 %d 个）' % (
    IS_FLAVOR.sum(), (~IS_FLAVOR).sum()))

W = dict(sp=0.20, shap=0.35, rf=0.20, en=0.15, mi=0.10)


def masked_rank(a):
    r = rankdata(a)
    r[~IS_FLAVOR] = -1.0            # 非候选置最低
    return r


score = (W['sp'] * np.apply_along_axis(masked_rank, 1, imp_sp) +
         W['shap'] * np.apply_along_axis(masked_rank, 1, imp_shap) +
         W['rf'] * np.apply_along_axis(masked_rank, 1, imp_rf) +
         W['en'] * np.apply_along_axis(masked_rank, 1, imp_en) +
         W['mi'] * np.apply_along_axis(masked_rank, 1, imp_mi))
fused_n = (score - score.min()) / (score.max() - score.min())   # 越大越重要
top3 = {}
for j, s in enumerate(SDIM):
    order = np.argsort(-fused_n[j])[:10]
    top3[s] = [(COMP[k], float(fused_n[j, k])) for k in order]
for s in ['A_hops_citrus', 'A_esters_all', 'diacetyl', 'overall']:
    print('   %-14s → %s' % (s, [t[0] for t in top3[s][:3]]))

# ============================================================ 3 Jaccard
TS = {}
for nm, arr in [('sp', imp_sp), ('shap', imp_shap), ('rf', imp_rf),
                ('en', imp_en), ('mi', imp_mi)]:
    TS[nm] = [set(np.argsort(-np.where(IS_FLAVOR, arr[j], -np.inf))[:3])
              for j in range(len(SDIM))]
jac = {}
for j, s in enumerate(SDIM):
    vals = []
    keys = list(TS)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            A1, B1 = TS[keys[a]][j], TS[keys[b]][j]
            vals.append(len(A1 & B1) / len(A1 | B1))
    jac[s] = float(np.mean(vals))
print('\n3. 方法间 Jaccard 均值 %.3f' % np.mean(list(jac.values())), flush=True)


# ============================================================ 4 bootstrap
def fast_abs_spearman(Xa, ya):
    ry = rankdata(ya); ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    rX = rankdata(Xa, axis=0); rX = (rX - rX.mean(0)) / (rX.std(0) + 1e-12)
    return np.abs((rX * ry[:, None]).mean(0))


NB, TOPK = 200, 10
print('4. bootstrap B=%d（入选口径：重采样后 Spearman 绝对秩相关前 %d）...' % (NB, TOPK),
      flush=True)
freq = {s: {c: 0 for c in COMP} for s in SDIM}
t0 = time.time()
for b in range(NB):
    idx = rng.integers(0, len(Y), len(Y))
    Xi = Xz[idx]
    for j, s in enumerate(SDIM):
        rho = np.nan_to_num(fast_abs_spearman(Xi, Y[idx, j]))
        rho = np.where(IS_FLAVOR, rho, -np.inf)
        for k in np.argsort(-rho)[:TOPK]:
            freq[s][COMP[k]] += 1
    if (b + 1) % 50 == 0:
        print('    %d/%d（%.0fs）' % (b + 1, NB, time.time() - t0), flush=True)

# ============================================================ 5 汇总
rows = []
for s in SDIM:
    row = dict(感官维度=C.clean_sens_name(s), 分组=C.SENS_GROUP[s],
               方法Jaccard=round(jac[s], 3))
    for r, (cmp_, sc) in enumerate(top3[s][:3], 1):
        row['Top%d' % r] = cmp_
        row['Top%d综合分' % r] = round(sc, 4)
        row['Top%d入选频率' % r] = round(freq[s][cmp_] / NB, 3)
        row['Top%d类别' % r] = C.family_of(cmp_)
        row['Top%d化学名' % r] = META.loc[cmp_, 'chem_name'] if cmp_ in META.index else ''
    rows.append(row)
STAB = pd.DataFrame(rows)
STAB.to_csv(os.path.join(C.OUT, 'q2_top3.csv'), index=False, encoding='utf-8-sig')
top1_freq = float(np.mean(STAB['Top1入选频率'].values))
top3_freq = float(np.mean(STAB[['Top1入选频率', 'Top2入选频率', 'Top3入选频率']].values))
print('\nTop-1 平均入选频率 %.3f；Top-2/Top-3 %.3f' % (
    top1_freq, float(np.mean(STAB[['Top2入选频率', 'Top3入选频率']].values))))

# ============================================================ 6 化学合理性
OAV, _U = C.oav_matrix(X, COMP, META)
plaus = []
for j, s in enumerate(SDIM):
    for r, (cmp_, sc) in enumerate(top3[s][:3], 1):
        k = COMP.index(cmp_)
        thr = float(META.loc[cmp_, 'threshold']) if cmp_ in META.index else np.nan
        oav_med = float(np.nanmedian(OAV[cmp_])) if cmp_ in OAV.columns else np.nan
        plaus.append(dict(感官维度=C.clean_sens_name(s), 排名=r, 化合物=cmp_,
                          综合分=round(sc, 4), 入选频率=round(freq[s][cmp_] / NB, 3),
                          化学类别=C.family_of(cmp_),
                          来源=(META.loc[cmp_, 'origin'] if cmp_ in META.index else ''),
                          气味阈值=thr,
                          阈值单位=(META.loc[cmp_, 'unit'] if cmp_ in META.index else ''),
                          OAV中位数=oav_med,
                          异味标记=(META.loc[cmp_, 'off_flavor'] if cmp_ in META.index else ''),
                          Spearman=round(float(imp_sp[j][k]), 3),
                          SHAP=round(float(imp_shap[j][k]), 4),
                          稳定性选择频率=round(float(imp_en[j][k]), 3),
                          所属社区=int(mods[k])))
PL = pd.DataFrame(plaus)
PL.to_csv(os.path.join(C.OUT, 'q2_plausibility.csv'), index=False, encoding='utf-8-sig')
print('\nTop-3 化学类别：')
print(PL.groupby('化学类别').size().sort_values(ascending=False).to_string())
print('Top-3 来源：')
print(PL.groupby('来源').size().sort_values(ascending=False).to_string())
n_oav = int(PL['OAV中位数'].notna().sum())
n_gt1 = int((PL['OAV中位数'] > 1).sum())
print('可算 OAV 的 Top-3 条目 %d 个，OAV>1 的 %d 个（%.1f%%）' % (
    n_oav, n_gt1, 100 * n_gt1 / max(1, n_oav)))
# 同模块一致性
PL['社区'] = PL['所属社区']
same_mod = 0
for s in SDIM:
    sub = PL[PL['感官维度'] == C.clean_sens_name(s)]
    if len(sub) >= 2:
        vc = sub['所属社区'].value_counts()
        if vc.iloc[0] >= 2:
            same_mod += 1
print('Top-3 中至少两个落在同一社区的维度数：%d / %d（%.0f%%）' % (
    same_mod, len(SDIM), 100 * same_mod / len(SDIM)))

gj = dict(
    n_compounds=K, n_modules=int(n_mod), modularity=float(Qmod),
    edges=int(G.number_of_edges()), mean_degree=float(deg.mean()),
    mean_wdegree=float(wdeg.mean()),
    jaccard_methods_mean=float(np.mean(list(jac.values()))),
    top1_freq=top1_freq, top3_freq=top3_freq,
    top3={s: [t[0] for t in top3[s][:3]] for s in SDIM},
    top3_scores={s: [t[1] for t in top3[s][:3]] for s in SDIM},
    jaccard=jac,
    freq={s: {c: freq[s][c] / NB for c in COMP if freq[s][c] > 0} for s in SDIM},
    bootstrap=dict(B=NB, topk=TOPK, criterion='重采样后 Spearman 绝对秩相关前 10'),
    n_oav=int(n_oav), n_oav_gt1=int(n_gt1),
    oav_gt1_ratio=float(n_gt1 / max(1, n_oav)),
    same_module_ratio=float(same_mod / len(SDIM)),
    modules=mod_summary.reset_index().to_dict('records'),
)
json.dump(gj, open(os.path.join(C.OUT, 'q2_graph.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('\n已写出 q2_graph.json / q2_top3.csv / q2_plausibility.csv / q2_modules.csv')
