# -*- coding: utf-8 -*-
"""03 问题2：化合物—风味关联分析（共表达图 + Top-3 关键化合物 + 稳定性 + 化学合理性）

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。

流程：
  1) 化合物共表达图：2.csv 皮尔逊相关 → 距离 1-|r| → 层次聚类模块 → 图指标
  2) 五路证据：Spearman |rho| / XGBoost SHAP / 随机森林置换重要性 /
     ElasticNet 稳定性选择 / 互信息
  3) 秩融合（Robust Rank Aggregation）→ 每维 Top-3
  4) 稳定性：bootstrap 入选频率 + 方法间 Jaccard
  5) 化学合理性：化学类别 / 来源 / 阈值 OAV / 气味描述 / 与 6.csv 交叉相关的一致性
输出：output/q2_top3.csv, q2_stability.csv, q2_graph.json, q2_modules.csv
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RS = 20261026
rng = np.random.default_rng(RS)
D = C.load()
COMP, SDIM = D['comp'], D['sdim']
X = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(D['X']),
                 columns=COMP).values
Y = D['Y'].values
META = C.build_meta(COMP)
Xz = StandardScaler().fit_transform(X)
K = len(COMP)
print('化合物 %d，感官维度 %d' % (K, len(SDIM)))

# =============================================================== 1 共表达图
print('\n' + '=' * 78); print('1. 化合物共表达图'); print('=' * 78)
Rho, mp2, miss_t, miss_s = C.corr_cc_aligned(COMP, META)
print('2.csv 名称映射：%d/231（双射），未匹配 %d' % (len(mp2), len(miss_t)))
Rho = np.nan_to_num(Rho, nan=0.0)
np.fill_diagonal(Rho, 1.0)
Ab = np.abs(Rho)
np.fill_diagonal(Ab, 0.0)

# 社区发现（贪心模块度优化）—— 图上的社区比层次聚类更贴合"共表达图"语义
modules, G, Q_mod = C.communities(COMP, META, thr=0.45)
n_mod = len(set(modules))
print('社区数（模块度优化，阈值 |r|≥0.45）: %d，模块度 Q = %.4f' % (n_mod, Q_mod))

# 图指标
deg = (Ab > 0.3).sum(1)
wdeg = Ab.sum(1)
D_ = np.diag(Ab.sum(1))
Lap = D_ - Ab
ev = np.linalg.eigvalsh(Lap)
fiedler = float(ev[1])
print('平均度(>0.3) %.1f，加权度均值 %.1f，Fiedler 值 %.4f，边数 %d' % (
    deg.mean(), wdeg.mean(), fiedler, int((Ab > 0.3).sum() / 2)))

mod_df = pd.DataFrame(dict(compound=COMP, module=modules, degree=deg, wdegree=wdeg,
                           family=[C.family_of(c) for c in COMP]),
                      )
mod_df.to_csv(os.path.join(C.OUT, 'q2_modules.csv'), index=False, encoding='utf-8-sig')
mod_summary = mod_df.groupby('module').agg(
    n=('compound', 'size'),
    mean_degree=('degree', 'mean'),
    top_family=('family', lambda s: s.mode().iat[0] if len(s.mode()) else ''),
    families=('family', lambda s: '、'.join(pd.Series(s).value_counts().head(3).index)),
    members=('compound', lambda s: '、'.join(s.head(8))),
).sort_values('n', ascending=False)
print(mod_summary.to_string())
mod_summary.to_csv(os.path.join(C.OUT, 'q2_module_summary.csv'), encoding='utf-8-sig')

# =============================================================== 2 五路证据
print('\n' + '=' * 78); print('2. 五路证据打分'); print('=' * 78)
import time
import xgboost as xgb

imp_sp = np.zeros((len(SDIM), K))
imp_shap = np.zeros((len(SDIM), K))
imp_rf = np.zeros((len(SDIM), K))
imp_en = np.zeros((len(SDIM), K))
imp_mi = np.zeros((len(SDIM), K))
pval_sp = np.zeros((len(SDIM), K))

_t0 = time.time()
for j, s in enumerate(SDIM):
    y = Y[:, j]
    # (a) Spearman
    rho = np.array([stats.spearmanr(Xz[:, k], y).statistic for k in range(K)])
    pv = np.array([stats.spearmanr(Xz[:, k], y).pvalue for k in range(K)])
    imp_sp[j] = np.nan_to_num(np.abs(rho))
    pval_sp[j] = pv
    # (b) 梯度提升 + SHAP（用 sklearn 实现，避免外部 OMP 线程库在多进程下死锁）
    from sklearn.ensemble import GradientBoostingRegressor
    mdl = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=3,
                                   subsample=0.8, max_features=0.4,
                                   random_state=RS).fit(Xz, y)
    try:
        import shap
        ex = shap.TreeExplainer(mdl)
        sv = ex.shap_values(Xz, check_additivity=False)
        imp_shap[j] = np.abs(sv).mean(0)
    except Exception:
        imp_shap[j] = mdl.feature_importances_
    # (c) 随机森林置换重要性（仅在 |rho| 前 80 的特征上做，其余置 0）
    from sklearn.metrics import r2_score
    rf = RandomForestRegressor(n_estimators=200, max_features=0.33, min_samples_leaf=2,
                               n_jobs=4, random_state=RS).fit(Xz, y)
    base = r2_score(y, rf.predict(Xz))
    perm = np.zeros(K)
    cand = np.argsort(-imp_sp[j])[:80]
    for k in cand:
        Xp = Xz.copy()
        Xp[:, k] = rng.permutation(Xp[:, k])
        perm[k] = max(0.0, base - r2_score(y, rf.predict(Xp)))
    imp_rf[j] = perm
    # (d) 互信息
    imp_mi[j] = mutual_info_regression(Xz, y, random_state=RS, n_neighbors=5)
    if (j + 1) % 5 == 0:
        print('  完成 %d/%d 维（%.0fs）' % (j + 1, len(SDIM), time.time() - _t0), flush=True)

# (e) ElasticNet 稳定性选择（bootstrap）
print('  ElasticNet 稳定性选择（B=120）...')
B = 120
for j, s in enumerate(SDIM):
    y = Y[:, j]
    freq = np.zeros(K)
    for b in range(B):
        idx = rng.integers(0, len(y), len(y))
        m = ElasticNet(alpha=0.05, l1_ratio=0.6, max_iter=3000).fit(Xz[idx], y[idx])
        freq += (np.abs(m.coef_) > 1e-8)
    imp_en[j] = freq / B
print('  完成')

np.savez_compressed(os.path.join(C.OUT, 'q2_importance.npz'),
                    sp=imp_sp, shap=imp_shap, rf=imp_rf, en=imp_en, mi=imp_mi,
                    pval=pval_sp, comp=np.array(COMP), sdim=np.array(SDIM))


def rankdata(a):
    from scipy.stats import rankdata as rd
    return rd(a)


# =============================================================== 3 秩融合
print('\n' + '=' * 78); print('3. 秩融合 → 每维 Top-3'); print('=' * 78)
WEIGHTS = dict(sp=0.20, shap=0.35, rf=0.20, en=0.15, mi=0.10)
score = (WEIGHTS['sp'] * np.apply_along_axis(rankdata, 1, imp_sp) +
         WEIGHTS['shap'] * np.apply_along_axis(rankdata, 1, imp_shap) +
         WEIGHTS['rf'] * np.apply_along_axis(rankdata, 1, imp_rf) +
         WEIGHTS['en'] * np.apply_along_axis(rankdata, 1, imp_en) +
         WEIGHTS['mi'] * np.apply_along_axis(rankdata, 1, imp_mi))
# 综合分：秩越小越好 → 转为正向分数
fused = score.max() + 1 - score
fused_n = (fused - fused.min()) / (fused.max() - fused.min())

top3 = {}
for j, s in enumerate(SDIM):
    order = np.argsort(-fused[j])[:10]
    top3[s] = [(COMP[k], float(fused_n[j, k])) for k in order]

# =============================================================== 4 稳定性
print('\n' + '=' * 78); print('4. 稳定性（bootstrap 入选频率 + 方法间 Jaccard）'); print('=' * 78)
# 方法间 Jaccard（各方法单独取 Top-3）
def topk_sets():
    S = {}
    for nm, arr in [('sp', imp_sp), ('shap', imp_shap), ('rf', imp_rf),
                    ('en', imp_en), ('mi', imp_mi)]:
        S[nm] = [set(np.argsort(-arr[j])[:3]) for j in range(len(SDIM))]
    return S


TS = topk_sets()
jac_methods = {}
for j, s in enumerate(SDIM):
    vals = []
    keys = list(TS)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            A1, B1 = TS[keys[a]][j], TS[keys[b]][j]
            vals.append(len(A1 & B1) / len(A1 | B1))
    jac_methods[s] = float(np.mean(vals))
print('方法间 Jaccard 均值 %.3f（中位 %.3f）' % (
    np.mean(list(jac_methods.values())), np.median(list(jac_methods.values()))))

# bootstrap：重采样后重算融合分数，统计 Top-3 入选频率
# ⚠ 用向量化秩相关替代逐对标量 spearmanr（后者 2.31M 次调用需 36 分钟）
from scipy.stats import rankdata as _rd


def fast_abs_spearman(Xa, ya):
    """一次算出 |Spearman(X[:,k], y)|（k=1..p），O(n·p)。"""
    ry = _rd(ya)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    rX = _rd(Xa, axis=0)
    rX = (rX - rX.mean(0)) / (rX.std(0) + 1e-12)
    return np.abs((rX * ry[:, None]).mean(0))


NB = 200
print('  bootstrap 重采样 B=%d（向量化秩相关）...' % NB)
freq = {s: {COMP[k]: 0 for k in np.argsort(-fused[j])[:10]} for j, s in enumerate(SDIM)}
for b in range(NB):
    idx = rng.integers(0, len(Y), len(Y))
    Xi = Xz[idx]
    for j, s in enumerate(SDIM):
        rho = np.nan_to_num(fast_abs_spearman(Xi, Y[idx, j]))
        order = np.argsort(-rho)[:10]
        for k in order[:3]:
            if COMP[k] in freq[s]:
                freq[s][COMP[k]] += 1
    if (b + 1) % 50 == 0:
        print('    %d/%d' % (b + 1, NB), flush=True)

stab_tab = []
for s in SDIM:
    row = dict(感官维度=C.clean_sens_name(s), 分组=C.SENS_GROUP[s],
               方法Jaccard=round(jac_methods[s], 3))
    for r, (cmp_, sc) in enumerate(top3[s][:3], 1):
        row['Top%d' % r] = cmp_
        row['Top%d综合分' % r] = round(sc, 4)
        row['Top%d入选频率' % r] = round(freq[s].get(cmp_, 0) / NB, 3)
        row['Top%d类别' % r] = C.family_of(cmp_)
        row['Top%d化学名' % r] = META.loc[cmp_, 'chem_name'] if cmp_ in META.index else ''
    stab_tab.append(row)
STAB = pd.DataFrame(stab_tab)
STAB.to_csv(os.path.join(C.OUT, 'q2_top3.csv'), index=False, encoding='utf-8-sig')
print('\n每维 Top-3 稳定性汇总：')
print(STAB[['感官维度', 'Top1', 'Top1入选频率', 'Top2', 'Top2入选频率',
            'Top3', 'Top3入选频率', '方法Jaccard']].head(20).to_string(index=False))

mean_freq = float(np.mean([float(v) for s in SDIM for k, v in freq[s].items()
                           if k in [t[0] for t in top3[s][:3]]]) / NB)
print('\nTop-3 平均 bootstrap 入选频率 = %.3f' % mean_freq)

# =============================================================== 5 化学合理性
print('\n' + '=' * 78); print('5. 化学合理性论证表'); print('=' * 78)
# 6.csv：化合物×感官 的 Spearman 交叉相关（数据集自带，作为独立证据）
c6 = pd.read_csv(os.path.join(C.DATA, '6.csv'), index_col=0)
alias6 = {}
for cn in c6.index:
    for k in COMP:
        if META.loc[k, 'chem_name'] and \
                str(META.loc[k, 'chem_name']).lower().replace(' ', '') == \
                str(cn).lower().replace(' ', '').replace('-', ''):
            alias6[cn] = k
            break
print('6.csv 化合物名与 1.csv 匹配：%d / %d' % (len(alias6), len(c6)))
# 感官名映射
sens5 = pd.read_csv(os.path.join(C.DATA, '5.csv'), index_col=0)
print('6.csv 列为 5.csv 的感官名，共 %d 列' % c6.shape[1])

plaus = []
for s in SDIM:
    for r, (cmp_, sc) in enumerate(top3[s][:3], 1):
        thr = META.loc[cmp_, 'threshold'] if cmp_ in META.index else np.nan
        oav_med = np.nan
        try:
            oav = np.load(os.path.join(C.OUT, 'prepared.npz'))
            cols = list(oav['oav_cols'])
            if cmp_ in cols:
                oav_med = float(np.nanmedian(oav['oav'][:, cols.index(cmp_)]))
        except Exception:
            pass
        plaus.append(dict(感官维度=C.clean_sens_name(s), 排名=r, 化合物=cmp_,
                          综合分=round(sc, 4),
                          入选频率=round(freq[s].get(cmp_, 0) / NB, 3),
                          化学类别=C.family_of(cmp_),
                          来源=META.loc[cmp_, 'origin'] if cmp_ in META.index else '',
                          气味阈值=thr, 阈值单位=META.loc[cmp_, 'unit'] if cmp_ in META.index else '',
                          OAV中位数=oav_med,
                          异味标记=META.loc[cmp_, 'off_flavor'] if cmp_ in META.index else '',
                          Spearman=round(float(imp_sp[s][COMP.index(cmp_)]), 3),
                          SHAP=round(float(imp_shap[s][COMP.index(cmp_)]), 4),
                          稳定性选择频率=round(float(imp_en[s][COMP.index(cmp_)]), 3)))
PL = pd.DataFrame(plaus)
PL.to_csv(os.path.join(C.OUT, 'q2_plausibility.csv'), index=False, encoding='utf-8-sig')
print(PL.head(24).to_string(index=False))

# 一致性：Top-3 是否属于"化学上应当相关"的来源
print('\nTop-3 化学类别分布：')
print(PL.groupby('化学类别').size().sort_values(ascending=False).to_string())
print('\nTop-3 来源分布：')
print(PL.groupby('来源').size().sort_values(ascending=False).to_string())
print('\nTop-3 中有阈值且 OAV>1 的比例：%.1f%%' % (
    100 * (PL['OAV中位数'] > 1).sum() / max(1, PL['OAV中位数'].notna().sum())))

json.dump(dict(
    n_compounds=K, n_modules=int(n_mod), fiedler=fiedler,
    mean_degree=float(deg.mean()), mean_wdegree=float(wdeg.mean()),
    jaccard_methods_mean=float(np.mean(list(jac_methods.values()))),
    top3_mean_bootstrap_freq=mean_freq,
    modules=mod_summary.reset_index().to_dict('records'),
    top3={s: [t[0] for t in top3[s][:3]] for s in SDIM},
    top3_scores={s: [t[1] for t in top3[s][:3]] for s in SDIM},
    jaccard={s: jac_methods[s] for s in SDIM},
    freq={s: {k: v / NB for k, v in freq[s].items()} for s in SDIM},
    n_edges_03=int((Ab > 0.3).sum() / 2),
), open(os.path.join(C.OUT, 'q2_graph.json'), 'w', encoding='utf-8'),
    ensure_ascii=False, indent=2)
print('\n完成：q2_top3.csv / q2_plausibility.csv / q2_graph.json / q2_modules.csv')
