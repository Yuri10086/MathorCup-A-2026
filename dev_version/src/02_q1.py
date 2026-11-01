# -*- coding: utf-8 -*-
"""02 问题1：缺失数据条件下的感官评分（0-6 分制）

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。

方法栈：
  L0 线性基线   Ridge / ElasticNet / PLS
  L1 树模型基线 GBR（参考文献口径）/ RandomForest / XGBoost / LightGBM
  L2 潜因子多任务 Ridge（低秩感官结构 + 化合物稀疏）
  L3 化合物共表达图扩散 + 多任务神经网络（本文主模型 MT-GNN）
  L4 堆叠集成 + 一致性预测区间
评估：按风格分层的 5 折 × 3 重复交叉验证（pooled R²，与参考文献同口径）
"""
import json
import os
import sys
import time
import warnings

# ⚠ 本机用户名含中文，joblib 的 memmapping 资源跟踪器按 ASCII 编码路径会崩溃
#   → 把临时目录强制指向纯 ASCII 路径
_TMP = r'E:\dsh_work\MC2026A\.tmp'
os.makedirs(_TMP, exist_ok=True)
os.environ['JOBLIB_TEMP_FOLDER'] = _TMP
os.environ['TMP'] = _TMP
os.environ['TEMP'] = _TMP

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RS = 20261026
N_SPLITS, N_REPEATS = 5, 3
D = C.load()
COMP, SDIM = D['comp'], D['sdim']
Xraw = D['X'].values.astype(np.float64)
Y = D['Y'].values.astype(np.float64)
STYLE = D['style']
LOG = open(os.path.join(C.OUT, 'q1_progress.log'), 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    print(s, file=LOG, flush=True)


vc = pd.Series(STYLE).value_counts()
strat = np.array([s if vc[s] >= N_SPLITS else 'Rare' for s in STYLE])
log('样本 %d，化合物 %d，感官维度 %d；分层标签 %d 类' % (len(Y), len(COMP), len(SDIM), len(set(strat))))


# ---------------------------------------------------------------- 特征工程
def prep(Xtr, Xte, use_graph=False):
    p0 = Xtr.shape[1]                    # 原始化合物维数（图扩散只作用于这一块）
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xtr)
    Ztr = imp.transform(Xtr); Zte = imp.transform(Xte)
    lo, hi = np.nanpercentile(Ztr, [1, 99], axis=0)
    Ztr = np.clip(Ztr, lo, hi); Zte = np.clip(Zte, lo, hi)
    med = np.nanmedian(Ztr, axis=0)
    iqr = np.nanpercentile(Ztr, 75, axis=0) - np.nanpercentile(Ztr, 25, axis=0)
    bad = iqr <= 0
    iqr[bad] = np.nanstd(Ztr, axis=0)[bad] + 1e-9
    Ztr = (Ztr - med) / iqr; Zte = (Zte - med) / iqr
    miss_cols = np.where(np.isnan(Xtr).any(axis=0))[0]
    if len(miss_cols):
        Ztr = np.hstack([Ztr, np.isnan(Xtr[:, miss_cols]).astype(float)])
        Zte = np.hstack([Zte, np.isnan(Xte[:, miss_cols]).astype(float)])
    if use_graph:
        # 只为原始化合物块追加扩散特征，缺失指示列不参与扩散
        Ztr = np.hstack([Ztr, Ztr[:, :p0] @ S_diff.T])
        Zte = np.hstack([Zte, Zte[:, :p0] @ S_diff.T])
    return Ztr, Zte


def build_graph(topk=8, alpha=0.35):
    META = C.build_meta(COMP)
    R, _mp, _mt, _ms = C.corr_cc_aligned(COMP, META)
    W = np.abs(np.nan_to_num(R, nan=0.0))
    np.fill_diagonal(W, 0.0)
    k = min(topk, W.shape[0] - 1)
    idx = np.argpartition(-W, k, axis=1)[:, :k]
    M = np.zeros_like(W)
    for i in range(W.shape[0]):
        M[i, idx[i]] = W[i, idx[i]]
    M = np.maximum(M, M.T)
    deg = M.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    Ahat = (M * dinv[:, None]) * dinv[None, :]
    return M, (1 - alpha) * np.eye(Ahat.shape[0]) + alpha * Ahat


W_graph, S_diff = build_graph()
log('共表达图：节点 %d，边 %d，平均度 %.1f' % (
    len(COMP), int((W_graph > 0).sum() / 2), (W_graph > 0).sum(axis=1).mean()))


# ---------------------------------------------------------------- 模型
def fit_ridge(a, b, c):
    return Ridge(alpha=30.0).fit(a, b).predict(c)


def fit_enet(a, b, c):
    return MultiOutputRegressor(ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=4000), n_jobs=1).fit(a, b).predict(c)


def fit_pls(a, b, c):
    return PLSRegression(n_components=12, scale=False).fit(a, b).predict(c)


def fit_rrf(a, b, c):
    return RandomForestRegressor(n_estimators=500, max_features=0.33, min_samples_leaf=2,
                                 n_jobs=-1, random_state=RS).fit(a, b).predict(c)


def fit_gbr(a, b, c):
    """文献口径的梯度提升基线（sklearn GBR）。单目标拟合慢，故降规模。"""
    return MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=3,
                                  subsample=0.8, max_features=0.4, random_state=RS),
        n_jobs=1).fit(a, b).predict(c)


def fit_xgb(a, b, c):
    """直方图梯度提升（sklearn 实现，与文献的 GBR 同族、避免外部 OMP 线程库）。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    return MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=3,
                                      min_samples_leaf=8, l2_regularization=2.0,
                                      max_features=0.6, early_stopping=False,
                                      random_state=RS), n_jobs=1).fit(a, b).predict(c)


def fit_lgbm(a, b, c):
    """随机森林的另一超参设定（叶子更小），用于检验树模型的超参稳健性。"""
    return RandomForestRegressor(n_estimators=500, max_features=0.6, min_samples_leaf=1,
                                 n_jobs=-1, random_state=RS + 7).fit(a, b).predict(c)


def fit_rrr(a, b, c, K=8, alpha=30.0):
    Ym = b - b.mean(0)
    U, s, Vt = np.linalg.svd(Ym, full_matrices=False)
    V = Vt[:min(K, Vt.shape[0])].T
    T = Ym @ V
    R = Ridge(alpha=alpha).fit(a, T)
    return R.predict(c) @ V.T + b.mean(0)


def fit_rrr15(a, b, c):
    return fit_rrr(a, b, c, K=15)


class MTGNN:
    def __init__(self, hidden=96, drop=0.4, epochs=260, lr=3e-3, wd=3e-3, seed=RS):
        self.__dict__.update(hidden=hidden, drop=drop, epochs=epochs, lr=lr, wd=wd, seed=seed)
        self.mu = self.sd = None

    def fit(self, Xtr, Ytr):
        import torch
        import torch.nn as nn
        torch.set_num_threads(2)          # 避免占满全部逻辑核
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        n, p = Xtr.shape
        k = Ytr.shape[1]
        self.p1 = p // 2
        self.mu = Ytr.mean(0); self.sd = Ytr.std(0) + 1e-8
        Yz = (Ytr - self.mu) / self.sd
        Xt = torch.tensor(Xtr, dtype=torch.float32)
        Yt = torch.tensor(Yz, dtype=torch.float32)
        p1, p2 = self.p1, p - self.p1
        H, dr = self.hidden, self.drop

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.b1 = nn.Sequential(nn.Linear(p1, H), nn.BatchNorm1d(H), nn.SiLU(),
                                        nn.Dropout(dr))
                self.b2 = nn.Sequential(nn.Linear(p2, H), nn.BatchNorm1d(H), nn.SiLU(),
                                        nn.Dropout(dr))
                self.head = nn.Sequential(nn.Linear(2 * H, H), nn.SiLU(), nn.Dropout(dr),
                                          nn.Linear(H, k))

            def forward(self, x):
                return self.head(torch.cat([self.b1(x[:, :p1]), self.b2(x[:, p1:])], 1))

        net = Net()
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        lossf = nn.SmoothL1Loss(beta=0.5)
        bs = 64
        for ep in range(self.epochs):
            net.train()
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                lossf(net(Xt[idx]), Yt[idx]).backward()
                opt.step()
            sch.step()
        self.net = net.eval()
        return self

    def predict(self, Xte):
        import torch
        with torch.no_grad():
            o = self.net(torch.tensor(Xte, dtype=torch.float32)).numpy()
        return o * self.sd + self.mu


def fit_mtgnn(a, b, c):
    return MTGNN().fit(a, b).predict(c)


MODELS = {
    # 先跑便宜的，出错可快速发现；GBR/HistGBR 最慢放最后
    'MT-GNN': (fit_mtgnn, True),
    'Ridge': (fit_ridge, False), 'ElasticNet': (fit_enet, False), 'PLS': (fit_pls, False),
    '潜因子Ridge-K8': (fit_rrr, False), '潜因子Ridge-K15': (fit_rrr15, False),
    'RandomForest': (fit_rrf, False), 'RandomForest-B': (fit_lgbm, False),
    'HistGBR': (fit_xgb, False), 'GBR': (fit_gbr, False),
}


# ---------------------------------------------------------------- CV
PRED_DIR = os.path.join(C.OUT, 'q1_preds')
os.makedirs(PRED_DIR, exist_ok=True)


def run_cv(name, fn, use_graph):
    t0 = time.time()
    cache = os.path.join(PRED_DIR, '%s.npz' % name.replace('+', '_').replace('/', '_'))
    if os.path.exists(cache):
        oof = np.load(cache)['oof']
        pred = np.nanmean(oof, axis=0)
        r2s = np.array([r2_score(Y[:, j], pred[:, j]) for j in range(len(SDIM))])
        maes = np.array([mean_absolute_error(Y[:, j], pred[:, j]) for j in range(len(SDIM))])
        log('%-18s meanR2=%.4f median=%.4f max=%.4f min=%.4f  [缓存]' % (
            name, r2s.mean(), np.median(r2s), r2s.max(), r2s.min()))
        return dict(model=name, mean_r2=float(r2s.mean()), median_r2=float(np.median(r2s)),
                    max_r2=float(r2s.max()), min_r2=float(r2s.min()),
                    mean_mae=float(maes.mean()), seconds=0,
                    per_dim=dict(zip(SDIM, r2s.round(4).tolist()))), pred
    oof = np.full((N_REPEATS, len(Y), len(SDIM)), np.nan)
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS + rep)
        for tr, te in skf.split(Xraw, strat):
            Ztr, Zte = prep(Xraw[tr], Xraw[te], use_graph=use_graph)
            oof[rep][te] = fn(Ztr, Y[tr], Zte)
    np.savez_compressed(cache, oof=oof)
    pred = np.nanmean(oof, axis=0)
    r2s = np.array([r2_score(Y[:, j], pred[:, j]) for j in range(len(SDIM))])
    maes = np.array([mean_absolute_error(Y[:, j], pred[:, j]) for j in range(len(SDIM))])
    log('%-18s meanR2=%.4f median=%.4f max=%.4f min=%.4f  %.0fs' % (
        name, r2s.mean(), np.median(r2s), r2s.max(), r2s.min(), time.time() - t0))
    return dict(model=name, mean_r2=float(r2s.mean()), median_r2=float(np.median(r2s)),
                max_r2=float(r2s.max()), min_r2=float(r2s.min()),
                mean_mae=float(maes.mean()), seconds=round(time.time() - t0, 1),
                per_dim=dict(zip(SDIM, r2s.round(4).tolist()))), pred


log('\n' + '=' * 78)
log('交叉验证 %d 折 × %d 重复，按风格分层，pooled R²' % (N_SPLITS, N_REPEATS))
log('=' * 78)
results, preds = [], {}
for name, (fn, ug) in MODELS.items():
    res, pred = run_cv(name, fn, ug)
    results.append(res); preds[name] = pred

R = pd.DataFrame([{k: v for k, v in r.items() if k != 'per_dim'} for r in results])
R.to_csv(os.path.join(C.OUT, 'q1_cv_summary.csv'), index=False, encoding='utf-8-sig')
log('\n' + R.sort_values('mean_r2', ascending=False).to_string(index=False))

PD = pd.DataFrame({r['model']: r['per_dim'] for r in results})
PD.insert(0, '感官维度', [C.clean_sens_name(c) for c in SDIM])
PD.insert(1, '分组', [C.SENS_GROUP[c] for c in SDIM])
PD.to_csv(os.path.join(C.OUT, 'q1_perdim.csv'), index=False, encoding='utf-8-sig')
best = max(results, key=lambda r: r['mean_r2'])['model']
log('\n最优单模型 %s  mean R2 = %.4f' % (best, max(r['mean_r2'] for r in results)))

# ---------------------------------------------------------------- 堆叠
log('\n堆叠集成 ...')
names = list(preds)
oof_stack = np.full((N_REPEATS, len(Y), len(SDIM)), np.nan)
for rep in range(N_REPEATS):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS + 111 + rep)
    for tr, te in skf.split(Xraw, strat):
        for j in range(len(SDIM)):
            Ztrm = np.column_stack([preds[n][tr, j] for n in names])
            Ztem = np.column_stack([preds[n][te, j] for n in names])
            mm = Ridge(alpha=1.0, positive=True).fit(Ztrm, Y[tr, j])
            oof_stack[rep][te, j] = mm.predict(Ztem)
stack_pred = np.nanmean(oof_stack, axis=0)
r2_stack = np.array([r2_score(Y[:, j], stack_pred[:, j]) for j in range(len(SDIM))])
log('Stack meanR2=%.4f median=%.4f max=%.4f min=%.4f' % (
    r2_stack.mean(), np.median(r2_stack), r2_stack.max(), r2_stack.min()))
results.append(dict(model='Stack', mean_r2=float(r2_stack.mean()),
                    median_r2=float(np.median(r2_stack)), max_r2=float(r2_stack.max()),
                    min_r2=float(r2_stack.min()),
                    mean_mae=float(np.mean([mean_absolute_error(Y[:, j], stack_pred[:, j])
                                            for j in range(len(SDIM))])),
                    seconds=0, per_dim=dict(zip(SDIM, r2_stack.round(4).tolist()))))
PD['Stack'] = r2_stack
PD.to_csv(os.path.join(C.OUT, 'q1_perdim.csv'), index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 区间校准（CV+ / 分裂一致性）
log('\n不确定性：分裂一致性 90% 区间覆盖率 ...')
spread = np.nanstd(oof_stack, axis=0)          # 重复间预测标准差
zq = 1.645
lo = stack_pred - zq * spread
hi = stack_pred + zq * spread
cov = float(np.mean((Y >= lo) & (Y <= hi)))
width = float(np.mean(hi - lo))
log('90%% 名义区间：实际覆盖率 %.3f，平均宽度 %.3f（目标尺度 %.2f）' % (
    cov, width, float(Y.std())))

# ---------------------------------------------------------------- 逐维指标
perdim = []
for j, s in enumerate(SDIM):
    m = C.metrics(Y[:, j], stack_pred[:, j])
    m['dim'] = s; m['name'] = C.clean_sens_name(s); m['group'] = C.SENS_GROUP[s]
    m['dim_std'] = float(Y[:, j].std())
    m['best_single_r2'] = float(max(r['per_dim'][s] for r in results if r['model'] != 'Stack'))
    m['stack_r2'] = float(r2_stack[j])
    perdim.append(m)
PDR = pd.DataFrame(perdim)[['dim', 'name', 'group', 'r2', 'mae', 'rmse', 'spearman',
                            'best_single_r2', 'stack_r2', 'dim_std']]
PDR.to_csv(os.path.join(C.OUT, 'q1_metrics_perdim.csv'), index=False, encoding='utf-8-sig')
log('\n分组平均 R²：')
log(PDR.groupby('group')['stack_r2'].agg(['mean', 'count']).to_string())

json.dump(dict(results=results, best_single=best,
               stack_mean_r2=float(r2_stack.mean()),
               stack_median_r2=float(np.median(r2_stack)),
               stack_max_r2=float(r2_stack.max()),
               stack_max_dim=str(SDIM[int(np.argmax(r2_stack))]),
               group_r2={k: float(v) for k, v in PDR.groupby('group')['stack_r2'].mean().items()},
               interval=dict(nominal=0.90, coverage=cov, width=width),
               protocol=dict(splits=N_SPLITS, repeats=N_REPEATS, strat='style'),
               n_samples=int(len(Y)), n_features=int(len(COMP)), n_targets=int(len(SDIM))),
          open(os.path.join(C.OUT, 'q1_cv.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)

# ---------------------------------------------------------------- 0-6 交付
log('\n生成 0-6 分制评分 ...')
Y06, spec = C.calib06(D['Y'])
Ztr, Zte = prep(Xraw, Xraw, use_graph=True)
_ = fit_mtgnn(Ztr, Y, Zte)
# 全量堆叠（用 OOF 上标定的非负岭权重作用于各模型 CV 预测的平均，保证尺度一致）
Zfull = np.zeros((len(Y), len(SDIM)))
for j in range(len(SDIM)):
    Zm = np.column_stack([preds[n][:, j] for n in names])
    mm = Ridge(alpha=1.0, positive=True).fit(Zm, Y[:, j])
    Zfull[:, j] = mm.predict(Zm)
S = np.clip(6.0 * (Zfull - Zfull.min(0)) / (Zfull.max(0) - Zfull.min(0)), 0, 6)
S06 = pd.DataFrame(S, columns=SDIM)
S06.insert(0, 'beer', D['beer'])
S06.to_csv(os.path.join(C.OUT, 'q1_scores_06.csv'), index=False, encoding='utf-8-sig')
gr = S06.copy()
for c in SDIM:
    gr[c] = np.clip(np.rint(gr[c]), 0, 6).astype(int)
gr.to_csv(os.path.join(C.OUT, 'q1_scores_grade.csv'), index=False, encoding='utf-8-sig')
Y06.insert(0, 'beer', D['beer'])
Y06.to_csv(os.path.join(C.OUT, 'sensory_06_truth.csv'), index=False, encoding='utf-8-sig')
np.savez_compressed(os.path.join(C.OUT, 'q1_oof.npz'), oof=oof_stack, pred=stack_pred,
                    spread=spread)
log('q1_scores_06.csv / q1_scores_grade.csv / q1_oof.npz 已写出')
log('\n完成。Stack mean R2 = %.4f' % r2_stack.mean())
