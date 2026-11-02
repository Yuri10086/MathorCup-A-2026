# -*- coding: utf-8 -*-
"""02_q1.py —— 问题1 主流程：多模型横向比较 + 堆叠 + 0–6 交付

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek）。

模型栈（同一 CV 协议下横向比较）：
  L0 线性      RidgeCV（标准化 + 自动选 α，诚实基线）/ ElasticNet / PLS
  L1 树        RandomForest / HistGradientBoosting / GradientBoosting（文献口径）
  L2 潜因子    对 Y 做 SVD 取 K 个感官因子再岭回归（RRR）
  L3 主模型    MT-GNN（双分支共享隐层、Huber 损失、一次输出 50 维）
  L4 堆叠      以各模型 OOF 为元特征，逐维非负岭回归
CV：按风格分层 5 折 × 3 重复；重复间预测离散度 → 预测区间
交付：真值分位切点把预测 z 分映射 0–6 整数档（与真值同分布），并给出连续 0–6

设计要点：**每个折的预处理只算一次**，所有模型共享，避免 120 次重复预处理。
单模型结果落盘到 output/q1_preds/<name>.npz，重跑自动复用。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q1lib as C

T0 = time.time()
RS = C.RS
D = C.load()
COMP, SDIM = D['comp'], D['sdim']
Y = D['Y'].values.astype(float)
STYLE = D['style']
SCALE, _ = C.colscale()
XH = C.harmonize(D['X'], COMP, SCALE)
STRAT = C.strat_labels(STYLE)
CATS = sorted(set(STYLE))
PRED_DIR = os.path.join(C.OUT, 'q1_preds')
os.makedirs(PRED_DIR, exist_ok=True)
LOG = open(os.path.join(C.OUT, 'q1_progress.log'), 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    print(s, file=LOG, flush=True)


log('样本 %d，化合物 %d，感官维度 %d；风格 %d 类 → 分层标签 %d 类' % (
    len(Y), len(COMP), len(SDIM), len(set(STYLE)), len(set(STRAT))))

# ---------------------------------------------------------------- 模型
def fit_ridgecv(a, b, c):
    from sklearn.linear_model import RidgeCV
    from sklearn.multioutput import MultiOutputRegressor
    m = MultiOutputRegressor(RidgeCV(alphas=np.logspace(-1, 4, 40)), n_jobs=1).fit(a, b)
    return np.clip(m.predict(c), -6, 8)


def fit_enet(a, b, c):
    from sklearn.linear_model import ElasticNet
    from sklearn.multioutput import MultiOutputRegressor
    return MultiOutputRegressor(ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=4000),
                                n_jobs=1).fit(a, b).predict(c)


def fit_pls(a, b, c):
    from sklearn.cross_decomposition import PLSRegression
    return PLSRegression(n_components=12, scale=False).fit(a, b).predict(c)


def fit_rrr(a, b, c):
    from sklearn.linear_model import Ridge
    Ym = b - b.mean(0)
    _, _, Vt = np.linalg.svd(Ym, full_matrices=False)
    V = Vt[:8].T
    R = Ridge(alpha=30.0).fit(a, Ym @ V)
    return R.predict(c) @ V.T + b.mean(0)


def fit_rf(a, b, c):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=500, max_features=0.33, min_samples_leaf=2,
                                 n_jobs=1, random_state=RS).fit(a, b).predict(c)


def fit_hgb(a, b, c):
    """直方图梯度提升（sklearn 无原生多输出，用 MultiOutput 封装）。

    50 个输出 × 15 折量级较大，故 max_iter=150、lr=0.08，属于同族但更省时的设定。
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    return MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=3,
                                      min_samples_leaf=10, l2_regularization=2.0,
                                      max_features=0.5, early_stopping=False,
                                      random_state=RS), n_jobs=1).fit(a, b).predict(c)


class MTGNN:
    """双分支共享隐层多任务网络：一次输出 50 维感官，Huber 损失。"""

    def __init__(self, hidden=96, drop=0.35, epochs=240, lr=3e-3, wd=3e-3, seed=RS):
        self.__dict__.update(hidden=hidden, drop=drop, epochs=epochs, lr=lr,
                             wd=wd, seed=seed)

    def fit(self, Xtr, Ytr):
        import torch
        import torch.nn as nn
        torch.set_num_threads(4)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        n, p = Xtr.shape
        k = Ytr.shape[1]
        p1 = p // 2
        self.mu, self.sd = Ytr.mean(0), Ytr.std(0) + 1e-8
        Xt = torch.tensor(Xtr, dtype=torch.float32)
        Yt = torch.tensor((Ytr - self.mu) / self.sd, dtype=torch.float32)
        H, dr = self.hidden, self.drop
        p2 = p - p1

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
        for _ in range(self.epochs):
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
    'RidgeCV': fit_ridgecv,
    'ElasticNet': fit_enet,
    'PLS': fit_pls,
    'RRR-K8': fit_rrr,
    'RandomForest': fit_rf,
    'HistGBR': fit_hgb,
    'MT-GNN': fit_mtgnn,
}

# ---------------------------------------------------------------- CV
log('\n' + '=' * 78)
log('交叉验证 %d 折 × %d 重复，按风格分层，pooled R²' % (C.N_SPLITS, C.N_REPEATS))
log('=' * 78)


def run_model(name, fn):
    fp = os.path.join(PRED_DIR, '%s.npz' % name)
    if os.path.exists(fp):
        oof = np.load(fp)['oof']
        log('%-14s [复用缓存]' % name)
    else:
        from sklearn.model_selection import StratifiedKFold
        oof = np.full((C.N_REPEATS, len(Y), len(SDIM)), np.nan)
        t0 = time.time()
        for rep in range(C.N_REPEATS):
            skf = StratifiedKFold(n_splits=C.N_SPLITS, shuffle=True, random_state=RS + rep)
            for fold, (tr, te) in enumerate(skf.split(XH, STRAT)):
                opt = dict(style_cats=CATS, style_tr=STYLE[tr], style_te=STYLE[te])
                A, B = C.prep(XH.iloc[tr], XH.iloc[te], SCALE, COMP, opt)
                oof[rep][te] = fn(A, Y[tr], B)
                log('  %s rep%d fold%d %5.1fs' % (name, rep, fold, time.time() - t0))
        np.savez_compressed(fp, oof=oof)
        log('%-14s 用 %5.0fs' % (name, time.time() - t0))
    pred = np.nanmean(oof, axis=0)
    r = C.per_dim_r2(Y, pred)
    log('%-14s meanR2=%.4f median=%.4f max=%.4f min=%.4f' % (
        name, np.nanmean(r), np.median(r), np.nanmax(r), np.nanmin(r)))
    return oof, r


OOF, RES = {}, {}
for name, fn in MODELS.items():
    OOF[name], RES[name] = run_model(name, fn)
names = list(MODELS)

# ---------------------------------------------------------------- 堆叠
log('\n堆叠集成（逐维非负岭回归，避免负系数外推）...')
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

stack_oof = np.full((C.N_REPEATS, len(Y), len(SDIM)), np.nan)
stack_w = {}
for rep in range(C.N_REPEATS):
    skf = StratifiedKFold(n_splits=C.N_SPLITS, shuffle=True, random_state=RS + 111 + rep)
    for tr, te in skf.split(XH, STRAT):
        for j in range(len(SDIM)):
            Ztr = np.column_stack([OOF[n][rep][tr, j] for n in names])
            Zte = np.column_stack([OOF[n][rep][te, j] for n in names])
            mm = Ridge(alpha=1.0, positive=True).fit(Ztr, Y[tr, j])
            stack_oof[rep][te, j] = mm.predict(Zte)
            if rep == 0:
                stack_w[SDIM[j]] = dict(zip(names, mm.coef_.round(4).tolist()))
stack_pred = np.nanmean(stack_oof, axis=0)
r_stack = C.per_dim_r2(Y, stack_pred)
RES['Stack'] = r_stack
log('Stack          meanR2=%.4f median=%.4f max=%.4f min=%.4f' % (
    np.nanmean(r_stack), np.median(r_stack), np.nanmax(r_stack), np.nanmin(r_stack)))

CV = pd.DataFrame([dict(model=m, mean_r2=float(np.nanmean(v)), median_r2=float(np.median(v)),
                        min_r2=float(np.nanmin(v)), max_r2=float(np.nanmax(v)))
                   for m, v in RES.items()]).sort_values('mean_r2', ascending=False)
CV.to_csv(os.path.join(C.OUT, 'q1_cv_summary.csv'), index=False, encoding='utf-8-sig')
log('\n' + CV.to_string(index=False))

perdim = []
from scipy.stats import spearmanr
for j, s in enumerate(SDIM):
    single = max(RES[n][j] for n in names)
    ps = [np.corrcoef(Y[:, j], OOF[n][:, :, j].mean(0))[0, 1] for n in names]
    perdim.append(dict(dim=s, stack_r2=float(r_stack[j]),
                       best_single_r2=float(single),
                       stack_minus_best=float(r_stack[j] - single),
                       best_single_name=names[int(np.argmax([RES[n][j] for n in names]))],
                       mean_mae=float(np.mean(np.abs(Y[:, j] - stack_pred[:, j]))),
                       spearman=float(spearmanr(Y[:, j], stack_pred[:, j]).statistic),
                       dim_std=float(Y[:, j].std())))
PD = pd.DataFrame(perdim)
GROUP = {}
for c in SDIM:
    GROUP[c] = ('Aroma' if c.startswith('A_') else
                'Flavour' if c.startswith('F_') else
                'Fault' if c in ('dms', 'diacetyl', 't2n', 'stale_hops', 'metallic', 'X4vg') else
                'Mouthfeel' if c in ('body', 'bitternes', 'co2', 'aftertaste', 'alcohol',
                                     'sweetness', 'acidity') else
                'Spice/野生' if c in ('orange', 'coriander', 'clove', 'lactic', 'acetic',
                                      'barnyard') else 'Overall')
PD['group'] = [GROUP[c] for c in SDIM]
PD.to_csv(os.path.join(C.OUT, 'q1_perdim.csv'), index=False, encoding='utf-8-sig')

log('\n逐维 R²：>=0.4 的 %d 维，0~0.4 的 %d 维，<0 的 %d 维' % (
    int((PD.stack_r2 >= 0.4).sum()), int(((PD.stack_r2 >= 0) & (PD.stack_r2 < 0.4)).sum()),
    int((PD.stack_r2 < 0).sum())))
log('堆叠优于最优单模型的维数：%d/50' % int((PD.stack_minus_best > 0).sum()))
log('R² 与该维标签标准差的 Spearman：%.3f' % spearmanr(PD.stack_r2, PD.dim_std).statistic)
grp = PD.groupby('group')['stack_r2'].agg(['mean', 'count']).sort_values('mean', ascending=False)
log('\n分组平均 R²：')
log(grp.to_string())

# ---------------------------------------------------------------- 不确定性
spread = np.nanstd(stack_oof, axis=0)
lo, hi = stack_pred - 1.645 * spread, stack_pred + 1.645 * spread
cov = float(np.mean((Y >= lo) & (Y <= hi)))
width = float(np.mean(hi - lo))
log('\n90%% 名义预测区间（重复间分裂一致性）：覆盖率 %.4f，平均宽度 %.4f，标签 sd=%.3f'
    % (cov, width, float(Y.std())))
log('→ 覆盖率远低于 90%%：3 次重复的预测离散度**低估**了真实不确定性，如实报告。')

# ---------------------------------------------------------------- 0–6 交付
log('\n生成 0–6 分制交付表 ...')
# 三个候选口径，全部落盘、全部量化，主交付选"分位映射校准"：
#   G1 真值分位切点直接切预测      —— 最直观，但预测被回归稀释 → 档位全挤到中间
#   G2 分位映射校准（quantile mapping）—— 把预测的秩映射到真值的 1/7 分位点，
#                                        是预报校准的标准做法；输出分布与真值一致
#   G3 方差还原后再切            —— 先把预测标准差放大到真值尺度，再切档
QT = np.linspace(0, 100, 8)[1:-1]
cuts = np.array([np.percentile(Y[:, j], QT) for j in range(len(SDIM))])   # (50, 6)
edges_all = np.array([np.concatenate([[Y[:, j].min() - 1e-9], cuts[j],
                                      [Y[:, j].max() + 1e-9]])
                      for j in range(len(SDIM))])                        # (50, 8)


def grade_by_cuts(V):
    G = np.zeros(V.shape, dtype=int)
    for j in range(V.shape[1]):
        G[:, j] = (V[:, j][:, None] > cuts[j][None, :]).sum(1)
    return np.clip(G, 0, 6)


def grade_quantile(V):
    """分位映射校准：把预测值在该维内的**秩**均匀切成 7 档。

    这与"把秩穿过真值经验分位函数再切 7 档"等价（真值切点本身就是 1/7 分位点），
    是预报校准里的标准做法，单调、无参数，且保证每维 7 档的边缘分布与真值一致。
    """
    G = np.zeros(V.shape, dtype=int)
    n = V.shape[0]
    for j in range(V.shape[1]):
        order = np.argsort(np.argsort(V[:, j]))      # 0..n-1 的名次
        G[:, j] = np.minimum((order * 7) // n, 6)
    return G


k_dim = Y.std(0) / np.maximum(stack_pred.std(0), 1e-9)
P_rest = Y.mean(0) + (stack_pred - stack_pred.mean(0)) * k_dim[None, :]

G1 = grade_by_cuts(stack_pred)
G2 = grade_quantile(stack_pred)
G3 = grade_by_cuts(P_rest)
Y06 = grade_by_cuts(Y)

GRADE = G2                      # 主交付
CONT = np.zeros_like(stack_pred)
for j in range(len(SDIM)):
    CONT[:, j] = np.clip(np.interp(stack_pred[:, j], edges_all[j], np.arange(8)), 0, 6)

for arr, fn, nd in [(GRADE, 'q1_scores_grade.csv', 0), (CONT, 'q1_scores_06.csv', 4),
                    (Y06, 'sensory_06_truth.csv', 0), (G1, 'q1_scores_grade_direct.csv', 0),
                    (G3, 'q1_scores_grade_varrestore.csv', 0)]:
    T = pd.DataFrame(np.round(arr, nd).astype(int) if nd == 0 else np.round(arr, nd),
                     columns=SDIM)
    T.insert(0, 'beer', D['beer'])
    T.to_csv(os.path.join(C.OUT, fn), index=False, encoding='utf-8-sig')


def tvd(a, b):
    pa = np.bincount(a.ravel(), minlength=7) / a.size
    pb = np.bincount(b.ravel(), minlength=7) / b.size
    return float(0.5 * np.abs(pa - pb).sum())


log('方差还原系数 k：中位 %.3f（范围 %.2f ~ %.2f）→ 预测被回归稀释' % (
    np.median(k_dim), k_dim.min(), k_dim.max()))
log('%-24s %8s %8s %8s' % ('口径', '完全命中', '±1档', 'TVD'))
GMAP = {}
for nm, G in [('G1 直接切真值分位', G1), ('G2 分位映射校准(主交付)', G2),
              ('G3 方差还原后切档', G3)]:
    a1 = float(np.mean(G == Y06))
    a2 = float(np.mean(np.abs(G - Y06) <= 1))
    d = tvd(G, Y06)
    GMAP[nm] = dict(exact=a1, within1=a2, tvd=d,
                    dist=np.bincount(G.ravel(), minlength=7).tolist())
    log('%-24s %8.4f %8.4f %8.4f' % (nm, a1, a2, d))
log('档位分布 真值 %s' % np.bincount(Y06.ravel(), minlength=7).tolist())
for nm in GMAP:
    log('         %-22s %s' % (nm.split(' ')[0], GMAP[nm]['dist']))

acc = GMAP['G2 分位映射校准(主交付)']['exact']
acc1 = GMAP['G2 分位映射校准(主交付)']['within1']
log('主交付：整数档逐格命中率 %.4f，±1 档内 %.4f（7 档基线 0.143 / 0.429）' % (acc, acc1))
per_acc = np.array([np.mean(GRADE[:, j] == Y06[:, j]) for j in range(len(SDIM))])
log('逐维完全命中率：中位 %.3f，最好 %.3f（%s），最差 %.3f（%s）' % (
    np.median(per_acc), per_acc.max(), SDIM[int(per_acc.argmax())],
    per_acc.min(), SDIM[int(per_acc.argmin())]))

json.dump(dict(results=CV.to_dict(orient='records'),
               stack_mean_r2=float(np.nanmean(r_stack)),
               best_single=max(names, key=lambda n: np.nanmean(RES[n])),
               best_single_r2=float(max(np.nanmean(RES[n]) for n in names)),
               group_r2={k: float(v) for k, v in grp['mean'].items()},
               interval=dict(nominal=0.90, coverage=cov, width=width),
               grade=dict(chosen='G2 quantile mapping', exact_acc=acc, within1=acc1,
                          alternatives=GMAP,
                          k_restore_median=float(np.median(k_dim)),
                          dist_pred=GMAP['G2 分位映射校准(主交付)']['dist'],
                          dist_truth=np.bincount(Y06.ravel(), minlength=7).tolist(),
                          per_dim_acc=dict(zip(SDIM, per_acc.round(4).tolist()))),
               per_dim=dict(zip(SDIM, np.round(r_stack, 4).tolist())),
               stack_weights=stack_w,
               protocol=dict(splits=C.N_SPLITS, repeats=C.N_REPEATS, strat='style')),
          open(os.path.join(C.OUT, 'q1_cv.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
np.savez_compressed(os.path.join(C.OUT, 'q1_result.npz'), stack_oof=stack_oof,
                    stack_pred=stack_pred, pred_restored=P_rest, spread=spread,
                    grade=GRADE, grade_direct=G1, grade_varrestore=G3, grade_truth=Y06,
                    cont=CONT, cuts=cuts, k_dim=k_dim)
log('\n完成，总耗时 %.0fs。Stack mean R² = %.4f' % (time.time() - T0, np.nanmean(r_stack)))


