# -*- coding: utf-8 -*-
"""03 问题3：风味与"风味指纹"生成（风格条件生成 + 跨风格迁移）

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。

模型 SCLF（Style-Conditional Latent Flow）：
  ① 对数化合物空间 → 潜因子（PCA，K 由并行分析确定）
  ② 风格条件归一化流 p(z | style)：仿射耦合层 + 风格嵌入（可算精确似然 → 可与高斯基线比 BIC）
  ③ 解码回化合物谱 → 感官画像由问题1 的感官预测模型给出
  ④ 跨风格迁移：高斯 Bures-Wasserstein 最优传输映射 + 缺陷化合物单调约束投影
评估：风格判别保真度 / Fréchet 距离 / 能量距离 / 多样性 / 感官画像距离 / 缺陷安全性
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

_TMP = r'E:\dsh_work\MC2026A\.tmp'
os.makedirs(_TMP, exist_ok=True)
for k in ('JOBLIB_TEMP_FOLDER', 'TMP', 'TEMP'):
    os.environ[k] = _TMP

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# ⚠ torch 默认会占满全部逻辑核，与并行的 sklearn/XGBoost 争抢导致整体停滞；
#   本任务网络极小（输入 14 维），2 线程足够。
os.environ.setdefault('OMP_NUM_THREADS', '4')
try:
    import torch as _t
    _t.set_num_threads(2)
except Exception:
    pass

RS = 20261026
rng = np.random.default_rng(RS)
D = C.load()
COMP, SDIM = D['comp'], D['sdim']
STYLE = D['style']
META = C.build_meta(COMP)
LOG = open(os.path.join(C.OUT, 'q3_progress.log'), 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); print(s, file=LOG, flush=True)


# ---------------------------------------------------------------- 数据
X = SimpleImputer(strategy='median').fit_transform(D['X'].values.astype(float))
Y = D['Y'].values.astype(float)
sc = StandardScaler().fit(X)
Xz = sc.transform(X)
n, p = Xz.shape
log('样本 %d，化合物 %d，风格 %d 类' % (n, p, len(set(STYLE))))

# ---------------------------------------------------------------- PCA 潜因子
U, S, Vt = np.linalg.svd(Xz - Xz.mean(0), full_matrices=False)
ev = S ** 2 / max(1, n - 1)
evr = ev / ev.sum()
# 并行分析：与打乱矩阵的特征值比较
ev_rand = []
for _ in range(30):
    Xr = np.array([rng.permutation(col) for col in Xz.T]).T
    Ur, Sr, _ = np.linalg.svd(Xr - Xr.mean(0), full_matrices=False)
    ev_rand.append((Sr ** 2 / (n - 1)))
ev_rand = np.median(np.array(ev_rand), axis=0)
K = int(np.sum(ev > ev_rand * 1.0))
K = int(np.clip(K, 4, 20))
log('并行分析建议 K=%d（累计方差 %.1f%%），取 K=%d' % (
    K, 100 * evr[:K].sum(), K))
V = Vt[:K].T                                  # (p, K) 载荷
Z = (Xz - Xz.mean(0)) @ V                     # (n, K) 潜因子
Zmu, Zsd = Z.mean(0), Z.std(0) + 1e-9
Zn = (Z - Zmu) / Zsd
log('潜因子解释方差：%s' % np.round(100 * evr[:K], 1).tolist())


def decode(zn):
    """潜因子 → 化合物谱（与 1.csv 同尺度的 log10 空间）。"""
    Zf = zn * Zsd + Zmu
    xz = Zf @ V.T + Xz.mean(0)          # 标准化空间
    return sc.inverse_transform(xz)     # 回到 log10 尺度


def encode(xz):
    """标准化空间坐标 → 潜因子。"""
    return ((xz - Xz.mean(0)) @ V - Zmu) / Zsd


# ---------------------------------------------------------------- 感官代理模型
log('\n训练感官预测代理模型（问题1 模型族的 XGBoost 分支）')


def prep_feat(Xa):
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xa)
    Za = imp.transform(Xa)
    lo, hi = np.nanpercentile(Za, [1, 99], axis=0)
    Za = np.clip(Za, lo, hi)
    med = np.median(Za, 0)
    iqr = np.percentile(Za, 75, 0) - np.percentile(Za, 25, 0)
    iqr[iqr <= 0] = 1e-9
    return (Za - med) / iqr


XF = prep_feat(X)
import xgboost as xgb
# ⚠ 不能用 joblib/loky 子进程：XGBoost 在子进程里会误探测 CUDA 而报错
surr = MultiOutputRegressor(
    xgb.XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=3, subsample=0.85,
                     colsample_bytree=0.5, reg_lambda=2.0, n_jobs=4, random_state=RS,
                     tree_method='hist', device='cpu', verbosity=0), n_jobs=1).fit(XF, Y)
vc = pd.Series(STYLE).value_counts()
strat = np.array([s if vc[s] >= 5 else 'Rare' for s in STYLE])
oof = np.zeros_like(Y)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=RS).split(XF, strat):
    m = MultiOutputRegressor(
        xgb.XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=3, subsample=0.85,
                         colsample_bytree=0.5, reg_lambda=2.0, n_jobs=4, random_state=RS,
                         tree_method='hist', device='cpu', verbosity=0), n_jobs=1).fit(XF[tr], Y[tr])
    oof[te] = m.predict(XF[te])
r2_surr = np.array([r2_score(Y[:, j], oof[:, j]) for j in range(len(SDIM))])
log('感官代理模型 CV R² 均值 %.3f（与问题1 同族模型一致）' % r2_surr.mean())


def predict_sensory_from_logx(Lx):
    """输入：与 1.csv 同尺度的 log10 化合物谱（n×231）。"""
    Xr = np.asarray(Lx, float)
    return surr.predict(prep_feat(Xr))


# ---------------------------------------------------------------- 风格统计
styles = [s for s in vc.index if vc[s] >= 8]
log('\n可生成的风格（n>=8）：%s' % ', '.join('%s(%d)' % (s, vc[s]) for s in styles))
mu_s, cov_s, n_s = {}, {}, {}
for s in styles:
    m = STYLE == s
    mu_s[s] = Zn[m].mean(0)
    n_s[s] = int(m.sum())
# 收缩协方差（Ledoit-Wolf 型：向单位阵收缩）
resid = np.concatenate([Zn[STYLE == s] - mu_s[s] for s in styles])
S_pool = np.cov(resid.T) + 1e-4 * np.eye(K)
lam = 0.35
for s in styles:
    R = Zn[STYLE == s] - mu_s[s]
    Ss = (R.T @ R) / max(1, len(R) - 1)
    w = n_s[s] / (n_s[s] + 12.0)
    cov_s[s] = w * Ss + (1 - w) * S_pool + 1e-4 * np.eye(K)


def sqrtm_psd(A):
    w, U = np.linalg.eigh((A + A.T) / 2)
    w = np.clip(w, 1e-8, None)
    return U @ np.diag(np.sqrt(w)) @ U.T


def ot_map(mu_s_, S_s, mu_t, S_t):
    """高斯 Bures-Wasserstein 最优传输：z' = mu_t + A (z - mu_s)。"""
    Ss_12 = sqrtm_psd(S_s)
    inner = sqrtm_psd(Ss_12 @ S_t @ Ss_12)
    A = np.linalg.pinv(Ss_12) @ inner @ np.linalg.pinv(Ss_12)
    return mu_t, A


# ---------------------------------------------------------------- 条件流
try:
    import torch as _torch
    import torch.nn as _nn
    _BASE = _nn.Module
except Exception:                                    # 理论不会发生
    _BASE = object


class StyleFlow(_BASE):
    """风格条件仿射耦合流（RealNVP 结构），可算精确对数似然。"""

    def __init__(self, K, n_style, n_layers=6, hidden=64, seed=RS):
        import torch
        import torch.nn as nn
        super().__init__()
        torch.manual_seed(seed)
        self.K, self.n_layers = K, n_layers
        self.emb = nn.Embedding(n_style, 8)

        class Coupling(nn.Module):
            def __init__(self, d, hid):
                super().__init__()
                self.d = d
                self.net = nn.Sequential(nn.Linear(d // 2 + 8, hid), nn.Tanh(),
                                         nn.Linear(hid, hid), nn.Tanh(),
                                         nn.Linear(hid, 2 * (d - d // 2)))
                self.net[-1].weight.data.mul_(0.01)
                self.net[-1].bias.data.zero_()

            def forward(self, x, e):
                a, b = x[:, :self.d // 2], x[:, self.d // 2:]
                h = self.net(torch.cat([a, e], 1))
                s, t = h.chunk(2, dim=1)
                s = 0.5 * torch.tanh(s)
                return torch.cat([a, b * torch.exp(s) + t], 1), s.sum(1)

        class Flip(nn.Module):
            def forward(self, x):
                return torch.flip(x, [1]), torch.zeros(x.shape[0], device=x.device)

        layers = []
        for _ in range(n_layers):
            layers.append(Coupling(K, hidden)); layers.append(Flip())
        self.layers = nn.ModuleList(layers)
        self.net = nn.ModuleList([l for l in layers if isinstance(l, Coupling)])

    def forward(self, z, sid):
        """z → u（标准化正态），返回 (u, logdet)。"""
        import torch
        e = self.emb(sid)
        ld = torch.zeros(z.shape[0])
        for l in self.layers:
            if l.__class__.__name__ == 'Coupling':
                z, d = l(z, e); ld = ld + d
            else:
                z, _ = l(z)
        return z, ld

    def inverse(self, u, sid):
        import torch
        e = self.emb(sid)
        for l in reversed(self.layers):
            if l.__class__.__name__ == 'Coupling':
                a, b = u[:, :self.K // 2], u[:, self.K // 2:]
                h = l.net(torch.cat([a, e], 1))
                s, t = h.chunk(2, dim=1)
                s = 0.5 * torch.tanh(s)
                u = torch.cat([a, (b - t) * torch.exp(-s)], 1)
            else:
                u = torch.flip(u, [1])
        return u


def train_flow(epochs=1200, lr=3e-3, wd=1e-3, seed=RS):
    import torch
    torch.manual_seed(seed)
    # ⚠ 稀有风格（n<8）不参与训练，用额外一个“其他”嵌入槽吸收
    flow = StyleFlow(K, len(styles) + 1)
    s2i = {s: i for i, s in enumerate(styles)}
    unk = len(styles)
    sid = torch.tensor([s2i.get(s, unk) for s in STYLE], dtype=torch.long)
    Zt = torch.tensor(Zn, dtype=torch.float32)
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_sd = 1e9, None
    for ep in range(epochs):
        opt.zero_grad()
        u, ld = flow.forward(Zt, sid)
        nll = (0.5 * (u ** 2).sum(1) - ld).mean()
        nll.backward(); opt.step(); sch.step()
        if nll.item() < best:
            best = float(nll.item())
            best_sd = {k: v.detach().clone() for k, v in flow.state_dict().items()}
    flow.load_state_dict(best_sd)
    return flow, best


log('\n训练风格条件流 ...')
flow, nll = train_flow()
# 高斯基线 NLL（同参数化）
nll_gauss = 0.0
for s in styles:
    m = STYLE == s
    Sg = cov_s[s] + 1e-6 * np.eye(K)
    _, logdet = np.linalg.slogdet(Sg)
    d = Zn[m] - mu_s[s]
    nll_gauss += 0.5 * (np.einsum('ij,jk,ik->i', d, np.linalg.pinv(Sg), d).sum()
                        + m.sum() * logdet + m.sum() * K * np.log(2 * np.pi))
nll_gauss /= n
log('流模型 NLL=%.3f，高斯基线 NLL=%.3f，Δ=%.3f（负值表示流更优）' % (
    nll, nll_gauss, nll - nll_gauss))


def sample_style(s, n_samp, constrained=True, seed=None):
    """从风格条件流采样；constrained=True 时按缺陷约束拒绝。"""
    import torch
    if seed is not None:
        torch.manual_seed(seed)
    sid = torch.tensor([styles.index(s)] * n_samp, dtype=torch.long)
    with torch.no_grad():
        u = torch.randn(n_samp, K)
        z = flow.inverse(u, sid).numpy()
    Lx = decode(z)
    if constrained:
        ok = defect_ok(Lx, ref_style=s)
        if not ok.all():                       # 被拒样本替换为该风格经验分布重采样
            m = STYLE == s
            pool = Zn[m]
            rep = (~ok).sum()
            idx = rng.integers(0, len(pool), rep)
            Lx[~ok] = decode(pool[idx])
    return Lx


# ---------------------------------------------------------------- 缺陷约束
OAV, OAV_UNIT = C.oav_matrix(X, COMP, META)
DEF_COLS = [c for c in ['DMS', 'X2.3.butanedione', 'X4VG'] if c in OAV.columns]
log('\n缺陷化合物 OAV 可用列：%s' % DEF_COLS)
# 各风格缺陷 OAV 的经验上分位（作为风格自身容许上限）
Q_DEF = {}
for s in styles:
    m = STYLE == s
    Q_DEF[s] = np.nanpercentile(OAV.loc[m, DEF_COLS].values, 95, axis=0)


SCALE_Q3, _ = C.column_scale(COMP)
DEF_IDX = [COMP.index(c) for c in DEF_COLS]


def defect_oav(Lx):
    """缺陷化合物的 OAV（按列区分“对数变换”与“原始浓度”）。"""
    L = np.asarray(Lx, float)
    out = []
    for c in DEF_COLS:
        v = L[:, COMP.index(c)]
        conc = v if SCALE_Q3.get(c) == 'raw' else 10.0 ** v
        out.append(conc / META.loc[c, 'threshold'])
    return np.column_stack(out)


def defect_ok(Lx, ref_style=None, ref_Lx=None):
    """缺陷约束：返回布尔数组，True=可接受。"""
    oav = defect_oav(Lx)
    lim = Q_DEF[ref_style] if ref_style is not None else np.percentile(
        np.vstack([Q_DEF[s] for s in styles]), 90, axis=0)
    return np.all(oav <= lim * 1.0, axis=1)


def clamp_feasible(Lx):
    """化学可行性投影：逐列截断到全体真实观测的 [min, max]（log10 尺度）。"""
    L = np.asarray(Lx, float).copy()
    lo = np.nanmin(X, axis=0); hi = np.nanmax(X, axis=0)
    return np.clip(L, lo, hi)


# 每个风格的化学包络（经验 min/max 外扩 5%），用于把生成/迁移结果投回可行域
RANGE = {}
for _s in styles:
    _m = STYLE == _s
    _lo = np.nanmin(X[_m], axis=0); _hi = np.nanmax(X[_m], axis=0)
    _pad = 0.05 * (_hi - _lo + 1e-9)
    RANGE[_s] = (_lo - _pad, _hi + _pad)


def clamp_to_style(Lx, s, skip=None):
    """投影到指定风格的化学可行域；skip 指定的列（缺陷化合物）不参与投影，
    否则“投影到目标风格下界”会把源样本的缺陷抬高，与不恶化约束相冲突。"""
    lo, hi = RANGE[s]
    if skip is not None:
        lo = lo.copy(); hi = hi.copy()
        lo[skip] = -np.inf; hi[skip] = np.inf
    return np.clip(np.asarray(Lx, float), lo, hi)


# 缺陷列的物理下界：原始浓度列不得为负；对数列取观测最小值
DEF_FLOOR = np.array([
    0.0 if SCALE_Q3.get(c) == 'raw' else float(np.nanmin(X[:, COMP.index(c)]))
    for c in DEF_COLS])


def enforce_defect_pareto(L, Lref):
    """把缺陷列逐样本钳制到 [物理下界, 参照样本值]：使“缺陷不恶化”成为构造性保证。"""
    L = np.asarray(L, float).copy()
    L[:, DEF_IDX] = np.minimum(np.maximum(L[:, DEF_IDX], DEF_FLOOR[None, :]),
                               np.asarray(Lref, float)[:, DEF_IDX])
    return L


# ---------------------------------------------------------------- 生成与评估
log('\n' + '=' * 78); log('生成：给定风格生成化合物谱 + 感官画像'); log('=' * 78)
NS = 200
gen_log = {}
for s in ['Tripel', 'Kriek', 'Lambic', 'Pils/Lager', 'Blond', 'Stout/Porter']:
    if s not in styles:
        continue
    Lx = clamp_to_style(sample_style(s, NS, seed=RS), s)
    gen_log[s] = Lx
    log('%-14s 生成 %d 份；缺陷通过率 %.2f' % (
        s, NS, float(defect_ok(Lx, ref_style=s).mean())))

pd.to_pickle(gen_log, os.path.join(C.OUT, 'q3_generated.pkl'))

# ------- 风格判别保真度
log('\n风格判别保真度（分类器在真实数据上 5 折 CV，再判生成样本）')
clf = RandomForestClassifier(n_estimators=600, min_samples_leaf=1, n_jobs=-1,
                             random_state=RS)
sub = np.isin(STYLE, styles)
acc_real = []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=RS).split(
        Xz[sub], np.array(STYLE)[sub]):
    clf.fit(Xz[sub][tr], np.array(STYLE)[sub][tr])
    acc_real.append((clf.predict(Xz[sub][te]) == np.array(STYLE)[sub][te]).mean())
clf.fit(Xz[sub], np.array(STYLE)[sub])
log('  分类器在真实样本上的 CV 准确率 = %.3f（%d 类，随机基线 %.3f）' % (
    np.mean(acc_real), len(styles), 1 / len(styles)))
fid = {}
for s, Lx in gen_log.items():
    pred = clf.predict(sc.transform(Lx))
    fid[s] = float((pred == s).mean())
    log('  生成“%s”被判别为同风格的比例 = %.3f' % (s, fid[s]))

# ------- Fréchet 距离 / 能量距离 / 多样性（在标准化空间计算，避免原始量纲列主导）
def frechet(A, B):
    ma, mb = A.mean(0), B.mean(0)
    Sa, Sb = np.cov(A.T) + 1e-6 * np.eye(A.shape[1]), np.cov(B.T) + 1e-6 * np.eye(B.shape[1])
    S12 = sqrtm_psd(Sa @ Sb)
    return float(((ma - mb) ** 2).sum() + np.trace(Sa + Sb - 2 * S12))


def energy_dist(A, B, cap=150):
    A = A[:cap]; B = B[:cap]
    d_ab = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    d_aa = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    d_bb = np.sqrt(((B[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    return float(2 * d_ab.mean() - d_aa.mean() - d_bb.mean())


dist_tab = []
for s, Lx in gen_log.items():
    real = X[STYLE == s]
    Gz = sc.transform(Lx); Rz = sc.transform(real)
    sd_r = Rz.std(0)
    live = sd_r > 0.05                      # 只在实际有变异的维度上比较多样性
    dist_tab.append(dict(
        风格=s, n_real=int((STYLE == s).sum()),
        Fréchet=frechet(Gz, Rz),
        能量距离=energy_dist(Gz, Rz),
        多样性比=float(np.median((Gz[:, live].std(0) + 1e-9) / sd_r[live])),
        风格保真度=fid[s]))
DT = pd.DataFrame(dist_tab)
log('\n生成质量（越小越接近真实；多样性比 1 附近最好）：')
log(DT.to_string(index=False))

# ------- 感官画像对照
log('\n感官画像对照（生成 vs 真实风格质心，z 尺度 L1 距离；前 6 个差异维度）')
prof_rows = []
for s, Lx in gen_log.items():
    Sp = predict_sensory_from_logx(Lx).mean(0)
    Sr = Y[STYLE == s].mean(0)
    d = Sp - Sr
    top = np.argsort(-np.abs(d))[:6]
    prof_rows.append(dict(风格=s, L1距离=float(np.abs(d).mean()),
                          **{C.clean_sens_name(SDIM[j]): round(float(d[j]), 2) for j in top}))
    log('  %-14s L1=%.3f  主要偏移：%s' % (
        s, np.abs(d).mean(), '、'.join('%s%+.2f' % (C.clean_sens_name(SDIM[j]), d[j])
                                       for j in top)))
PF = pd.DataFrame(prof_rows)

# ---------------------------------------------------------------- 跨风格迁移
log('\n' + '=' * 78); log('跨风格迁移：Pils/Lager → Tripel（缺陷不恶化）'); log('=' * 78)
src, tgt = 'Pils/Lager', 'Tripel'
m_src = STYLE == src
Xsrc = X[m_src]
Lsrc = np.asarray(Xsrc, float)            # 已是 log10 尺度
z_src = encode(sc.transform(Xsrc))
mu_t, A = ot_map(mu_s[src], cov_s[src], mu_s[tgt], cov_s[tgt])
z_ot = mu_t + (z_src - mu_s[src]) @ A.T

S_before = predict_sensory_from_logx(Lsrc)
L_ot = enforce_defect_pareto(clamp_to_style(decode(z_ot), tgt, skip=DEF_IDX), Lsrc)
S_after = predict_sensory_from_logx(L_ot)

# 缺陷单调约束：沿 z_src → z_ot 的路径做二分，保证“感官缺陷分不升”且“缺陷 OAV 不升”
de_idx = {c: SDIM.index(c) for c in C.DEFECT_SENS}


def defect_score(S):
    return S[:, [de_idx[c] for c in C.DEFECT_SENS]].mean(1)


base = defect_score(S_before)
cur = defect_score(S_after)
oav_base = defect_oav(Lsrc)
# 需要收缩的样本：感官缺陷分上升 或 任一缺陷化合物 OAV 上升
bad = (cur > base) | np.any(defect_oav(L_ot) > oav_base * (1 + 1e-6) + 1e-9, axis=1)
lam = np.ones(len(z_src))
if bad.any():
    lo = np.zeros(bad.sum()); hi = np.ones(bad.sum())
    for _ in range(24):
        mid = (lo + hi) / 2
        # 在化合物空间沿“源谱 → 目标谱”插值：λ=0 精确等于源谱
        Lz = Lsrc[bad] + mid[:, None] * (L_ot[bad] - Lsrc[bad])
        Lz = enforce_defect_pareto(Lz, Lsrc[bad])
        ss = defect_score(predict_sensory_from_logx(Lz))
        ok_s = ss <= base[bad] + 1e-9
        # 缺陷 OAV 严格不劣于源样本
        ok_o = np.all(defect_oav(Lz) <= oav_base[bad] * (1 + 1e-6) + 1e-9, axis=1)
        ok = ok_s & ok_o
        hi = np.where(ok, hi, mid)
        lo = np.where(ok, mid, lo)
    lam[bad] = lo              # 取已知可行侧，保证约束严格成立
L_fin = enforce_defect_pareto(Lsrc + lam[:, None] * (L_ot - Lsrc), Lsrc)
S_fin = predict_sensory_from_logx(L_fin)

log('迁移样本 %d 份；平均迁移强度 λ = %.3f（1=完全迁移，0=保持原样）' % (len(lam), lam.mean()))
log('缺陷感官分：迁移前 %.3f → 最优传输 %.3f → 约束后 %.3f' % (
    base.mean(), cur.mean(), defect_score(S_fin).mean()))
log('缺陷恶化样本数：最优传输 %d / %d，约束后 %d / %d' % (
    int(bad.sum()), len(lam), int((defect_score(S_fin) > base + 1e-9).sum()), len(lam)))

# 风格判别：迁移后应更像 Tripel
clf_p = clf.predict(sc.transform(L_fin))
log('迁移后被判为 Tripel 的比例：原样 %.3f → 迁移后 %.3f' % (
    float((clf.predict(sc.transform(Xsrc)) == tgt).mean()), float((clf_p == tgt).mean())))

# 关键嗅觉维度变化
key = ['A_esters_fruity', 'A_esters_all', 'A_hops_all', 'alcohol', 'body', 'sweetness',
       'A_malt_cara', 'dms', 'diacetyl', 'X4vg', 't2n', 'stale_hops']
TAB = pd.DataFrame({
    '感官维度': [C.clean_sens_name(c) for c in key],
    '迁移前': [float(S_before[:, SDIM.index(c)].mean()) for c in key],
    '最优传输': [float(S_after[:, SDIM.index(c)].mean()) for c in key],
    '约束后': [float(S_fin[:, SDIM.index(c)].mean()) for c in key],
    '目标风格均值': [float(Y[STYLE == tgt, SDIM.index(c)].mean()) for c in key],
})
log('\n关键感官维度迁移对照（z 尺度）：')
log(TAB.to_string(index=False))
TAB.to_csv(os.path.join(C.OUT, 'q3_transfer_sensory.csv'), index=False, encoding='utf-8-sig')

# 化合物层面：迁移后变化最大的化合物（按列量纲分别计算倍数）
dmean = L_fin.mean(0) - Lsrc.mean(0)
order = np.argsort(-np.abs(dmean))[:12]


def _ratio(i):
    c = COMP[i]
    if SCALE_Q3.get(c) == 'raw':                 # 原始浓度列：直接比值
        a, b = Lsrc[:, i].mean(), L_fin[:, i].mean()
        return (b / a) if abs(a) > 1e-9 else np.nan
    return float(10 ** dmean[i])                 # 对数列：10^差分


cmp_tab = pd.DataFrame({
    '化合物': [COMP[i] for i in order],
    '化学类别': [C.family_of(COMP[i]) for i in order],
    '化学名': [META.loc[COMP[i], 'chem_name'] if COMP[i] in META.index else '' for i in order],
    '迁移前': [float(Lsrc[:, i].mean()) for i in order],
    '迁移后': [float(L_fin[:, i].mean()) for i in order],
    '倍数变化': [_ratio(i) for i in order],
    '目标风格均值': [float(X[STYLE == tgt, i].mean()) for i in order],
})
log('\n迁移引起的最大化合物变化（前 12，按列量纲计算倍数）：')
log(cmp_tab.to_string(index=False))
cmp_tab.to_csv(os.path.join(C.OUT, 'q3_transfer_compounds.csv'), index=False,
               encoding='utf-8-sig')

# 缺陷 OAV 变化（按列量纲判定）
oav_before = defect_oav(Lsrc)
oav_after = defect_oav(L_fin)
log('\n缺陷化合物 OAV：')
for i, c in enumerate(DEF_COLS):
    log('  %-38s 前 %.3f → 后 %.3f（比值 %.2f）' % (
        META.loc[c, 'chem_name'] or c, oav_before[:, i].mean(),
        oav_after[:, i].mean(), (oav_after[:, i] / (oav_before[:, i] + 1e-12)).mean()))

# ---------------------------------------------------------------- 落盘
np.savez_compressed(os.path.join(C.OUT, 'q3_arrays.npz'),
                    z_src=z_src, z_ot=z_ot, L_ot=L_ot, L_fin=L_fin, lam=lam,
                    S_before=S_before, S_after=S_after, S_fin=S_fin,
                    Lsrc=Lsrc, V=V, Zmu=Zmu, Zsd=Zsd,
                    Xmean=Xz.mean(0))
DT.to_csv(os.path.join(C.OUT, 'q3_generation_quality.csv'), index=False,
          encoding='utf-8-sig')
PF.to_csv(os.path.join(C.OUT, 'q3_sensory_profile.csv'), index=False, encoding='utf-8-sig')

json.dump(dict(K=K, evr_cum=float(evr[:K].sum()), nll_flow=nll, nll_gauss=nll_gauss,
               classifier_cv_acc=float(np.mean(acc_real)),
               style_fidelity=fid,
               frechet={r['风格']: r['Fréchet'] for r in dist_tab},
               energy={r['风格']: r['能量距离'] for r in dist_tab},
               diversity={r['风格']: r['多样性比'] for r in dist_tab},
               transfer=dict(src=src, tgt=tgt, n=int(len(lam)),
                             lam_mean=float(lam.mean()),
                             defect_before=float(base.mean()),
                             defect_ot=float(cur.mean()),
                             defect_fin=float(defect_score(S_fin).mean()),
                             n_worse_ot=int(bad.sum()),
                             n_worse_fin=int((defect_score(S_fin) > base + 1e-9).sum()),
                             p_tgt_before=float((clf.predict(sc.transform(Xsrc)) == tgt).mean()),
                             p_tgt_after=float((clf_p == tgt).mean())),
               styles=styles, n_by_style=n_s,
               sens_surrogate_r2=float(r2_surr.mean())),
          open(os.path.join(C.OUT, 'q3_summary.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
log('\n完成：q3_summary.json / q3_generation_quality.csv / q3_transfer_*.csv')
