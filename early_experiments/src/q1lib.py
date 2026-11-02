# -*- coding: utf-8 -*-
"""q1lib.py —— 2026 MathorCup 赛道A 问题1 共享库

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek）。

内容：
  load()        统一读入 1.csv / 4.csv 并按 beer 对齐
  colscale()    逐列判定 1.csv 数值是 log10 变换值还是原始浓度（8.xlsx 校准表回归）
  prep()        折内特征工程（量纲统一 / 中位数插补 / 缩尾 / 标准化 / 缺失与封底指示 / 风格哑变量）
  CV 工具       按风格分层的 5 折 × R 重复交叉验证，pooled R²
"""
import json
import os
import sys
import time
import warnings

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'output')
FIG = os.path.join(ROOT, 'figs')
LOGD = os.path.join(ROOT, 'logs')
_TMP = os.path.join(ROOT, '.tmp')
for _d in (OUT, FIG, LOGD, _TMP):
    os.makedirs(_d, exist_ok=True)
for _k in ('JOBLIB_TEMP_FOLDER', 'TMP', 'TEMP'):
    os.environ[_k] = _TMP          # 本机路径含中文，joblib memmap 资源跟踪器会崩

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

RS = 20261027
N_SPLITS, N_REPEATS = 5, 3
META_CHEM = ('beer', 'beer_id', 'tasting_category_fine')
FLOOR = -1.81781854                # 1.csv 的检出限地板值（整列共用的常数）


def patch_joblib_sequential():
    """把 sklearn 内部的 joblib 并行执行器替换为顺序执行。

    本机沙箱禁止进程/线程池创建命名管道，`sklearn` 的 HistGradientBoosting 分箱、
    多输出封装与随机森林在 n_jobs>1 时会抛 `PermissionError: [WinError 5]`。
    顺序执行的结果与并行完全一致（只是慢），因此这里是纯粹的工程适配，
    不改变任何数值结论。
    """
    import sklearn.utils.parallel as sup

    class SeqParallel(sup.Parallel):
        """顺序执行版 Parallel：直接调用每个 delayed 任务，不建线程池。"""

        def __call__(self, iterable):
            out = []
            for item in iterable:
                if isinstance(item, tuple):
                    fn, args, kwargs = item[0], item[1], item[2]
                    out.append(fn(*args, **kwargs))
                else:
                    out.append(item())
            return out

    targets = []
    for modname in ('sklearn.multioutput',
                    'sklearn.ensemble._hist_gradient_boosting.binning',
                    'sklearn.ensemble._hist_gradient_boosting.gradient_boosting',
                    'sklearn.ensemble._forest',
                    'sklearn.linear_model._coordinate_descent',
                    'sklearn.linear_model._ridge'):
        try:
            import importlib
            m = importlib.import_module(modname)
            if hasattr(m, 'Parallel'):
                m.Parallel = SeqParallel
                targets.append(modname)
        except Exception:
            pass
    os.environ['JOBLIB_MULTIPROCESSING'] = '0'
    warnings.filterwarnings('ignore', message='.*should be used with.*Parallel.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    return targets


PATCHED = patch_joblib_sequential()


# ------------------------------------------------------------------ 读入
def load():
    chem = pd.read_csv(os.path.join(DATA, '1.csv'))
    sens = pd.read_csv(os.path.join(DATA, '4.csv'))
    chem = chem.sort_values('beer').reset_index(drop=True)
    sens = sens.sort_values('beer').reset_index(drop=True)
    assert np.array_equal(chem['beer'].values, sens['beer'].values), 'beer 序号不一致'
    comp = [c for c in chem.columns if c not in META_CHEM]
    sdim = [c for c in sens.columns if c not in META_CHEM]
    X = chem[comp].apply(pd.to_numeric, errors='coerce')     # 'NA' → NaN
    Y = sens[sdim].apply(pd.to_numeric, errors='coerce')
    return dict(chem=chem, sens=sens, comp=comp, sdim=sdim, X=X, Y=Y,
                style=chem['tasting_category_fine'].astype(str).values,
                beer=chem['beer'].values)


# ------------------------------------------------------------------ 量纲判定
def colscale(refresh=False):
    """逐列判定 'log10'（对数值）或 'raw'（原始浓度）。

    判据：与 8.xlsx 'Calibrated chemical dataset' 的校准浓度**逐样本**比对（同一批 250 样本）。
      * 若 10^v ≈ 校准浓度 的样本比例 > 0.95  →  该列是 log10 值
      * 若 v   ≈ 校准浓度 的样本比例 > 0.95  →  该列是原始浓度
    用"逐样本命中比例"而非"中位相对偏差"：后者会被大量近乎 0 的微量值压成 0，
    在 DMS / 乳酸 / kcal 这类跨越数量级的列上会误判（实测过）。
    无校准信息的列默认按 log10 处理（其值域 −4.5~6 与其余对数列一致，且多数列有负值）。
    """
    fp = os.path.join(OUT, 'colscale.json')
    if os.path.exists(fp) and not refresh:
        d = json.load(open(fp, encoding='utf-8'))
        return d['scale'], d['detail']
    D = load()
    cal = pd.read_excel(os.path.join(DATA, '8.xlsx'),
                        sheet_name='Calibrated chemical dataset')
    cal.columns = [str(c).strip() for c in cal.columns]
    # 8.xlsx 的行序与 4.csv 原始顺序一致，必须先按 beer 对齐再用！（否则逐样本比对全错）
    cal = cal.sort_values('beer').reset_index(drop=True)
    assert np.array_equal(pd.to_numeric(cal['beer']).values.astype(np.int64),
                          D['beer'].astype(np.int64)), '8.xlsx 与 1.csv 的 beer 序号不一致'
    scale, detail = {}, {}
    for c in D['comp']:
        v = D['X'][c].values.astype(float)
        if c in cal.columns:
            s = pd.to_numeric(cal[c], errors='coerce').values.astype(float)
            m = np.isfinite(v) & np.isfinite(s)
            if m.sum() >= 20:
                vv, ss = v[m], s[m]
                # 用"绝对容差 + 相对容差"混合判据：校准表里有大量 0 值，
                # 纯相对偏差在 0 处会爆掉，必须加绝对项。
                rel = lambda a, b: np.abs(a - b) / np.maximum(np.abs(b), 1.0)
                f_pow = float(np.mean(rel(np.power(10.0, np.clip(vv, -300, 300)), ss) < 0.01))
                f_id = float(np.mean(rel(vv, ss) < 0.01))
                if f_pow > 0.95 and f_pow >= f_id:
                    tag = 'log10'
                elif f_id > 0.95:
                    tag = 'raw'
                else:
                    tag = 'log10' if (vv < 0).any() else ('log10' if f_pow >= f_id else 'raw')
                detail[c] = dict(tag=tag, f_pow=f_pow, f_id=f_id, calibrated=True,
                                 n_pair=int(m.sum()))
                scale[c] = tag
                continue
        detail[c] = dict(tag='log10', f_pow=None, f_id=None, calibrated=False, n_pair=0)
        scale[c] = 'log10'
    json.dump(dict(scale=scale, detail=detail), open(fp, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    return scale, detail


def harmonize(X, comp, scale):
    """量纲统一：把 33 列原始浓度取 log10，使 231 列处于同一动态范围。

    说明：标准化后的线性模型对单调变换敏感（重尾列会把量级拉到 1e4），
    这一步是纯单调变换，不改变任何树模型的排序，只让线性/距离类模型可用。
    """
    Z = X.copy()
    for c in comp:
        if scale.get(c) == 'raw':
            v = Z[c].values.astype(float)
            pos = v[np.isfinite(v) & (v > 0)]
            eps = max(0.5 * (pos.min() if pos.size else 1e-6), 1e-9)
            Z[c] = np.log10(np.maximum(v, eps))
    return Z


def censoring(X, comp, frac=0.05):
    """检出限（左删失）统计：返回 (封底指示列清单, 统计表)。"""
    V = X[comp].values.astype(float)
    eq = np.isclose(V, FLOOR, atol=1e-9)
    finite = np.isfinite(V)
    colfrac = eq.sum(0) / np.maximum(finite.sum(0), 1)
    mask = colfrac >= frac
    return [comp[i] for i in np.where(mask)[0]], colfrac


# ------------------------------------------------------------------ 特征工程
def prep(Xtr, Xte, scale, cols, opt):
    """折内特征工程。opt: use_dummy / use_style / use_graph / winsor。

    所有统计量只在训练折上估计，避免信息泄漏。
    """
    from sklearn.impute import SimpleImputer
    comp = list(Xtr.columns)
    A = Xtr[comp].values.astype(float)
    B = Xte[comp].values.astype(float)

    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(A)
    A, B = imp.transform(A), imp.transform(B)
    if opt.get('winsor', True):
        lo, hi = np.nanpercentile(A, [1, 99], axis=0)
        A, B = np.clip(A, lo, hi), np.clip(B, lo, hi)
    mu, sd = np.nanmean(A, 0), np.nanstd(A, 0)
    sd = np.where(sd > 0, sd, 1.0)
    A, B = (A - mu) / sd, (B - mu) / sd
    blocks = [A, B]

    if opt.get('use_dummy', True):
        # 缺失指示（9 列）
        mc = np.where(np.isnan(Xtr[comp].values.astype(float)).any(0))[0]
        if len(mc):
            blocks[0] = np.hstack([blocks[0],
                                   np.isnan(Xtr[comp].values.astype(float))[:, mc].astype(float)])
            blocks[1] = np.hstack([blocks[1],
                                   np.isnan(Xte[comp].values.astype(float))[:, mc].astype(float)])
        # 封底（检出限）指示：把"是否检出"与"检出后浓度多少"分开
        eq_tr = np.isclose(Xtr[comp].values.astype(float), FLOOR, atol=1e-9).astype(float)
        eq_te = np.isclose(Xte[comp].values.astype(float), FLOOR, atol=1e-9).astype(float)
        kc = np.where(eq_tr.mean(0) >= 0.05)[0]
        if len(kc):
            blocks[0] = np.hstack([blocks[0], eq_tr[:, kc]])
            blocks[1] = np.hstack([blocks[1], eq_te[:, kc]])

    if opt.get('use_graph', False):
        S = opt['graph_op']
        p0 = len(comp)
        blocks[0] = np.hstack([blocks[0], blocks[0][:, :p0] @ S.T])
        blocks[1] = np.hstack([blocks[1], blocks[1][:, :p0] @ S.T])

    if opt.get('use_style', False):
        cats = opt['style_cats']
        otr = np.zeros((len(Xtr), len(cats)))
        ote = np.zeros((len(Xte), len(cats)))
        si = {c: i for i, c in enumerate(cats)}
        for i, s in enumerate(opt['style_tr']):
            otr[i, si[s]] = 1.0
        for i, s in enumerate(opt['style_te']):
            ote[i, si[s]] = 1.0
        blocks[0] = np.hstack([blocks[0], otr])
        blocks[1] = np.hstack([blocks[1], ote])
    return blocks[0], blocks[1]


def build_graph_op(comp, topk=8, alpha=0.35):
    """2.csv 共表达图的扩散算子 S = (1−α)I + α·Â（仅作特征块消融用）。"""
    A = pd.read_csv(os.path.join(DATA, '2.csv'), index_col=0)
    A.columns = [str(c).strip() for c in A.columns]
    nm2 = {str(c): i for i, c in enumerate(comp)}
    # 2.csv 用规范化学名，1.csv 用清理过的列名；用 8.xlsx 的 chemical_name 对齐
    th = pd.read_excel(os.path.join(DATA, '8.xlsx'), sheet_name='Sensory thresholds')
    alias = {}
    for _, r in th.iterrows():
        fn = str(r['field_name_python']).strip()
        cn = str(r['chemical_name']).strip()
        if fn in nm2 and cn and cn != 'nan':
            alias.setdefault(cn.lower(), fn)
    pos = []
    for c in A.columns:
        key = c.strip().lower()
        f = alias.get(key) or (key if key in nm2 else None)
        pos.append(nm2.get(f, -1) if f else -1)
    n = len(comp)
    M = np.zeros((n, n))
    idx = [(i, j) for i, p in enumerate(pos) if p >= 0 for j in range(len(pos))
           if pos[j] >= 0]
    for i, p in enumerate(pos):
        if p < 0:
            continue
        for j, q in enumerate(pos):
            if q >= 0 and i != j:
                M[p, q] = abs(float(A.iloc[i, j]))
    np.fill_diagonal(M, 0.0)
    k = min(topk, n - 1)
    top = np.argpartition(-M, k, axis=1)[:, :k]
    W = np.zeros_like(M)
    for i in range(n):
        W[i, top[i]] = M[i, top[i]]
    W = np.maximum(W, W.T)
    deg = W.sum(1)
    dinv = np.where(deg > 0, 1 / np.sqrt(deg), 0.0)
    return (1 - alpha) * np.eye(n) + alpha * (W * dinv[:, None]) * dinv[None, :]


# ------------------------------------------------------------------ 指标
def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else np.nan


def per_dim_r2(Y, P):
    return np.array([r2(Y[:, j], P[:, j]) for j in range(Y.shape[1])])


# ------------------------------------------------------------------ 维度中文名 / 分组
SENS_ZH = {
    'A_malt_all': '麦芽香-总体', 'A_malt_grain': '麦芽香-谷物', 'A_malt_bread': '麦芽香-面包',
    'A_malt_cara': '麦芽香-焦糖', 'A_malt_burn': '麦芽香-烟熏焦',
    'A_hops_all': '酒花香-总体', 'A_hops_citrus': '酒花香-柑橘',
    'A_hops_tropical': '酒花香-热带果', 'A_hops_noble': '酒花香-草本贵族',
    'A_hops_woody': '酒花香-木质',
    'A_esters_all': '酯香-总体', 'A_esters_ethac': '酯香-乙酸乙酯',
    'A_esters_isoaa': '酯香-乙酸异戊酯', 'A_esters_flower': '酯香-花香',
    'A_esters_fruity': '酯香-果香',
    'F_malt_all': '麦芽味-总体', 'F_malt_grain': '麦芽味-谷物', 'F_malt_bread': '麦芽味-面包',
    'F_malt_cara': '麦芽味-焦糖', 'F_malt_burn': '麦芽味-烟熏焦',
    'F_hops_all': '酒花味-总体', 'F_hops_citrus': '酒花味-柑橘',
    'F_hops_tropical': '酒花味-热带果', 'F_hops_noble': '酒花味-草本贵族',
    'F_hops_woody': '酒花味-木质',
    'F_esters_all': '酯味-总体', 'F_esters_ethac': '酯味-乙酸乙酯',
    'F_esters_isoaa': '酯味-乙酸异戊酯', 'F_esters_flower': '酯味-花香',
    'F_esters_fruity': '酯味-果香',
    'acidity': '酸度', 'bitternes': '苦度', 'sweetness': '甜度', 'X4vg': '4-VG 丁香',
    'diacetyl': '双乙酰', 'dms': 'DMS 蒸煮蔬菜', 'metallic': '金属味',
    'stale_hops': '陈酒花', 't2n': '反式-2-壬烯醛', 'orange': '橙香',
    'coriander': '芫荽', 'clove': '丁香酚', 'lactic': '乳酸', 'acetic': '乙酸',
    'barnyard': '马厩 Brett', 'alcohol': '醇感', 'aftertaste': '回味', 'body': '酒体',
    'co2': '碳酸感', 'overall': '总体喜好',
}

GROUP = {}
for _c in SENS_ZH:
    GROUP[_c] = ('Aroma' if _c.startswith('A_') else
                 'Flavour' if _c.startswith('F_') else
                 'Fault' if _c in ('dms', 'diacetyl', 't2n', 'stale_hops', 'metallic',
                                   'X4vg') else
                 'Mouthfeel' if _c in ('body', 'bitternes', 'co2', 'aftertaste',
                                       'alcohol', 'sweetness', 'acidity') else
                 'Spice/野生' if _c in ('orange', 'coriander', 'clove', 'lactic',
                                        'acetic', 'barnyard') else 'Overall')


# ------------------------------------------------------------------ CV
def strat_labels(style, n_splits=N_SPLITS):
    vc = pd.Series(style).value_counts()
    return np.array([s if vc[s] >= n_splits else 'Rare' for s in style])


def run_cv(models, R=None, repeats=None, tag=''):
    """按风格分层 5 折 × repeats 重复；返回 (oof[R,n,50], 逐模型 R², 耗时表)。

    models: dict[name] = (fit_fn, opt)
    折内预处理 → fit_fn(Xtr, Ytr, Xte, opt) → 预测。
    """
    from sklearn.model_selection import StratifiedKFold
    D = load()
    scale, _ = colscale()
    Xh = harmonize(D['X'], D['comp'], scale)
    Y = D['Y'].values.astype(float)
    style = D['style']
    strat = strat_labels(style)
    reps = repeats or N_REPEATS
    cats = sorted(set(style))
    oof = {k: np.full((reps, len(Y), Y.shape[1]), np.nan) for k in models}
    tt = {k: 0.0 for k in models}
    for rep in range(reps):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS + rep)
        for fold, (tr, te) in enumerate(skf.split(Xh, strat)):
            for name, (fn, opt0) in models.items():
                opt = dict(opt0)
                opt.update(style_cats=cats, style_tr=style[tr], style_te=style[te])
                t0 = time.time()
                A, B = prep(Xh.iloc[tr], Xh.iloc[te], scale, D['comp'], opt)
                P = fn(A, Y[tr], B, opt)
                oof[name][rep][te] = P
                tt[name] += time.time() - t0
                print('[%s] rep%d fold%d %-14s %6.1fs' % (tag, rep, fold, name,
                                                          time.time() - t0), flush=True)
    res = {}
    for name in models:
        pred = np.nanmean(oof[name], axis=0)
        r = per_dim_r2(Y, pred)
        res[name] = dict(mean_r2=float(np.nanmean(r)), median_r2=float(np.median(r)),
                         min_r2=float(np.nanmin(r)), max_r2=float(np.nanmax(r)),
                         per_dim=r, seconds=round(tt[name], 1))
    return oof, res, Y
