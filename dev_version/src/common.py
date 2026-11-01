# -*- coding: utf-8 -*-
"""common.py —— 赛道A 全流程共享模块

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。

内容：
  * load()            统一读入并对齐
  * COMP_META         化合物元数据（化学类别 / 来源 / 阈值 / 异味标记 / 中文名）
  * DEFECT            缺陷化合物集合
  * calib06()         0-6 分制标定与反标定
  * metrics()         R2 / MAE / Spearman
  * setup_style()     科研出版级绘图风格（沿用 AutoMCM-PlotSci）
"""
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'output')
FIG = os.path.join(ROOT, 'figs')
for _d in (OUT, FIG):
    os.makedirs(_d, exist_ok=True)

CHEM_META_COLS = ('beer', 'beer_id', 'tasting_category_fine')
SENS_META_COLS = ('beer', 'beer_id', 'tasting_category_fine')

# ------------------------------------------------------------------ 读入
def load():
    chem = pd.read_csv(os.path.join(DATA, '1.csv'))
    sens = pd.read_csv(os.path.join(DATA, '4.csv'))
    chem = chem.sort_values('beer').reset_index(drop=True)
    sens = sens.sort_values('beer').reset_index(drop=True)
    assert np.array_equal(chem['beer'].values, sens['beer'].values)
    comp = [c for c in chem.columns if c not in CHEM_META_COLS]
    sdim = [c for c in sens.columns if c not in SENS_META_COLS]
    d = dict(chem=chem, sens=sens, comp=comp, sdim=sdim,
             X=chem[comp].astype(float), Y=sens[sdim].astype(float),
             style=chem['tasting_category_fine'].values,
             beer=chem['beer'].values)
    return d


# ------------------------------------------------------------------ 0-6 标定
def calib06(Y):
    """逐维 min-max 映射到官方 0-6 分制；返回 (Y06, spec)。"""
    spec = {}
    Y06 = pd.DataFrame(index=Y.index)
    for c in Y.columns:
        lo, hi = float(Y[c].min()), float(Y[c].max())
        spec[c] = (lo, hi)
        Y06[c] = 6.0 * (Y[c] - lo) / (hi - lo)
    return Y06, spec


def inv_calib06(Z, spec):
    out = pd.DataFrame(index=Z.index)
    for c in Z.columns:
        lo, hi = spec[c]
        out[c] = lo + Z[c] * (hi - lo) / 6.0
    return out


# ------------------------------------------------------------------ 化合物元数据
# 化学类别判定：按名称模式（顺序敏感）
_FAMILY_RULES = [
    ('酯类', r'acetate|butyrate|hexanoate|octanoate|decanoate|dodecanoate|propionate|'
             r'isobutyrate|isovalerate|benzoate|cinnamate|formate|furoate|tiglate|'
             r'succinate|salicylate|laurate|hexadecanoate|octadecanoate|tetradecanoate|'
             r'pentadecanoate|nicotinate|hydrocinnamate|anisate|geranate|phenylacetate|'
             r'pentanoate|lactate|esters_sum|isobutyrate'),
    ('硫化物', r'^DMS$|^DES$|^H2S$|^MeSH$|^CS2$|SO2|sulfur_sum|methylthio|thiol'),
    ('萜烯类', r'terpineol|terpinen|linalool|geraniol|citronellol|nerol|nerolidol|limonene|'
              r'phellandrene|humulene|caryophyllene|farnes|ocimene|camphor|isoborneol|'
              r'eudesmol|muurolol|cadinol|spathulenol|valencene|germacrene|aromadendrene|'
              r'cubenol|calacorene|manool|ambroxide|damascenone|ionone|geranyl|thujene|'
              r'prenyl|phellandrene|terpenyl|ocimenone|citral|cyclohomocitral'),
    ('醛类', r'acetaldehyde|methylbutanal|benzaldehyde|phenylacetaldehyde|nonanal|dodecanal|'
            r'nonenal|butenal|cinnamaldehyde|cumin|salicylaldehyde|apofarnesal|farnesal'),
    ('酮类', r'butanedione|hepten.2.one|undecanone|car.3.en.2.one|furan.3.one|acetylpyrrole|isomaltol'),
    ('醇类', r'alcohol$|alcohol\b|^glycerol$|\.ol$|hexenol|octen|hexanol|octanol|nonanol|'
            r'decanol|dodecanol|tetradecanol|pentanol|methyl.1.octanol|nonan.2.ol'),
    ('酚类', r'X4VG|guaiacol|cresol|eugenol|vanillin|isoeugenol'),
    ('呋喃类', r'furfural|furan|benzofuran'),
    ('酸类', r'acetic_acid|lactic_acid|acids_sum'),
    ('吡嗪类', r'pyrazine'),
    ('内酯类', r'lactone'),
    ('理化参数', r'^pH$|color\.EBC|^bitterness$|CO2\.PSI|ethanol\.\.v|protein\.g|^iron$|'
                r'^ammonia|betaglucan|^GFS$|unfermentables|^kcalperc$|propylene\.glycol|'
                r'butylated\.hydroxytoluene|aroma_hops_sum'),
]


def family_of(name):
    n = str(name)
    for fam, pat in _FAMILY_RULES:
        if re.search(pat, n, flags=re.I):
            return fam
    return '其他'


# 阈值表 → 1.csv 列名的显式别名（人工核对，覆盖全部有阈值的化合物）
ALIAS = {
    'acetic acid.g.L.': 'acetic_acid.g.L.', 'bitterness': 'bitterness',
    'lactic acid': 'lactic_acid.mg.L.', 'ethanol..v.v.': 'ethanol..v.v.',
    'isoamyl alcohol': 'isoamyl_alcohol', 'isobutanol': 'isobutanol',
    'phenethyl alcohol': 'phenethyl_alcohol', 'acetaldehyde': 'acetaldehyde',
    'furfural': 'furfural', 'glycerol': 'glycerol', 'GFS': 'GFS',
    'ethyl 2-methyl butyrate': 'ethyl_2.methyl_butyrate',
    'ethyl isovalerate': 'ethyl_isovalerate', 'ethyl acetate': 'ethyl_acetate',
    'ethyl butyrate': 'ethyl.butyrate', 'ethyl decanoate': 'ethyl_decanoate',
    'ethyl hexanoate': 'ethyl_hexanoate', 'ethyl isobutyrate': 'ethyl.isobutyrate',
    'ethyl nicotinate': 'ethyl.nicotinate', 'ethyl octanoate': 'ethyl_octanoate',
    'ethyl propionate': 'ethyl_propionate', 'isopentyl acetate': 'isopentyl_acetate',
    'isobutyl acetate': 'isobutyl_acetate', 'octyl acetate': 'octyl_acetate',
    'phenethyl acetate': 'phenethyl_acetate', '2,3-butanedione': 'X2.3.butanedione',
    'ammonia': 'ammonia.mg.L.', 'iron': 'iron', 'protein.g.L.': 'protein.g.L.',
    'X4VG': 'X4VG', 'methylpyrazine': 'methylpyrazine', 'CS2': 'CS2', 'DES': 'DES',
    'DMS': 'DMS', 'H2S': 'H2S', 'SO2-mg.L.': 'SO2.mg.L.',
    'alpha-humulene': 'alpha.humulene', '1-alpha-terpineol': 'X1.alpha.terpineol',
    'E-caryophyllene': 'E.caryophyllene', 'citronellol': 'citronellol',
    'geraniol': 'geraniol', 'linalool': 'linalool', 'Z-nerolidol': 'Z.nerolidol',
    '(E)-2-nonenal': 'X.E..2.nonenal', '1-octen-3-ol': 'X1.octen.3.ol',
    '1-propanol': 'X1.propanol', '2-methyl-1-butanol': 'X2.methyl.1.butanol',
    '(E)-beta-damascenone': 'X.E..beta.damascenone', 'alpha-ionone': 'X.E..alpha.ionone',
    'beta-ionone': 'X.E..beta.ionone', 'phenethyl alcohol': 'phenethyl_alcohol',
    '4-ethyl-guaiacol': 'X4.ethyl.guaiacol', 'guaiacol': None,
}


def build_meta(comp):
    """构建化合物元数据表：中文别名、类别、来源、阈值、异味。"""
    th = pd.read_excel(os.path.join(DATA, '8.xlsx'), sheet_name='Sensory thresholds')
    th['field_name_python'] = th['field_name_python'].astype(str).str.strip()
    row = {}
    for _, r in th.iterrows():
        k = ALIAS.get(r['field_name_python'], r['field_name_python'])
        if k is None:
            k = r['field_name_python']
        row[k] = r
    cat_src = {}
    for _, r in th.iterrows():
        k = ALIAS.get(r['field_name_python'], r['field_name_python'])
        if k is None:
            continue
        cat_src[k] = r
    recs = []
    for c in comp:
        r = row.get(c)
        recs.append(dict(
            compound=c,
            family=family_of(c),
            chem_name=(r['chemical_name'] if r is not None else ''),
            origin=(r['origin_paper'] if r is not None else ''),
            unit=(r['unit'] if r is not None else ''),
            threshold=(float(r['threshold_mean']) if (r is not None and
                       np.isfinite(pd.to_numeric(r['threshold_mean'], errors='coerce'))) else np.nan),
            off_flavor=(r['off_flavors'] if (r is not None and
                        isinstance(r['off_flavors'], str)) else ''),
        ))
    M = pd.DataFrame(recs).set_index('compound')
    return M


# 缺陷化合物（题目点名 + 数据集可对应的标志物）
DEFECT = {
    'DMS': dict(name='二甲基硫', th=17.25, unit='μg/L', note='蒸煮蔬菜/玉米'),
    'X2.3.butanedione': dict(name='双乙酰', th=0.29925, unit='mg/L', note='奶油/黄油'),
    'X4VG': dict(name='4-乙烯基愈创木酚', th=0.20633, unit='mg/L', note='丁香/药味'),
    'X.E..2.nonenal': dict(name='反式-2-壬烯醛', th=np.nan, unit='mg/L', note='纸板/陈化'),
    'X3.methylthio.propanol': dict(name='3-甲硫基丙醇', th=np.nan, unit='mg/L', note='陈酒花/熟土豆'),
    'ethyl.isobutyrate': dict(name='异丁酸乙酯', th=2.6, unit='mg/L', note='陈酒花/奶酪'),
}
DEFECT_SENS = ['dms', 'diacetyl', 'X4vg', 't2n', 'stale_hops']

# 感官维度中文名
SENS_ZH = {
    'A_malt_all': '麦芽香-总体', 'A_malt_grain': '麦芽香-谷物', 'A_malt_bread': '麦芽香-面包',
    'A_malt_cara': '麦芽香-焦糖', 'A_malt_burn': '麦芽香-烟熏焦',
    'A_hops_all': '酒花香-总体', 'A_hops_citrus': '酒花香-柑橘',
    'A_hops_tropical': '酒花香-热带水果', 'A_hops_noble': '酒花香-草本贵族',
    'A_hops_woody': '酒花香-木质树脂',
    'A_esters_all': '酯香-总体', 'A_esters_ethac': '酯香-乙酸乙酯',
    'A_esters_isoaa': '酯香-乙酸异戊酯', 'A_esters_flower': '酯香-花香',
    'A_esters_fruity': '酯香-果香',
    'F_malt_all': '麦芽味-总体', 'F_malt_grain': '麦芽味-谷物', 'F_malt_bread': '麦芽味-面包',
    'F_malt_cara': '麦芽味-焦糖', 'F_malt_burn': '麦芽味-烟熏焦',
    'F_hops_all': '酒花味-总体', 'F_hops_citrus': '酒花味-柑橘',
    'F_hops_tropical': '酒花味-热带水果', 'F_hops_noble': '酒花味-草本贵族',
    'F_hops_woody': '酒花味-木质树脂',
    'F_esters_all': '酯味-总体', 'F_esters_ethac': '酯味-乙酸乙酯',
    'F_esters_isoaa': '酯味-乙酸异戊酯', 'F_esters_flower': '酯味-花香',
    'F_esters_fruity': '酯味-果香',
    'acidity': '酸度', 'bitternes': '苦度', 'sweetness': '甜度', 'X4vg': '4-VG 丁香',
    'diacetyl': '双乙酰奶油', 'dms': 'DMS 蒸煮蔬菜', 'metallic': '金属味',
    'stale_hops': '陈酒花', 't2n': '反式-2-壬烯醛纸板', 'orange': '橙香',
    'coriander': '芫荽', 'clove': '丁香酚', 'lactic': '乳酸', 'acetic': '乙酸',
    'barnyard': '马厩Brett', 'alcohol': '醇感', 'aftertaste': '回味', 'body': '酒体',
    'co2': '碳酸感', 'overall': '总体喜好',
}
SENS_EN = {  # 图件用英文短名
    'A_malt_all': 'Malt aroma', 'A_malt_grain': 'Grain', 'A_malt_bread': 'Bread',
    'A_malt_cara': 'Caramel', 'A_malt_burn': 'Smoky', 'A_hops_all': 'Hops aroma',
    'A_hops_citrus': 'Citrus', 'A_hops_tropical': 'Tropical', 'A_hops_noble': 'Noble',
    'A_hops_woody': 'Woody', 'A_esters_all': 'Esters aroma', 'A_esters_ethac': 'Ethyl acetate',
    'A_esters_isoaa': 'Isoamyl acetate', 'A_esters_flower': 'Floral',
    'A_esters_fruity': 'Fruity', 'F_malt_all': 'Malt taste', 'F_malt_grain': 'Grain',
    'F_malt_bread': 'Bread', 'F_malt_cara': 'Caramel', 'F_malt_burn': 'Smoky',
    'F_hops_all': 'Hops taste', 'F_hops_citrus': 'Citrus', 'F_hops_tropical': 'Tropical',
    'F_hops_noble': 'Noble', 'F_hops_woody': 'Woody', 'F_esters_all': 'Esters taste',
    'F_esters_ethac': 'Ethyl acetate', 'F_esters_isoaa': 'Isoamyl acetate',
    'F_esters_flower': 'Floral', 'F_esters_fruity': 'Fruity', 'acidity': 'Acidity',
    'bitternes': 'Bitterness', 'sweetness': 'Sweetness', 'X4vg': '4-VG', 'diacetyl': 'Diacetyl',
    'dms': 'DMS', 'metallic': 'Metallic', 'stale_hops': 'Stale hops', 't2n': 'trans-2-nonenal',
    'orange': 'Orange', 'coriander': 'Coriander', 'clove': 'Clove', 'lactic': 'Lactic',
    'acetic': 'Acetic', 'barnyard': 'Barnyard', 'alcohol': 'Alcohol', 'aftertaste': 'Aftertaste',
    'body': 'Body', 'co2': 'Carbonation', 'overall': 'Overall',
}
SENS_GROUP = {c: ('Aroma' if c.startswith('A_') else
                  'Flavour' if c.startswith('F_') else
                  'Mouthfeel' if c in ('body', 'bitternes', 'co2', 'aftertaste', 'alcohol',
                                       'sweetness', 'acidity') else
                  'Fault' if c in ('dms', 'diacetyl', '4vg', 'X4vg', 't2n', 'stale_hops',
                                   'metallic') else
                  'Spice/野生' if c in ('orange', 'coriander', 'clove', 'lactic', 'acetic',
                                        'barnyard') else 'Overall')
             for c in SENS_ZH}


def clean_sens_name(c):
    return SENS_ZH.get(c, c)


def column_scale(comp=None):
    """判定 1.csv 每一列是“log10 变换”还是“原始浓度”。

    ⚠ 附件说明只提到"少量理化参数列保留原始量纲"，但实测 231 列中有 11 列
    **同样是原始浓度**（β-大马酮、γ-壬内酯、双乙酰、反式-2-壬烯醛等
    µg/L 级超痕量致香物），必须逐列判定。
    判据：用 8.xlsx 校准浓度表回归 ——
      * log10(con) = a·v + b，若 a≈1 且 R²>0.95 → log10 变换；
      * 若 median(con / v) ∈ [0.5, 2] → 原始浓度。
    返回 dict(compound → 'log10' | 'raw') 与未判定列清单。
    """
    if comp is None:
        comp = load()['comp']
    cal = pd.read_excel(os.path.join(DATA, '8.xlsx'), sheet_name='Calibrated chemical dataset')
    cal.columns = [str(c).strip() for c in cal.columns]
    out, unknown, detail = {}, [], {}
    X = None
    for c in comp:
        if c in cal.columns:
            if X is None:
                X = load()['X']
            v = X[c].values.astype(float)
            con = cal[c].values.astype(float)
            m = np.isfinite(v) & np.isfinite(con) & (con > 0)
            if m.sum() >= 20:
                vv, cc = v[m], con[m]
                # 直接检验两种恒等关系（数据精度内应为 0）
                d_log = float(np.median(np.abs(np.log10(cc) - vv)))
                d_raw = float(np.median(np.abs(cc - vv) / np.maximum(np.abs(vv), 1e-9)))
                detail[c] = dict(d_log=d_log, d_raw=d_raw)
                has_neg = bool((vv < 0).any())
                if d_log <= d_raw and d_log < 0.05:
                    out[c] = 'log10'
                elif d_raw < 0.05 and not has_neg:
                    out[c] = 'raw'
                else:                       # 兜底：有负值必为对数
                    out[c] = 'log10' if has_neg else ('raw' if d_raw < d_log else 'log10')
                continue
        out[c] = 'log10'          # 无校准信息的列默认按对数变换处理
        if c not in cal.columns:
            unknown.append(c)
    return out, unknown


def to_concentration(v, c, scale):
    """1.csv 数值 → 真实浓度（按列判定结果）。"""
    return np.asarray(v, float) if scale.get(c) == 'raw' else 10.0 ** np.asarray(v, float)


def oav_matrix(X, comp, meta):
    """由 1.csv 数值还原真实浓度并计算 OAV（仅含阈值可用且单位一致的化合物）。"""
    scale, _ = column_scale(comp)
    cols, units = [], []
    for c in comp:
        if c not in meta.index:
            continue
        thr = meta.loc[c, 'threshold']
        u = meta.loc[c, 'unit']
        if not np.isfinite(thr) or thr <= 0:
            continue
        if u in ('IBU', 'v/v', 'A.U.', '', 'nan'):
            continue
        cols.append(c); units.append(u)
    Xv = np.asarray(X, float)
    idx = [comp.index(c) for c in cols]
    conc = np.column_stack([to_concentration(Xv[:, j], c, scale)
                            for j, c in zip(idx, cols)])
    thr = np.array([meta.loc[c, 'threshold'] for c in cols])
    return pd.DataFrame(conc / thr, columns=cols), pd.Series(units, index=cols)


# ------------------------------------------------------------------ 2.csv 名称映射
# 2.csv 用规范化学名（Acetaldehyde / Carbon disulfide ...），1.csv 用乱码列名；
# 其中 11 个名称存在拼写差异（Terpiryl / Tobacarol / Kongol ...），需人工对齐。
MAP2_OVERRIDE = {
    'Carbon dioxide': 'CO2.PSI.', 'Cis-3-hexenol': 'X3Z.hexenol',
    'Furfuryl alcohol': 'X2.furanmethanol', 'Methionol': 'X3.methylthio.propanol',
    'O-cresol': 'ortho.cresol', 'Terpiryl acetate': 'delta.terpinyl.acetate',
    'Tobacarol': 'dihydroisocaryophyllene.epoxide', 'Epicubenol': 'X1.10.di.epi.cubenol',
    'Caryophylla-4,8-dien-5-ol': 'caryophylla.4.12..8.13..dien.5.alpha.ol',
    'Kongol': 'selin.11.en.4.alpha.ol',
    'Ethyl 3-hexanoate': 'ethyl.hex..3E..enoate',
    'Ethyl 2-hexenoate': 'ethyl.hex..2E..enoate',
}
_GREEK = {'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
          'ö': 'o', 'é': 'e', 'ß': 'ss'}


def _norm_name(s):
    s = str(s)
    for a, b in _GREEK.items():
        s = s.replace(a, b)
    s = s.lower()
    s = s.replace('(e)', 'e').replace('(z)', 'z')
    return re.sub(r'[^a-z0-9]', '', s)


def build_map2csv(comp=None, meta=None):
    """返回 {2.csv 名称: 1.csv 列名}，保证双射；无法对齐的丢弃并报告。"""
    import difflib
    if comp is None:
        comp = load()['comp']
    if meta is None:
        meta = build_meta(comp)
    c2 = pd.read_csv(os.path.join(DATA, '2.csv'), index_col=0)
    names2 = list(c2.index)
    used, mp = set(), {}
    for nm in names2:                              # ① 人工对齐优先
        t = MAP2_OVERRIDE.get(nm)
        if t and t in comp and t not in used:
            mp[nm] = t; used.add(t)
    key_self = {}
    for c in comp:
        key_self.setdefault(_norm_name(c), c)
    key_chem = {}
    for c in comp:
        cn = meta.loc[c, 'chem_name']
        if isinstance(cn, str) and cn.strip():
            key_chem.setdefault(_norm_name(cn), c)
    for nm in names2:                              # ② 原名 / 化学名精确匹配
        if nm in mp:
            continue
        k = _norm_name(nm)
        t = key_self.get(k) or key_chem.get(k)
        if t and t not in used:
            mp[nm] = t; used.add(t)
    pool = {**key_self, **key_chem}
    for nm in names2:                              # ③ 模糊匹配
        if nm in mp:
            continue
        for cand in difflib.get_close_matches(_norm_name(nm), list(pool), n=6, cutoff=0.84):
            t = pool[cand]
            if t not in used:
                mp[nm] = t; used.add(t); break
    return mp, [c for c in comp if c not in used], [nm for nm in names2 if nm not in mp]


def corr_cc_aligned(comp=None, meta=None):
    """返回按 1.csv 列顺序排列的 231×231 皮尔逊相关矩阵（未匹配处为 NaN）。"""
    if comp is None:
        comp = load()['comp']
    if meta is None:
        meta = build_meta(comp)
    mp, miss_t, miss_s = build_map2csv(comp, meta)
    A = pd.read_csv(os.path.join(DATA, '2.csv'), index_col=0)
    A = A.loc[[k for k in A.index if k in mp], [k for k in A.columns if k in mp]]
    tgt = [mp[k] for k in A.index]
    B = np.full((len(comp), len(comp)), np.nan)
    pos = {c: i for i, c in enumerate(comp)}
    for i, ti in enumerate(tgt):
        for j, tj in enumerate(tgt):
            B[pos[ti], pos[tj]] = A.values[i, j]
    return B, mp, miss_t, miss_s


def communities(comp=None, meta=None, thr=0.35, seed=7):
    """共表达图上的社区发现（贪心模块度优化）。

    返回 (labels, G, modularity)；labels 为长度 231 的整数数组（0..C-1，
    按社区规模降序重编号），G 为 networkx 图。
    """
    import networkx as nx
    B, _mp, _mt, _ms = corr_cc_aligned(comp, meta)
    Ab = np.abs(np.nan_to_num(B, nan=0.0))
    np.fill_diagonal(Ab, 0.0)
    G = nx.Graph()
    G.add_nodes_from(range(len(comp)))
    for i in range(len(comp)):
        for j in range(i + 1, len(comp)):
            if Ab[i, j] >= thr:
                G.add_edge(i, j, weight=float(Ab[i, j]))
    parts = nx.community.greedy_modularity_communities(G, weight='weight')
    parts = sorted(parts, key=len, reverse=True)
    lab = np.full(len(comp), -1, dtype=int)
    for c, p in enumerate(parts):
        lab[list(p)] = c
    # 孤立点归入最后一个社区
    lab[lab < 0] = len(parts)
    Q = nx.community.modularity(G, [set(np.where(lab == c)[0]) for c in np.unique(lab)],
                                weight='weight')
    return lab, G, float(Q)


# ------------------------------------------------------------------ 指标
def r2(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ssr = np.nansum((y - p) ** 2)
    sst = np.nansum((y - np.nanmean(y)) ** 2)
    return 1.0 - ssr / sst if sst > 0 else np.nan


def metrics(y, p):
    from scipy.stats import spearmanr
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    m = np.isfinite(y) & np.isfinite(p)
    rho = spearmanr(y[m], p[m]).statistic if m.sum() > 3 else np.nan
    return dict(r2=r2(y, p), mae=float(np.mean(np.abs(y[m] - p[m]))),
                rmse=float(np.sqrt(np.mean((y[m] - p[m]) ** 2))), spearman=float(rho))


# ------------------------------------------------------------------ 绘图风格
def setup_style(palette_name='clean'):
    """沿用 AutoMCM-PlotSci 的出版级风格（字体阶梯 / 配色 / 导出参数）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import sys
        sys.path.insert(0, r'E:\dsh_work\AutoMCM-PlotSci')
        from plot_science import style as _style
        _style(lang='zh', palette_name=palette_name)
        return 'plot_science'
    except Exception as e:  # 降级
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
            'axes.unicode_minus': False,
            'figure.dpi': 120, 'savefig.dpi': 300,
            'axes.spines.top': False, 'axes.spines.right': False,
            'font.size': 9, 'axes.linewidth': 0.8,
        })
        print('[style] plot_science 不可用，已降级：%s' % e)
        return 'fallback'


if __name__ == '__main__':
    d = load()
    print('化学 %d 化合物, 感官 %d 维, 样本 %d' % (len(d['comp']), len(d['sdim']), len(d['X'])))
    M = build_meta(d['comp'])
    print('元数据：阈值可用 %d，异味标记 %d' % (M['threshold'].notna().sum(),
                                                (M['off_flavor'] != '').sum()))
    print(M['family'].value_counts().to_string())
    Y06, spec = calib06(d['Y'])
    print('0-6 标定后范围 %.2f ~ %.2f' % (Y06.values.min(), Y06.values.max()))
    json.dump({k: list(v) for k, v in spec.items()},
              open(os.path.join(OUT, 'calib_spec.json'), 'w'), ensure_ascii=False)
    print('ok')
