# -*- coding: utf-8 -*-
"""01 数据审计与预处理（MC2026 赛道A · 啤酒酿造化合物与风味预测）

本程序及代码是在人工智能工具辅助下完成的。
AI 工具：DeepSeek-V4.1-Flash，开发机构：深度求索（DeepSeek），版本发布：2026 年。

职责：
  1) 结构核对（对齐、缺失、量纲）
  2) 对数变换底数核实（用 8.xlsx 的"Calibrated chemical dataset"反解）
  3) 检出限左删失识别
  4) 感官 0-6 分制标定
  5) 阈值/来源/异味标记元数据表 + OAV
输出：output/audit_report.txt, output/prepared.npz, output/meta.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'output')
os.makedirs(OUT, exist_ok=True)

REP = open(os.path.join(OUT, 'audit_report.txt'), 'w', encoding='utf-8')


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, file=REP)


def hdr(t):
    P('\n' + '=' * 92)
    P(t)
    P('=' * 92)


# ------------------------------------------------------------------ 读入
hdr('一、数据读入与结构')
chem = pd.read_csv(os.path.join(DATA, '1.csv'))
sens = pd.read_csv(os.path.join(DATA, '4.csv'))
corr_cc = pd.read_csv(os.path.join(DATA, '2.csv'), index_col=0)
corr_ss = pd.read_csv(os.path.join(DATA, '5.csv'), index_col=0)
corr_cs = pd.read_csv(os.path.join(DATA, '6.csv'), index_col=0)
rev = pd.read_csv(os.path.join(DATA, '7.csv'))
x8 = {s: pd.read_excel(os.path.join(DATA, '8.xlsx'), sheet_name=s) for s in
      pd.ExcelFile(os.path.join(DATA, '8.xlsx')).sheet_names}

P('1.csv  化学 %s' % str(chem.shape))
P('4.csv  感官 %s' % str(sens.shape))
P('2.csv  化合物相关 %s' % str(corr_cc.shape))
P('5.csv  感官相关   %s' % str(corr_ss.shape))
P('6.csv  化学×感官 %s' % str(corr_cs.shape))
P('7.csv  评论描述符 %s' % str(rev.shape))

COMP = [c for c in chem.columns if c not in ('beer', 'beer_id', 'tasting_category_fine')]
SENS = [c for c in sens.columns if c not in ('beer_id', 'tasting_category_fine')]
P('\n化合物列数 %d，感官列数 %d' % (len(COMP), len(SENS)))

# 对齐：以 beer 编号为唯一键（beer_id 在两表中不一致，见下）
if sens.index.name == 'beer' and 'beer' not in sens.columns:
    sens = sens.reset_index()
chem = chem.sort_values('beer').reset_index(drop=True)
sens = sens.sort_values('beer').reset_index(drop=True)
assert len(chem) == len(sens) == 250, '行数异常'
assert np.array_equal(chem['beer'].values, sens['beer'].values), 'beer 编号未对齐'
P('对齐后 250 行，beer 编号完全一致 ✓')
same_bid = np.array_equal(chem['beer_id'].values, sens['beer_id'].values)
P('\n⚠ beer_id 一致性：%s（两表同一 beer 编号的 beer_id 有 %d 行不同）'
  % ('一致' if same_bid else '不一致', int((chem['beer_id'].values != sens['beer_id'].values).sum())))
P('   处理：全流程以 beer 编号（1-250）为唯一主键，不使用 beer_id。')
P('   示例（beer, 1.csv beer_id, 4.csv beer_id）：')
diff = np.where(chem['beer_id'].values != sens['beer_id'].values)[0][:5]
for i in diff:
    P('     %3d  %s  %s' % (chem['beer'].iloc[i], chem['beer_id'].iloc[i], sens['beer_id'].iloc[i]))

# ------------------------------------------------------------------ 缺失
hdr('二、缺失与左删失')
X = chem[COMP].astype(float)
Y = sens[SENS].astype(float)
miss_cols = X.columns[X.isna().any()].tolist()
P('缺失单元格 %d / %d (%.3f%%)，涉及列 %d 个' % (
    int(X.isna().sum().sum()), X.size, 100 * X.isna().sum().sum() / X.size, len(miss_cols)))
for c in miss_cols:
    P('   %-42s 缺失 %3d (%.1f%%)  均值 %.3f' % (c, X[c].isna().sum(),
                                                 100 * X[c].isna().mean(), X[c].mean()))
P('感官矩阵缺失 %d' % int(Y.isna().sum().sum()))

# 左删失：某列 >=5% 的样本取到该列的最小值，视为检出限封底
cens = {}
for c in COMP:
    v = X[c].dropna()
    if len(v) == 0:
        continue
    lo = v.min()
    frac = float((np.abs(v - lo) < 1e-9).mean())
    if frac >= 0.05:
        cens[c] = frac
P('\n检出限封底（>=5%% 样本取到最小值）的化合物 %d 个：' % len(cens))
for c, f in sorted(cens.items(), key=lambda kv: -kv[1]):
    P('   %-42s %.1f%% 封底值 %.4f' % (c, 100 * f, X[c].min()))
P('封底单元格合计 %d (%.2f%% 的观测)' % (
    sum(int((np.abs(X[c] - X[c].min()) < 1e-9).sum()) for c in cens),
    100 * sum(int((np.abs(X[c] - X[c].min()) < 1e-9).sum()) for c in cens) / X.size))

# ------------------------------------------------------------------ 对数底数
hdr('三、对数变换底数核实（关键）')
cal = x8['Calibrated chemical dataset'].copy()
cal = cal.sort_values('beer').reset_index(drop=True)
assert np.array_equal(cal['beer'].values, chem['beer'].values)
CALC = [c for c in cal.columns if c in COMP]
P('校准浓度表提供 %d 个化合物的真实浓度（与 1.csv 同名可直连）' % len(CALC))

rec = []
for c in CALC:
    v = X[c].values.astype(float)
    conc = cal[c].values.astype(float)
    m = np.isfinite(v) & np.isfinite(conc) & (conc > 0)
    if m.sum() < 20:
        continue
    lg = np.log10(conc[m])
    # log10(c) = a*v + b  → 若底数 10，则 a ≈ 1；若底数 e，则 a ≈ 1/ln10 = 0.4343
    A = np.column_stack([v[m], np.ones(m.sum())])
    coef, *_ = np.linalg.lstsq(A, lg, rcond=None)
    pred = A @ coef
    ss = 1 - ((lg - pred) ** 2).sum() / ((lg - lg.mean()) ** 2).sum()
    rec.append(dict(compound=c, n=int(m.sum()), slope=float(coef[0]), intercept=float(coef[1]),
                    r2=float(ss),
                    ratio_obs=float(np.exp(np.log(10) * np.median(lg - coef[0] * v[m])))))
R = pd.DataFrame(rec)
P('回归式：log10(校准浓度) = slope × (1.csv 数值) + intercept')
P(R.sort_values('r2', ascending=False).head(20).to_string(
    index=False, float_format=lambda x: '%8.4f' % x))
P('\n稳健汇总（R2>0.95 的化合物，n=%d）：' % int((R.r2 > 0.95).sum()))
good = R[R.r2 > 0.95]
P('   slope 中位数 = %.4f   中位绝对偏差 = %.4f' % (good.slope.median(), (good.slope - 1).abs().median()))
P('   理论值：底数 10 → slope=1.0000；底数 e → slope=0.4343')
P('   以 log10 解释时的平均相对误差 = %.2f%%' % (100 * float(np.median(np.abs(good.slope - 1)))))
P('   以 ln  解释时的平均相对误差 = %.2f%%' % (100 * float(np.median(np.abs(good.slope - 1 / np.log(10))))))
LOG_BASE = 10 if abs(good.slope.median() - 1) < abs(good.slope.median() - 1 / np.log(10)) else 'e'
P('   >>> 判定：底数 = %s' % LOG_BASE)
P('   1.csv 数值 v → 原始浓度 C = 10^v （单位见阈值表的 unit 列）')

# 逐列判定：有 33 列虽无说明，实为原始浓度
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as _C
scale, unk = _C.column_scale(COMP)
raw = sorted([c for c, v in scale.items() if v == 'raw'])
logc = sorted([c for c, v in scale.items() if v == 'log10' and c not in unk])
P('\n⚠ 逐列判定（用校准表反证）：对数变换 %d 列，**原始浓度 %d 列**（附件说明未提及），'
  '无校准信息默认对数 %d 列' % (len(logc), len(raw), len(unk)))
P('   原始浓度列（多为 µg·L⁻¹ 级超痕量致香物，直接比对 con=v 精确成立）：')
for c in raw:
    P('     %-36s max=%12.3f' % (c, X[c].max()))
P('   → 后果：OAV 计算必须逐列区分，否则超痕量致香物的 OAV 会被高估/低估数个量级。')


# ------------------------------------------------------------------ 0-6 标定
hdr('四、感官 0-6 分制标定')
desc = Y.describe().T
P('标准化后各维均值 %.4f ~ %.4f，标准差 %.3f ~ %.3f' % (
    desc['mean'].min(), desc['mean'].max(), desc['std'].min(), desc['std'].max()))
P('→ 数据已被中心化，但标准差并非 1（说明并非标准 z 分数），且极差超界')
P('  取值范围 %.3f ~ %.3f；若为 [0,6] 的平移，则平移量无法同时满足两端 → 判定为"中心化后再缩放"')
P('  标定方案：逐维 min-max 线性映射回官方 0-6 分制  score = 6*(z-zmin)/(zmax-zmin)')
Y06 = Y.copy()
for c in SENS:
    lo, hi = Y[c].min(), Y[c].max()
    Y06[c] = 6.0 * (Y[c] - lo) / (hi - lo)
P('\n标定后 range %.3f ~ %.3f，各维均值 %.3f ~ %.3f' % (
    Y06.values.min(), Y06.values.max(), Y06.mean().min(), Y06.mean().max()))
P(Y06.describe().T[['mean', 'std', 'min', '50%', 'max']].head(12).to_string(
    float_format=lambda x: '%6.3f' % x))

# ------------------------------------------------------------------ 元数据
hdr('五、阈值 / 来源 / 异味标记（8.xlsx Sensory thresholds）')
th = x8['Sensory thresholds'].copy()
P('阈值表 %d 行，列：%s' % (th.shape[0], list(th.columns)))
th['field_name_python'] = th['field_name_python'].astype(str).str.strip()
m = th.set_index('field_name_python')
hit = [c for c in COMP if c in m.index]
P('与 1.csv 化合物名可匹配的条目：%d / %d' % (len(hit), len(COMP)))
sub = m.loc[hit]
P('\n来源（origin_paper）分布：')
P(sub['origin_paper'].astype(str).value_counts().to_string())
P('\n气味阈值可用条目：%d' % int(sub['threshold_mean'].notna().sum()))
P('异味标记（off_flavors）非空条目：%d' % int(sub['off_flavors'].notna().sum()))
P(sub[sub['off_flavors'].notna()][['chemical_name', 'off_flavors', 'threshold_mean',
                                   'unit', 'origin_paper']].to_string())

P('\n阈值表全量中 off_flavors 非空（含未匹配化合物）：')
P(th[th['off_flavors'].notna()][['field_name_python', 'chemical_name', 'off_flavors',
                                 'threshold_mean', 'threshold_median', 'unit']].to_string())

# ------------------------------------------------------------------ OAV
hdr('六、OAV = 浓度 / 阈值')
oav = pd.DataFrame(index=chem['beer'].values)
for c in CALC:
    if c not in m.index:
        continue
    thr = m.loc[c, 'threshold_mean']
    if not np.isfinite(thr) or thr <= 0:
        continue
    conc = cal[c].values.astype(float)
    oav[c] = conc / float(thr)
P('可算 OAV 的化合物 %d 个' % oav.shape[1])
if oav.shape[1]:
    st = pd.DataFrame({'阈值': [m.loc[c, 'threshold_mean'] for c in oav.columns],
                       '单位': [m.loc[c, 'unit'] for c in oav.columns],
                       'OAV中位数': oav.median().values,
                       'OAV>1占比': (oav > 1).mean().values,
                       '来源': [m.loc[c, 'origin_paper'] for c in oav.columns]},
                      index=oav.columns).sort_values('OAV中位数', ascending=False)
    P(st.to_string(float_format=lambda x: '%9.3f' % x))

# ------------------------------------------------------------------ 异常值
hdr('七、异常值筛查（稳健 z 分数 |z|>3.5）')
Xr = X.copy()
z = (Xr - Xr.median()) / (Xr.quantile(0.75) - Xr.quantile(0.25)).replace(0, np.nan) * 1.349
out_cnt = (z.abs() > 3.5).sum()
P('异常单元格 %d (%.2f%%)，涉及列 %d' % (
    int(out_cnt.sum()), 100 * out_cnt.sum() / X.size, int((out_cnt > 0).sum())))
P(out_cnt[out_cnt > 0].sort_values(ascending=False).head(15).to_string())
P('\n结论：异常值以"高浓度长尾"为主，属真实商业啤酒差异（如酸啤/烟熏），'
  '\n      不剔除，改用稳健缩放 + 分位截断（在建模脚本内以 winsorize 处理）。')

# ------------------------------------------------------------------ 风格
hdr('八、风格标签')
vc = chem['tasting_category_fine'].value_counts()
P(vc.to_string())
P('\n小众风格（n<6）合并策略见问题 3 脚本；本体保留原始 22 类。')

# ------------------------------------------------------------------ 缓存
hdr('九、缓存')
np.savez_compressed(
    os.path.join(OUT, 'prepared.npz'),
    X=X.values, Y=Y.values, Y06=Y06.values,
    beer=chem['beer'].values, beer_id=chem['beer_id'].values,
    comp=np.array(COMP), sens=np.array(SENS),
    style=chem['tasting_category_fine'].values,
    oav=oav.values if oav.shape[1] else np.zeros((250, 0)),
    oav_cols=np.array(list(oav.columns)),
)
meta = dict(n_chem=len(COMP), n_sens=len(SENS), log_base=str(LOG_BASE),
            calib='minmax_to_0_6', n_missing=int(X.isna().sum().sum()),
            n_censored=len(cens), censored=cens)
with open(os.path.join(OUT, 'meta.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
P('prepared.npz / meta.json 已写出')
REP.close()
print('audit done')
