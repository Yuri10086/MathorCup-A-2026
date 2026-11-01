# -*- coding: utf-8 -*-
"""统一其余图件中柱状图的配色与形态（改为 lollipop / 细柱 + 克制的墨蓝主色）。"""
import re

INK = '#23405C'
ACC = '#A8452C'
GRY = '#8C939B'
LGT = '#E4E8EC'

# ---------------- fig10(b) 化学类别：横条 -> lollipop ----------------
p = r'E:\dsh_work\MC2026A\src\07_fig_q2.py'
t = open(p, encoding='utf-8').read()
old = """ax.barh(np.arange(len(fam_top)), fam_top.values, color=BLUE, height=0.62,
        edgecolor='white', lw=0.3)
ax.set_yticks(np.arange(len(fam_top)))
ax.set_yticklabels(fam_top.index, fontsize=6.4)
for i, v in enumerate(fam_top.values):
    ax.text(v + 0.3, i, str(v), va='center', fontsize=6)"""
new = """for i, v in enumerate(fam_top.values):
    ax.plot([0, v], [i, i], color='#E4E8EC', lw=2.0, solid_capstyle='round', zorder=1)
    ax.scatter(v, i, s=40, color='#23405C', edgecolor='white', lw=0.5, zorder=3)
    ax.text(v + 0.5, i, str(v), va='center', fontsize=6.0, color='#333333')
ax.set_yticks(np.arange(len(fam_top)))
ax.set_yticklabels(fam_top.index, fontsize=6.4)
ax.grid(axis='x', ls=':', lw=0.4, color='#E4E8EC')"""
assert old in t, 'fig10 patch miss'
t = t.replace(old, new)
t = t.replace("ax.set_xlim(0, fam_top.max() * 1.18)", "ax.set_xlim(0, fam_top.max() * 1.22)")
open(p, 'w', encoding='utf-8').write(t)
print('fig10(b) -> lollipop')

# ---------------- fig04(b) 分组 R2：彩色粗柱 -> 同色细柱 ----------------
p = r'E:\dsh_work\MC2026A\src\06_fig_q1.py'
t = open(p, encoding='utf-8').read()
old4 = """gcol = [GCOL.get(g, GRAY) for g in grp.index]
ax.barh(np.arange(len(grp)), grp['mean'].values, color=gcol, height=0.62,
        edgecolor='white', lw=0.4)
ax.set_yticks(np.arange(len(grp)))
ax.set_yticklabels(['%s(%d)' % (g, n) for g, n in zip(grp.index, grp['count'])], fontsize=6.6)
for i, v in enumerate(grp['mean'].values):
    ax.text(v + 0.008, i, '%.3f' % v, va='center', fontsize=6.2)"""
new4 = """for i, (g, v) in enumerate(zip(grp.index, grp['mean'].values)):
    ax.plot([0, v], [i, i], color='#E4E8EC', lw=2.2, solid_capstyle='round', zorder=1)
    ax.scatter(v, i, s=42, color=GCOL.get(g, GRAY), edgecolor='white', lw=0.5, zorder=3)
    ax.text(v + 0.010, i, '%.3f' % v, va='center', fontsize=6.2, color='#333333')
ax.set_yticks(np.arange(len(grp)))
ax.set_yticklabels(['%s(%d)' % (g, n) for g, n in zip(grp.index, grp['count'])],
                   fontsize=6.6)
ax.grid(axis='x', ls=':', lw=0.4, color='#E4E8EC')"""
assert old4 in t, 'fig04 patch miss'
t = t.replace(old4, new4)
t = t.replace("ax.set_xlim(0, max(grp['mean']) * 1.30)", "ax.set_xlim(0, max(grp['mean']) * 1.32)")

# ---------------- fig07(a) 消融：柱 -> lollipop + 基准竖线 ----------------
old7a = """ax.bar(np.arange(len(an)), av, color=BLUE, width=0.62, edgecolor='white', lw=0.4)
ax.set_xticks(np.arange(len(an)))
ax.set_xticklabels(lab, fontsize=6.2, rotation=18, ha='right')
for i, v in enumerate(av):
    ax.text(i, v + 0.006, '%.3f' % v, ha='center', fontsize=6.2)
d0 = av[0]
ax.axhline(d0, color=GRAY, lw=0.8, ls=':')
ax.set_ylabel('平均 $R^2$')
ax.set_ylim(min(av) * 0.94, max(av) * 1.06)"""
new7a = """yy7 = np.arange(len(an))
for i, v in enumerate(av):
    ax.plot([min(av) * 0.985, v], [i, i], color='#E4E8EC', lw=2.0, solid_capstyle='round',
            zorder=1)
    ax.scatter(v, i, s=38, color=INK, edgecolor='white', lw=0.5, zorder=3)
    ax.text(v + (max(av) - min(av)) * 0.04 + 0.001, i, '%.4f' % v, va='center',
            fontsize=6.0, color='#333333')
d0 = av[0]
ax.axvline(d0, color=ACC, lw=0.9, ls='--')
ax.text(d0, len(an) - 0.35, '基准 A0', fontsize=5.8, color=ACC, ha='center', va='bottom')
ax.set_yticks(yy7)
ax.set_yticklabels(lab, fontsize=6.2)
ax.set_xlabel('平均 $R^2$')
ax.set_ylim(-0.6, len(an) - 0.3)
ax.set_xlim(min(av) * 0.985, max(av) + (max(av) - min(av)) * 0.55 + 0.004)
ax.grid(axis='x', ls=':', lw=0.4, color='#E4E8EC')"""
assert old7a in t, 'fig07a patch miss'
t = t.replace(old7a, new7a)

# ---------------- fig07(c) top-k：柱 -> lollipop ----------------
old7c = """ax.bar(np.arange(len(kk)), [st[x] for x in kk], color=BLUE, width=0.55,
       edgecolor='white', lw=0.4)
ax.set_xticks(np.arange(len(kk)))
ax.set_xticklabels(['k=%s' % x for x in kk], fontsize=6.6)
for i, x in enumerate(kk):
    ax.text(i, st[x] + 0.004, '%.3f' % st[x], ha='center', fontsize=6.2)
ax.set_ylim(min(st.values()) * 0.96, max(st.values()) * 1.05)
ax.set_ylabel('平均 $R^2$')"""
new7c = """v7 = [st[x] for x in kk]
for i, v in enumerate(v7):
    ax.plot([i, i], [min(v7) * 0.99, v], color='#E4E8EC', lw=2.0, solid_capstyle='round',
            zorder=1)
    ax.scatter(i, v, s=38, color=INK, edgecolor='white', lw=0.5, zorder=3)
    ax.text(i, v + (max(v7) - min(v7)) * 0.10 + 0.0005, '%.4f' % v, ha='center',
            fontsize=6.0, color='#333333')
ax.set_xticks(np.arange(len(kk)))
ax.set_xticklabels(['k=%s' % x for x in kk], fontsize=6.6)
ax.set_ylim(min(v7) * 0.985, max(v7) + (max(v7) - min(v7)) * 0.55)
ax.set_ylabel('平均 $R^2$')
ax.set_xlim(-0.6, len(kk) - 0.4)"""
assert old7c in t, 'fig07c patch miss'
t = t.replace(old7c, new7c)

# 统一主色常量
t = t.replace("BLUE, RED, GREEN, PURPLE, GRAY = '#1F6FD0', '#C0392B', '#00876C', '#6C4AA0', '#7A8794'",
              "BLUE, RED, GREEN, PURPLE, GRAY = '#23405C', '#A8452C', '#4A6E5A', '#5C7F9E', '#8C939B'\nINK, ACC = '#23405C', '#A8452C'")
open(p, 'w', encoding='utf-8').write(t)
print('fig04(b)/fig07(a)(c) -> lollipop')
