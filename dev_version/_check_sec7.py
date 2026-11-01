import io
import re

t = io.open(r'src\10_gen_results_tex.py', encoding='utf-8').read()
m = re.search(r"sec7 = r'''(.*?)'''", t, re.S)
tpl = m.group(1)
i = 0
bad = []
specs = 0
while i < len(tpl):
    if tpl[i] == '%':
        if i + 1 < len(tpl) and tpl[i + 1] == '%':
            i += 2
            continue
        mm = re.match(r"%[-+ 0#]*[\d.]*[sdifgeE]", tpl[i:])
        if mm:
            specs += 1
            i += len(mm.group(0))
            continue
        bad.append((i, tpl[max(0, i - 40):i + 12].replace('\n', ' ')))
    i += 1
print('合法占位符 %d 个；非法 %% %d 处' % (specs, len(bad)))
for pos, ctx in bad:
    print('  @%d ...%s...' % (pos, ctx))
