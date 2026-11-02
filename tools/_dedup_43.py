"""One-off: dedupe the 4.3+4.4 merge, keeping only info the merged section lacks."""
import pathlib

P = pathlib.Path(__file__).resolve().parents[1] / "build_v3" / "v3.md"
t = P.read_text(encoding="utf-8")

start = t.index("原始化学矩阵包含73个缺失单元。")
end = t.index("## 4.3 评分卡", start)
old = t[start:end]

# 4.2 首段已述 73 个缺失单元的填补规则、4VG/CS2/limonene 的低端堆积比例与"不推断真实检测限"。
# 旧 4.4 段只多出两点尚未出现的信息：Q3 生成用训练内经验分布表示编码地板、其余字段保留附件数值。
new = ("对生成任务（问题三），最低值以训练折内的经验分布表示编码地板，而非另设检测限模型；"
       "需要物理单位推断时只对经校准验证的字段讨论浓度，其余一律保留附件编码值。\n\n")

t = t[:start] + new + t[end:]
while "\n\n\n\n" in t:
    t = t.replace("\n\n\n\n", "\n\n\n")
P.write_text(t, encoding="utf-8")

print("removed", len(old), "-> kept", len(new))
i, j = t.index("## 4.2 化学尺度"), t.index("## 4.3 评分卡")
print(t[i:j])
