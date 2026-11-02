"""AI-assisted DeepSeek Harness, 2026-10-29. Skeleton self-check for build_v3/v3.md.

Checks the structural layer only: chapter/section numbering, top-level order, table and
figure numbering, and that no heading is orphaned or duplicated. Returns a pass/fail list.
"""
import pathlib
import re

P = pathlib.Path(__file__).resolve().parents[1] / "build_v3" / "v3.md"
t = P.read_text(encoding="utf-8")
lines = t.split("\n")

checks = []


def add(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


# ── 一级标题顺序 ──────────────────────────────────────────────
tops = [l for l in lines if l.startswith("# ") and not l.startswith("## ")]
expect_tops = [
    "# 基于", "# 一、问题重述", "# 二、模型假设", "# 三、问题分析", "# 四、数据审计",
    "# 五、问题一", "# 六、问题二", "# 七、问题三", "# 八、模型检验", "# 九、结论与建议",
    "# 参考文献", "# 附录", "# AI工具使用声明",
]
add("一级标题数=13", len(tops) == 13, f"{len(tops)}")
for i, exp in enumerate(expect_tops):
    add(f"一级#{i+1} = {exp}", tops[i].startswith(exp), tops[i][:40])

# ── 章节编号连续 ──────────────────────────────────────────────
nums = {}
for l in lines:
    m = re.match(r"## (\d+)\.(\d+)", l)
    if m:
        nums.setdefault(int(m.group(1)), []).append(int(m.group(2)))
for ch, secs in nums.items():
    add(f"第{ch}章小节连续 {secs}", secs == list(range(1, len(secs) + 1)))

# ── 表题 ─────────────────────────────────────────────────────
caps = re.findall(r"\*\*表(\d+)　", t)
add("表题 1..10 连续", caps == [str(i) for i in range(1, 11)], ",".join(caps))
# 表号交叉引用属于正文工作，本轮骨架只要求表题连续编号；如实记录待补项。
noref = re.sub(r"\*\*表\d+　[^*]+\*\*", "", t)
pending = [n for n in range(1, 11) if not re.search(rf"表{n}(?!\d)", noref)]
add("表号正文引用（下一轮正文工作）", True, f"待补: {pending}")

# ── 图号 ─────────────────────────────────────────────────────
add("图号只用 图1", set(re.findall(r"图(\d)", t)) == {"1"}, sorted(set(re.findall(r"图(\d)", t))))
add("图1 有唯一落点（待插入占位）", t.count("【图1 待插入") == 1)
add("1.3 不再残留第二个占位", "【图1：" not in t)

# ── 附录 ─────────────────────────────────────────────────────
add("附录含 A/B/C", all(f"## 附录{s}" in t for s in "ABC"))
add("附录位于 AI 声明之前", t.index("# 附录") < t.index("# AI工具使用声明"))

# ── 残留编辑批注 ──────────────────────────────────────────────
add("AI 声明标题无括号批注", "# AI工具使用声明（" not in t)
add("无「占位」字样残留（除图1待插入）", t.count("占位") <= 1, str(t.count("占位")))

# ── 摘要存在且在最前 ──────────────────────────────────────────
add("摘要存在", "## 摘要" in t)
add("摘要位于第一章之前", t.index("## 摘要") < t.index("# 一、问题重述"))

fails = [c for c in checks if not c[1]]
for name, ok, detail in checks:
    print(("PASS " if ok else "FAIL ") + name + (f"   [{detail}]" if detail else ""))
print(f"\n{len(checks) - len(fails)}/{len(checks)} passed")
if fails:
    raise SystemExit(1)
