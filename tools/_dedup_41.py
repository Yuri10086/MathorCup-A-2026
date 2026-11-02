"""One-off: remove the paragraph duplicated by merging 4.1+4.2 (skeleton pass, no new prose)."""
import pathlib

P = pathlib.Path(__file__).resolve().parents[1] / "build_v3" / "v3.md"
t = P.read_text(encoding="utf-8")

start = t.index("联接键的选择直接决定监督是否有效。")
end = t.index("对化学名称与相关矩阵", start)
dup = t[start:end]

# 该段与 4.1 首段同义（beer_name 行序 / 0.9777 / 0.0076 / 静默合并），且 `beer`、`beer_id`
# 等反引号内容与前段重复，故整段删除；保留其后唯一新增的“名称与相关矩阵核对”一句。
t = t[:start] + t[end:]

while "\n\n\n\n" in t:
    t = t.replace("\n\n\n\n", "\n\n\n")

P.write_text(t, encoding="utf-8")
print("removed", len(dup), "chars")

i, j = t.index("## 4.1"), t.index("## 4.2 化学尺度")
print(t[i:j])
