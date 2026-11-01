"""解析输入文件路径，避免引入训练流水线
run.py 模块加载时会调用 load()，导入它会强制要求 1.csv、4.csv、品酒员文件存在
先找data/raw后面再查找data/supplementary（作者公开的品酒员数据）"""

from pathlib import Path

R = Path(__file__).resolve().parents[2]
BASES = (R / "data/raw", R / "data/supplementary")


def find_input(name, required=True):
    # 返回 `name` 对应的第一个存在的路径，找不到则返回 None
    for base in BASES:
        p = base / name
        if p.is_file():
            return p
    if required:
        raise FileNotFoundError(
            f"cannot find {name} under data/raw or data/supplementary; see README_REBUILD.md "
            f"(official attachments go in data/raw, the per-taster file ships in data/supplementary)"
        )
    return None


def missing_inputs(names):
    #names`中不存在的名称列表，空列表表示全部可用
    return [n for n in names if find_input(n, required=False) is None]
