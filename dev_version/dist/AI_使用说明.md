# 人工智能工具使用说明

依据：《2026 年 MathorCup 数学应用挑战赛—大数据竞赛人工智能工具使用规定》
以及竞赛《论文格式及提交规范》对 AI 辅助的留痕要求。

## 一、使用的工具

| 项目 | 内容 |
|---|---|
| 工具名称 | DeepSeek-V4.1-Flash |
| 开发机构 | 深度求索（DeepSeek） |
| 版本发布 | 2026 年 |
| 使用环节 | 数据审计脚本编写、建模代码实现、论文语言润色与 LaTeX 排版 |
| 未使用的环节 | 赛题理解方向、模型结构选型、结论判断、创新点归属（均由作者确定） |

## 二、留痕落实情况

1. **辅助编程**：`src/` 下每个 `.py` 文件开头均有注释块，说明本程序在 AI 工具辅助下完成，
   并标明工具名称、开发机构、版本年份。
2. **辅助数据分析**：`output/` 下的结果文件由脚本自动生成，脚本头部同样带上述注释块；
   论文中所有结果数字均由 `src/10_gen_results_tex.py` 从结果文件直接生成，
   不存在手工转录。
3. **AI 生成的模型/公式**：论文中全部模型公式均给出可复核的推导过程或文献来源
   （见参考文献 [1]–[19]）。不存在无推导、无引用标识、且无法确认来源的模型或公式。

## 三、可复现性声明

全部结果、图件与交付表可由下列命令一次复现（需 Python 3.11 + xelatex）：

```
python src/01_audit.py        # 数据审计
python src/02_q1.py           # 问题一
python src/02b_ablation.py    # 消融与敏感性
python src/03_q2.py           # 问题二
python src/04_q3.py           # 问题三
python src/05_fig_data.py     # 图 1–3
python src/05b_fig_schematic.py  # 图 0、15
python src/06_fig_q1.py       # 图 4–7
python src/07_fig_q2.py       # 图 8–10
python src/08_fig_q3.py       # 图 11–14
python src/10_gen_results_tex.py  # 生成结果章节
python src/11_gen_abstract.py     # 生成摘要与结论
python src/09_gate.py         # 编译 + 格式门控
python src/12_package.py      # 打包
```

随机种子固定为 20261026，所有交叉验证划分由 `StratifiedKFold(shuffle=True, random_state=...)`
确定，结果可逐位复现。

> 说明：上述标注内容存在不被评阅认可的风险，特此声明。
