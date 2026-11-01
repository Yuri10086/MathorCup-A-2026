# MathorCup A 题：基于感官观测重建、互补化合物筛选与风险约束生成的啤酒风味建模

2026 年 MathorCup 数学应用挑战赛·大数据竞赛 **赛道 A** 参赛工程。围绕啤酒化学测量与感官评分在
尺度混合、重复观测和解释口径上的差异，完成三问建模：**感官评分预测 → 关键化合物筛选 → 风格条件生成与受约束迁移**。

> 本仓库为参赛作品的公开整理版：保留论文定稿、可复现代码、官方数据、结果报告与全部图件，
> 剔除了运行日志、模型二进制与中间副本（原工程 3.3 GB，此处约 27 MB）。

---

## 目录结构

```
├── code/                 核心工程（论文 PDF 生成链 + 三问解题代码 + 数据 + 报告）
│   ├── src/rebuild/      三问代码，一题一文件（q1.py / q2.py / q3.py）
│   ├── scripts/          一键复现脚本 run_final.ps1 / run_q1.ps1 / build_rebuild_pdf.py
│   ├── configs/          实验协议与依赖锁定（rebuild_requirements.txt）
│   ├── data/             官方附件 1–8（raw/）与处理后的中间表（processed/）
│   ├── paper/            论文 Markdown 源、TeX 中间件与 PDF 定稿
│   ├── reports/          三问分析报告、封闭实验报告、AI 使用日志
│   └── submission/       提交清单与说明
├── paper_final/          论文 v3（docx + Markdown 初稿 + 配套图）
├── figures/              最终成图 19 组（PNG + 矢量 PDF）
├── figures_project/      可独立运行的科研绘图工程（读自带 support/ 的 CSV 复现全部图）
├── docs/                 方案迭代记录 v2→v12 及评审修改清单
├── paper_template/       数学建模 LaTeX 模板（Overleaf 可直接编译）
└── contest_rules/        赛题 PDF、论文格式规范、承诺书与模板
```

---

## 三问方案速览

| 问题 | 任务 | 核心做法 | 关键结果 |
| --- | --- | --- | --- |
| **问题一** | 由化学测量预测 50 维感官评分 | 恢复"评分员等权 0–6 分面板目标"，比较传统回归、树集成与层次化神经网络；输出 250×53 评分卡 | ExtraTrees 取得 **MAE 0.4454 / RMSE 0.6305**，优于层次网络；非香气强度项 MAE 0.3486 |
| **问题二** | 高度共线化学特征下的关键化合物筛选 | 统一比较 Spearman 前 3、**2+1 条件增益**、完整前向、冗余控制四种规则，配 100 次 Bootstrap 稳定性 | 固定 2+1 规则内层 MAE 0.4988；含方法选择的外层 MAE 0.5024，量化了小样本选法的方差代价 |
| **问题三** | 风格条件生成与受约束迁移 | PCA 高斯、独立 Copula、联合 Copula、CVAE 四种生成器；22 风格分布 / 分类一致性 / 相关结构 / 联合低端模式四类诊断；9 款源酒四策略迁移 | CVAE 条件 SWD 0.7059 但多样性低且优势与测试集规模强相关（r=0.70）；硬约束把越阈次数由 7 降到 0（严格口径下仍有 4–5/9 款上升，已列为未解决局限） |

**核心方法论主张**：恢复真实观测目标、区分"固定筛选"与"完整选择流程"、隔离生成设计器与评价器，
三者共同构成可复核的证据链。所有结果均为同一批 250 款酒上的内部重采样证据，不等同外部盲评或真实酿造验证。

---

## 复现

环境见 `code/configs/rebuild_requirements.txt`（Python 3.13，numpy 2.3.5 / pandas 2.3.3 / scikit-learn 1.7.2 /
torch 2.13 / matplotlib 3.10.6 / networkx 3.5.0 / pypdf 6.21.3）。

```powershell
# Q2/Q3 封闭实验 + 论文 PDF 重建
powershell -ExecutionPolicy Bypass -File code/scripts/run_final.ps1

# 仅用已有结果刷新图表与论文
powershell -ExecutionPolicy Bypass -File code/scripts/run_final.ps1 -ReportOnly

# 第一题独立复现（训练 Q1 全部模型 → 乙醛判定 → 组装 250×53 评分卡）
powershell -ExecutionPolicy Bypass -File code/scripts/run_q1.ps1
```

脚本不写死任何机器路径，`python` / `pandoc` / `xelatex` 依次按参数、PATH、常见安装位置解析，
失败时直接报出缺失的工具名。详见 [`code/README_REBUILD.md`](code/README_REBUILD.md) 与
[`code/队员独立复核指引.md`](code/队员独立复核指引.md)。

**只重画图**（无需训练、无需完整工程）：

```powershell
cd figures_project
pip install -r requirements.txt
python plot_all.py            # 读自带 support/ 的真实结果 CSV，输出至 publication_figures/
```

---

## 数据来源

官方附件位于 `code/data/raw/`：`1.csv`（250 款酒化学测量）、`2.csv`（化学参数 Spearman）、
`3.xlsx`（品评表）、`4.csv`（感官分析结果）、`5.csv`（感官参数 Spearman）、`6.csv`（化学-感官 Spearman）、
`7.csv`（论坛评论）、`8.xlsx`（其他数据）。另含原研究公开的 2197 条逐评分员记录
（`Raw sensory data anonymized.csv`），用于把重复品评先在评分员内聚合、再按评分员等权构造参考目标。

---

## 诚实披露

- 评分卡存在一项已量化并披露的局限：预测等级全部落在 0–4 级，真值 ≥5 的 121 个单元被系统低估（平均低估 2.07）。
- 问题三严格口径下的迁移残余风险（4–5/9 款酒在任意微量上升口径下仍上升）列为未解决局限，未做美化。
- 附录含完整 AI 工具使用说明（`code/reports/AI_usage_log.md`、`code/submission/AI工具使用详情说明.pdf`）。

---

## 版权

本仓库内容为参赛队原创作品，仅用于学习、复现与技术交流；赛题原始材料版权归 MathorCup 组委会及原作者所有。
