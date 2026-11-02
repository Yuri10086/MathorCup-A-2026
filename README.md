# MathorCup A 题：基于感官观测重建、互补化合物筛选与风险约束生成的啤酒风味建模

2026 年 MathorCup 数学应用挑战赛·大数据竞赛 **赛道 A** 参赛工程。围绕啤酒化学测量与感官评分在
尺度混合、重复观测和解释口径上的差异，完成三问建模：**感官评分预测 → 关键化合物筛选 → 风格条件生成与受约束迁移**。

> 本仓库为参赛作品的公开整理版：保留论文定稿、可复现代码、官方数据、结果报告与全部图件，
> 剔除了运行日志、模型二进制与中间副本（原工程 3.3 GB，此处约 27 MB）。

---

## 目录结构

```
├── code/                 定稿核心工程（论文 PDF 生成链 + 三问解题代码 + 数据 + 报告）
│   ├── src/rebuild/      三问代码，一题一文件（q1.py / q2.py / q3.py）
│   ├── scripts/          一键复现脚本 run_final.ps1 / run_q1.ps1 / build_rebuild_pdf.py
│   ├── configs/          实验协议与依赖锁定（rebuild_requirements.txt）
│   ├── data/             官方附件 1–8（raw/）与处理后的中间表（processed/）
│   ├── paper/            论文 Markdown 源、TeX 中间件与 PDF 定稿
│   ├── reports/          三问分析报告、封闭实验报告、AI 使用日志
│   └── submission/       提交清单与说明
├── results/              全量实验结果（审计表、OOF、Bootstrap、生成谱、校验 JSON）
├── full_project/         2026-10-28 比赛提交版完整三问工程（Latin 排版论文 + 分题结果）
├── dev_version/          2026 早期开发版工程（另一套实现路径与 latex 论文）
├── paper_final/          论文 v3（docx + Markdown 初稿 + 配套图）
├── figures/              最终成图 19 组（PNG + 矢量 PDF）
├── figures_project/      可独立运行的科研绘图工程（读自带 support/ 的 CSV 复现全部图）
├── docs/                 解题思路定稿 v12（已内嵌 v3→v12 全部修订史）+ 交接说明 + 评估
├── early_experiments/    方案迭代期的早期实验工程（q1_work 的 src/output/figs）
├── build_v3/             论文 v3 的技术路线图（PNG + PDF）与 v3.md
├── tools/                辅助脚本（去重、骨架修复、路线图生成）
├── legacy_sources/       重构前的旧版模块（closed.py / closed_tables.py / q3_diagnostics.py）
├── reference_papers/     数据来源原论文：Schreurs et al., Nat. Commun. 2024（原文 + 中译）
├── precedent_papers/     历年 MathorCup 大数据竞赛优秀论文（2024×4、2025×4）
├── paper_template/       数学建模 LaTeX 模板（Overleaf 可直接编译）
└── contest_rules/        赛题原文 PDF、问题说明、格式规范、承诺书与模板
```

### 三层工程的关系

本仓库保留了三个**不同阶段**的工程，不是重复而是演进链，各自仍有对照价值：

| 目录 | 阶段 | 说明 |
| --- | --- | --- |
| `dev_version/` | 2026 早期开发版 | 第一套实现路径，含独立 `latex/` 论文与 `src/`、`output/` 结果 |
| `full_project/` | 10-28 比赛提交版 | 当时提交的三问完整工程（`q1`/`q2`/`q2_improved`/`q3` 分目录 + `manuscript/`）|
| `code/` + `results/` | 最终定稿版 | 三问合并为 `q1.py`/`q2.py`/`q3.py`，配封闭补充实验与 65 项一致性校验 |

### 关于模型文件

`results/` 中的 `.joblib` 模型权重（单文件 11–24 MB、总计约 220 MB）**未纳入版本库**：
它们可由 `code/scripts/run_final.ps1` 与 `run_q1.ps1` 从 `data/` 完整重训得到，
保留二进制反而会掩盖「结果能否复现」这一关键问题。所有中间结果 CSV/JSON 则完整保留。

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

## 版权与引用

本项目**代码**采用 [MIT License](LICENSE)，可自由使用、修改与再分发。

但请注意以下分区：

| 内容 | 授权 |
| --- | --- |
| `code/`、`figures_project/`、`early_experiments/`、`tools/` 等源代码 | MIT |
| 论文、报告、图表等文字与图像作品（`paper_final/`、`figures/`、`docs/`、`reports/`、`manuscript/`） | 保留署名权，转载/引用请注明出处 |
| `contest_rules/` 赛题原始材料 | 版权归 MathorCup 组委会所有，仅作存档 |
| `precedent_papers/` 历年优秀论文 | 版权归原作者所有，仅作学习参考 |
| `reference_papers/` Schreurs et al. 2024 | 版权归 Nature Communications / 原作者所有（CC BY 4.0 开放获取），此处仅存档以标明数据来源 |
| `code/data/` 官方附件与补充数据 | 版权归原研究作者/MathorCup 组委会所有 |

> `reference_papers/` 收录的是本赛题数据来源的原始论文（Schreurs et al., *Nature Communications* 15, 2024）。
> 该文以 **CC BY 4.0** 开放获取发布，转载须保留原作者署名与出处；若需引用请直接引用原文 DOI，而非本仓库。

本仓库仅为参赛作品存档与技术交流之用；如需引用论文结论，请以 `paper_final/v3.docx` 与 `code/paper/competition_final.pdf` 为准。

