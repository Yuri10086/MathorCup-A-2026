# MathorCup A 三问定稿工程

当前定稿：paper/competition_final.pdf（73 页：摘要 1 页、目录 1 页、正文 19 页、附录自第 22 页起，正文未超 30 页；以 outputs/rebuild/closed/pdf_quality.json 为唯一版式记录，本文件与其他文档一律不另记页数）。三组封闭补充实验已完成，此后停止模型探索；Q1 程序、结果、报告及原协议由 configs/closed_final_protocol.json 中的 SHA-256 冻结。

## 复现

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_final.ps1
```

脚本不再写死任何机器路径：`python`、`pandoc`、`xelatex` 依次按 `-Python/-Pandoc/-XeLaTeX` 参数、当前 PATH、常见安装位置解析，并在开始时打印实际使用的三个路径；也可用环境变量 `MC_PANDOC`、`MC_XELATEX` 指定。解析失败时脚本直接报出缺失的工具名，而不是给出与本机相关的错误。

默认仅运行最后三组：Q2 四规则训练内选择与 Bootstrap；Q3 原四模型的 22 风格诊断；Q3 固定评价器下 9 源酒四策略迁移。不会训练 Q1。模型与超参数均沿用原协议，不增加模型家族。所有结果进入 outputs/rebuild/closed。

仅用已完成结果更新图表与论文：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_final.ps1 -ReportOnly
```

原入口 scripts/run_rebuild.ps1 兼容转发到此流程；-SkipExperiments 对应 -ReportOnly。Q1 不重训，冻结产物随支撑包提供。

### 第一题（Q1）的独立复现

`run_final.ps1` 只跑 Q2/Q3 的封闭实验，不重建 Q1。若要从源代码完整复现第一题的交付评分卡，另有一个入口：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_q1.ps1
```

它按固定顺序执行，任一步失败即中止：

1. `src/rebuild/run.py --audit-only` —— 重建输入清单与审计表。注意 `run.py` 无论如何都会执行 `audit()`，所以 `outputs/rebuild/audit/raw_manifest.csv` 记录的是**当前这份输入的** SHA-256；若它与原有基线不同，脚本会改写 `raw_manifest.rebuilt.csv` 并保留原基线不动。`-SkipAudit` 只是省掉 `run_q1.ps1` 里的重复一次审计，**不能防止 manifest 被重算**。
2. `src/rebuild/run.py` —— 训练 Q1 全部模型，写出 `outputs/rebuild/q1/grades_oof_panel_mean.csv` 等 OOF 产物。
3. `src/rebuild/q1_acetaldehyde.py` —— 由附件1 实测浓度与附件8 阈值算出 acetaldehyde 的推断等级。
4. `src/rebuild/q1_complete.py` —— 组装 250×53 的交付评分卡。
5. `src/rebuild/audit_extended.py` + `src/rebuild/closed_verify.py` —— 扩展审计与一致性闸门。

前置条件：`data/raw/` 内需有官方附件 `1.csv`、`2.csv`、`3.xlsx`、`4.csv`、`5.csv`、`6.csv`、`7.csv`、`8.xlsx`、`数据说明.txt`；`data/supplementary/` 内有随包的 `Raw sensory data anonymized.csv`。缺任一项时脚本在开始前就报出具体缺失的文件名。

只想重建交付评分卡而不重训（例如只改了乙醛判定或字段映射）：依次执行第 3、4 步即可，耗时以秒计。

注意：Q1 重训会改变 `grades_oof_panel_mean.csv` 等产物。原先的冻结哈希闸门已于 2026-10-30 移除（它把每个 Q1 产物和 `run.py` 本身都做了 SHA-256，任何正当修改——哪怕只是改一行注释——都会让流程中断）。现在一致性由 `closed_verify.py` 的结果表检查保证（`Closed checks 65/65`），不再对任何文件做哈希断言。

## 代码组织（按竞赛三问对应，一题一文件）

`src/rebuild/` 的解题代码合并为三大文件，按竞赛三问一一对应：

| 问题 | 文件 | 内容 | 入口 |
| --- | --- | --- | --- |
| 问题一 | `q1.py` | 数据加载、面板监督目标、六模型对照、50 字段评分卡、乙醛判定、等级诊断 | `python src/rebuild/q1.py` |
| 问题二 | `q2.py` | 共表达网络、四条筛选规则、嵌套选择、100 次 bootstrap | `python src/rebuild/q2.py` |
| 问题三 | `q3.py` | 四种生成器、22 风格评价、四策略迁移、缺陷双口径 | `python src/rebuild/q3.py` |
| 三问共用 | `tables.py` | 汇总表、物理单位换算、跨问题统计 | `python src/rebuild/tables.py` |

辅助文件（三问共用，不含建模逻辑）：`paths.py`（输入解析）、`closed_verify.py` / `verify.py`（一致性闸门）、`closed_report.py` / `report.py` / `figures.py` / `closed_figures.py`（出图出字）、`supplement.py`（诊断）。

`q1.py` 同时承担数据层角色：`q2.py` / `q3.py` 通过 `from q1 import *` 复用其加载后的 `X`、`Y`、`targets`、`chem`、`style` 等。

## 环境与输入

依赖见 configs/rebuild_requirements.txt；PyTorch 自动检测 CUDA，本次 CPU 实际执行。PDF 需 Pandoc、XeLaTeX 和常用 Windows 字体；PDF 检查使用 pypdf（与主流程同一个 Python）。

输入分两类，**只有第二类随支撑包分发**：

1. **官方赛题附件（不随包，需自行放置）**：`1.csv`、`2.csv`、`3.xlsx`、`4.csv`、`5.csv`、`6.csv`、`7.csv`、`8.xlsx`、`数据说明.txt`。
   放到 `data/raw/`。来源为赛题官方下载页；校验值见 `outputs/rebuild/audit/raw_manifest.csv` 中对应的 9 行（该文件记录的是**原始运行输入的 SHA-256 清单**，不表示这些文件随包分发）。
2. **论文作者公开的逐评分员感官数据（随包分发）**：`data/supplementary/Raw sensory data anonymized.csv`。
   - 编码 cp1252；大小 321,065 字节；MD5 `a8f911d8c0061e096abf3b087678e164`；SHA-256 `78525c8cbbbd498578c451b06641969a20ada87d0fe4c67fe1e634ad34569628`
   - 原文件名含“（论文补充数据）”，为便于脚本读取已重命名；若自行下载，请另存为本文件名。
   - 来源 https://zenodo.org/records/23624915
   - 该文件是 Q1 监督目标的必需输入（`run.py` 直接从它构建 50 个目标的等权面板均分），因此必须能取到；放在 `data/supplementary/` 即可，无需再复制进 `data/raw/`。

代码通过 `src/rebuild/paths.py` 的 `find_input()` 解析输入：先找 `data/raw/`，再找 `data/supplementary/`。

**必需输入的最低集合**：`1.csv`（化合物矩阵，乙醛列也在这里）与 `Raw sensory data anonymized.csv`（随包）。

据实说明两种场景，避免误解：

- **只放随包的 `data/supplementary/` 时，任何脚本都跑不起来**。乙醛判定需要 `1.csv` 的实测浓度，评分卡组装需要 `outputs/rebuild/q1/grades_oof_panel_mean.csv`（已随包）与同一份 `1.csv`。缺少官方附件时脚本会明确报出缺失文件名并以退出码 1 结束，不会静默产出错误结果。
- **放置官方附件后**，可以直接跑 `run_q1.ps1` 做完整训练，也可以只跑第 3、4 步（乙醛判定 + 评分卡组装）而不重训。

`closed_verify.py` 同样需要官方附件，因为它按 `raw_manifest.csv` 逐项校验原始输入。

自检：`python src/rebuild/closed_verify.py` 会按文件名在两个输入目录中解析，并逐个比对 `raw_manifest.csv` 记录的 SHA-256，任何缺失或改动都会被指出。注意该脚本本身会导入训练模块，因此**运行前必须已有官方附件**；缺附件时它会直接报出缺失文件名并以退出码 1 结束（实测于干净解压目录）。

## 最终结果入口

- Q1：outputs/rebuild/q1 为冻结目录；交付评分卡 outputs/rebuild/q1/aroma_flavour_template_with_unobserved.csv 共 53 列、250 行、无空值：beer + 3.xlsx summary 的全部 50 个 0—6 面板监督字段 + A_acetaldehyde/F_acetaldehyde 两列阈值锚定推断等级（全为 0）。acetaldehyde 的判定依据（附件1 实测浓度全部低于附件8 的 50.0 mg/L 感官检出阈值）见 outputs/rebuild/q1_complete/acetaldehyde_grading.json，逐酒浓度与等级见 acetaldehyde_graded.csv；字段来源与监督状态见 scorecard_complete_provenance.csv；Z 先行路线的同口径对照见 scale_comparability.csv。
- Q2：outputs/rebuild/closed/q2/top3_50x3.csv 与兼容入口 outputs/rebuild/q2/top3_50x3.csv 逐字节相同；最终训练内选择为 2+1。性能、固定规则/完整选法 Bootstrap 和替代组指标都在同一目录。
- 生成：closed/generation/metrics_22styles_by_context.csv 为 264 行；metrics_22styles_summary.csv 为 88 行。原协议中 20 风格有独立测试酒，Faro 和 Flanders old brown 为 NA，相关结构仅 4 风格样本支持；不可用 NA 不被填成零。**选型逻辑见 outputs/rebuild/q3_complete/selection_logic.json 与 selection_criteria_table.csv**：交付生成器由诊断之前冻结的验证折规则决定，不是按条件 SWD 排名；CVAE 的表面优势与测试集大小相关（r=0.70），仅在小样本风格上领先。
- 迁移：closed/migration/per_beer.csv 为 9 酒 × 4 策略；nine_sources.csv 记录独立角色，evaluator_hash_check.json 证明评价器未改动；所有策略选择完才评分。**缺陷判定两套口径并列**：严格口径（任意微量上升）无约束 6/9、点约束 5/9、不确定性约束 4/9；赛题原文的超阈值口径为 7 / 0 / 0 次越阈。逐项见 outputs/rebuild/q3_complete/defect_criteria.json 与 defect_threshold_crossing.csv。
- 图表：closed/figures 含 PNG、可编辑 SVG 和 PDF；Q1 原图保留。
- 论文与报告：paper/competition_final.pdf、reports/00 至 05、08_closed_experiments.md、AI_usage_log.md。
- 提交准备：submission/论文定稿.pdf、AI工具使用详情说明.pdf、支撑材料.zip、真实签署前的官方承诺书模板及待补项。

## 两套汇总文件的分工（避免数字打架）

- outputs/rebuild/closed/final_summary.json 是**封闭补充实验的唯一权威口径**：Q3 生成指标为 22 风格 × 多上下文的宏观均值（四模型 SWD 均值 CVAE 0.7059、PCA 高斯 0.7516、Copula 联合 0.8637、Copula 独立 0.8676），迁移为 9 源酒四策略的配对结果。论文与报告中的所有封闭实验数字都取自这一份。
- outputs/rebuild/summary.json 是**早期单划分/单折聚合口径**的遗留记录（其 Q3 SWD 为 0.27—0.34 量级，production 记为 CopulaIndependent），只用于追溯上一版流程，不作为任何结论依据，也不与 final_summary.json 交叉引用。
- 两者数值不同是因为评价集合不同（单划分 vs 22 风格宏观），不是同一实验的两种结果；引用时一律注明口径。

## 复现精度（不是所有产物都能逐字节复现）

重跑同一流程时，不同组件的可复现程度不同，交付前应当知道这个差别：

- **树模型与传统模型（ET / RF / HGB / Ridge / PLS / Mean）**：固定随机种子后逐字节可复现，重跑差异在 1e-15 量级（实测量化）。
- **深度网络（MLP / Hierarchy / HierarchyRater）**：在 CPU 上仅达到 1e-1 量级一致（实测重训后个别 OOF 单元最大偏差 0.577）。这是 CPU 上算子顺序与线程调度的非确定性，不是代码 bug。后果是：`q1/grades_oof_panel_mean.csv` 在重训后可能有 1 个单元的取整等级发生翻转。
- **交付评分卡**：上一条的翻转不影响交付件——重训后 250×53 的评分卡 13,000 格与重训前 0 差异。原因是交付表取的是两次 OOF 平均后的取整等级，翻转单元未跨越取整边界。

因此：
- 若需要**逐字节一致**，请直接使用随包的冻结产物，不要重训。
- 若需要**从源码重建**，树模型部分可完全对上；深度网络部分允许 1e-1 量级的数值差异，交付评分卡仍应完全一致。若评分卡出现差异，说明流程被改动了，应视为异常而不是随机波动。

## 科学边界与参赛独立性

自动选法未超过固定 2+1；条件生成指标有取舍；迁移约束在训练代理下有效，但独立评价器下 stale_hops 仍有 4—5/9 款上升（无约束 6/9），因此不宣称“约束保证缺陷不恶化”。不得把观测编码不增加写成地板以下真实浓度保证，也不得宣称完成真实酿造盲评。

本工程如实披露 AI 主导参与。队员的真实独立核心工作、人工修改、队号和签署尚未提供，未伪造。官方要求核心建模、算法、求解与分析由队员独立完成，审阅签字不能代替。提交准备包尚需队伍完成这些真实事项，并按实际队号命名；未自动上传竞赛系统。
