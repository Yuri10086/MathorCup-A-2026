# MathorCup A题高级科研可视化｜Python全套

这份工程只制作**数据与实验结果图**。**图1三问总技术路线图由draw.io绘制，不在此包内。**

## 一键执行

Windows解压后在文件夹中执行：

```powershell
pip install -r requirements.txt
python plot_all.py
```

Linux/macOS类似。默认使用压缩包内的`support/`（来自上传的`MathorCup_A_final_support.zip`的**真实结果CSV**及官方`1.csv`），输出至`publication_figures/`。

若希望读取Codex原有工程内的最新结果，运行：

```powershell
python plot_all.py --project "D:/MathorCup2026_A" --output "D:/MathorCup2026_A/paper_figures"
```

或调整 `--dpi 300` 指定PNG分辨率。需要满足 `outputs/rebuild/` 的数据目录结构，且工程根目录或`data/raw/`有`1.csv`。所有图均由**真实CSV重算**，不会生成模拟实验数据。此脚本不执行任何模型训练。

## 主文图件（图1在draw.io单独完成）

| 图号 | 文件前缀 | 论文位置 | 数据来源 |
|---|---|---|---|
| 图2 | `02_数据审计_风格与最低值` | 4章 | audit/styles, audit/features |
| 图3 | `03_Q1_模型性能与维度差异` | 5.3 | q1/repeat_metrics, per_target |
| 图4 | `04_Q1_特征遮蔽与标签敏感性` | 5.5 | q1/sensitivity |
| 图5 | `05_Q2_有符号化合物网络` | 6.1 | q2/signed_edges |
| 图6 | `06_Q2_误差稳定性与冗余` | 6.3–6.4 | closed/q2/summary_metrics, stability_summary |
| 图7 | `07_Q3_生成模型六维对照` | 7.5 | closed/generation/macro_summary |
| 图8 | `08_Q3_三风格感官画像` | 7.3 | q3/generated_sensory50, audit/equal_rater_targets, 1.csv |
| 图9 | `09_Q3_逐风格评估支持范围` | 7.4 | closed/generation/style_coverage, metrics_22styles_summary |
| 图10 | `10_Q3_联合低值与相关结构` | 7.5 | closed/generation/metrics_22styles_summary |
| 图11 | `11_Q3_九酒迁移配对效果` | 7.7 | closed/migration/per_beer |
| 图12 | `12_Q3_迁移收益风险与拒绝率` | 7.7 | closed/migration/accepted_and_all |

## 追加5张附录图

`S01_Q1_评分真实预测密度`, `S02_Q1_50感官误差热图`, `S03_Q2_150化合物稳定性总图`, `S04_Q3_生成网络训练曲线`, `S05_Q3_单酒化学变化`。

图3用两轮结果统计误差条，图6分开比较固定2+1与训练内选择，图9对没有留出样本的风格保留空值。图11–12中的风险指标是**独立代理的预测变化**，不能解释成真实酿造安全认证。

## 输出与审稿

* `publication_figures/*.pdf`：可直接置入LaTeX，文本矢量。
* `publication_figures/*.svg`：可进一步编辑，文本尽可能保持矢量。
* `publication_figures/*.png`：高清预览。
* `publication_figures/figure_manifest.json`：每张图对应的输入CSV以及解释边界。
* `run.log`：本次一键运行日志。

**视觉规范**：白底、深蓝/青绿/灰/橙红、(a)(b)(c)多面板、数据与注释分层。所有中文标题在最终输出PDF中自行检查字型；Windows优先支持雅黑/黑体，Linux自动注册Noto CJK字体（字体不随包分发）。

**生成说明**：此Python可视化脚本采用AI辅助编写；原始模型结果来自你们现有封闭实验包。正式论文的图注和AI使用说明应据实披露脚本贡献、并由队员确认结果。
