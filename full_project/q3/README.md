# MathorCup赛道A·第三问正式实验复核包

本包包含已运行的全部输出及源程序。Q2独立包已交付，不依赖Q3；Q3复用Q1公开的原论文0–6原始品评参考均值，只用于感官代理，不用同批数据声称外部测试泛化。

输入文件：官方`1.csv`、`4.csv`及Q1产生的`reference_raw_panel_means_250.csv`，不能混淆同名前缀的其他文件。

## 一次复现（约需依赖torch/scipy/sklearn/numpy/pandas/matplotlib）

```bash
pip install -r requirements.txt
python run_all.py --data-dir "官方附件文件夹" --q1-reference "reference_raw_panel_means_250.csv"
```

训练默认200 epochs, 固定随机数。所有结果输出到`results/`。代码无需修改路径，但须用自己的真实文件路径。正式论文采用的结果已生成，参赛队须独立运行并人工检查。

## 已运行结果

- `generated_22styles_chemical231.csv`: 880×231风格条件合成化合物谱（加最低值质量校准）。
- `generated_22styles_sensory_0to6.csv`: 与生成化合物谱对应的50维感官代理预测。
- `flavor_fingerprint_by_style.csv`: 22风格的50维风味指纹。
- `q3_censor_repeated_summary.json`: 三随机留出实验，最低值误差下降38.0%–42.4%。
- `holdout_style_diagnostics.csv`: 20风格留出 SWD、风格分类率；条件流对照。
- `transfer_ablation.csv`: 无约束、四项、五项、五项优化的逐级消融。
- `pils_to_tripel_optimized_chemical231.csv`: 11款源啤酒的231维迁移谱。
- `pils_to_tripel_optimized_audit_11.csv`: 源缺陷、代理、目标风格距离逐行核验。
- `verification.json`: 输出独立校验，检查0违反、2个缺失的原始双乙酰观测。

**重点**：检测下限只由最低值集中堆积推断为疑似截尾，不是经过实验室确认的检出限。4项化学缺陷是数值不恶化约束，缺失的2项无法作为真实观测进行比较；第五项`stale_hops`是感官代理约束。原论文的逐评分员数据与官方250款对应，不能冒充外部验证集。

实际设计与代码均受OpenAI GPT-6辅助，参赛队应独立核查、重跑、修改并按官方AI工具规定披露。
