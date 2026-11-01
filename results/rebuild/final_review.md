# 最终内容与版式复核

- 原始文件与实验输出一致性：全部检查项通过，详见 verification.json 与 `python src/rebuild/closed_verify.py` 输出的 `Closed checks 101/101`。
- PDF 版式以 outputs/rebuild/closed/pdf_quality.json 为唯一记录，本文件不再另记页数以免版本漂移。当前记录：总 73 页，摘要 1 页、目录 1 页、正文 19 页、附录自第 22 页起，正文未超过 30 页。注意早期版本曾记 36 页、55 页、60 页、61 页与 69 页，均为旧稿；任何页数引用一律以 pdf_quality.json 为准。
- 最终 XeLaTeX 日志无缺字与 Overfull 警告（DMS/Diacetyl/4VG/trans-2-nonenal/Stale hops 这一长串缺陷名曾在正文造成 12.5pt 过宽，已在构建脚本中加入允许断行处理）；已渲染检查摘要、目录、正文、图表及 Top-3 附录。
- 文本检查器提示 3 类候选问题。语境复核：长横线主要用于数值范围与流程连接；三项并列对应三个真实任务；连续句式部分为模型配置及结果索引。保留有科学含义的表述，未按机械规则删去边界说明。
- 结论边界复核：未宣称深度学习胜过树模型；未宣称完整前向选择胜过 2+1；未宣称约束能保证地板以下真实浓度不增加；未宣称迁移约束能保证独立缺陷不恶化；未将内部重采样称为独立外部队列。
- Q1 评分卡已按 3.xlsx summary 的 50 字段定义交付（此前 30 字段系列名前缀过滤所致）；两列 acetaldehyde 由附件1 实测浓度对附件8 感官检出阈值判定后取 0（250 款酒全部低于 50.0 mg/L），判据见 outputs/rebuild/q1_complete/acetaldehyde_grading.json，交付表 53 列、无空值。该修复不涉及重训，记录见 outputs/rebuild/audit/protocol_amendment.md。
- Q3 生成器选型：交付模型由诊断之前冻结的验证折规则决定，未按条件 SWD 排名回改。CVAE 的表面 SWD 优势与测试集大小相关（r=0.70，仅 1—2 款测试酒的风格上领先，5 款以上时优势消失）；量化见 outputs/rebuild/q3_complete/selection_logic.json。
- Q3 迁移仅 9 款不同源酒，逐划分记录存在重复；统计按 beer 聚合后重采样，Wilcoxon 结果另给 Holm 校正。缺陷判定两套口径并列：严格口径（任意微量上升）stale_hops 上升为无约束 6/9、点约束 5/9、不确定性约束 4/9；赛题原文的超阈值口径（源未超阈值而迁移后越阈）为 7/0/0 次。两套口径差 0.05 分以内，远小于模型在 stale_hops 上的约 0.23 MAE，故只主张超阈值口径下硬约束有效，严格口径的残余风险列为未解决局限。逐项见 outputs/rebuild/q3_complete/defect_criteria.json。
- 论文与图表为 AI 辅助研究参考。人工决策表尚未填写，未声称已由队员独立完成。
