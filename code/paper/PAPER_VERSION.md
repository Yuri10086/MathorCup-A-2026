# 论文文稿版本说明

本目录只保留一条论文生成链。看到多份 `.md` / `.tex` 不是不同版本，而是同一次生成的源文件与切片：

| 文件 | 角色 | 说明 |
| --- | --- | --- |
| `competition_final.pdf` | **唯一交付件** | 供上传的定稿 PDF。正文页数、摘要/目录位置以 `outputs/rebuild/closed/pdf_quality.json` 为准，本文件不另记页数。 |
| `competition_draft.md` | **唯一内容源** | 含摘要、正文、参考文献、AI 声明，是 `closed_report.py` 读写的权威 Markdown。`rebuild_body.md`、`rebuild_abstract.txt` 都是它的切片，逐字节可由它推出。 |
| `competition_draft.tex` | 生成中间件 | 由 `build_rebuild_pdf.py` 从上面两个切片 + 附录经 pandoc 生成，供 XeLaTeX 编译。不建议手改，重跑即覆盖。 |
| `rebuild_body.md` / `rebuild_abstract.txt` / `rebuild_appendix.md` | 切片 | 供 pandoc 分块输入；内容与 `competition_draft.md` 一致。 |
| `build_rebuild/` | 编译临时目录 | LaTeX 日志与中间产物，不随支撑包分发。 |

## 修改流程（不要手改 PDF 或只改 Markdown）

1. 改 `src/rebuild/closed_report.py` 中的论文模板，或改 `paper/competition_draft.md` 里被 `closed_report.py` 原样切出的章节（问题一、数据审计、参考文献）。
2. `python -u src/rebuild/closed_report.py` —— 重建 `competition_draft.md` 及其切片。
3. `python scripts/build_rebuild_pdf.py` —— 重建 PDF。
4. `python scripts/final_quality.py` —— 刷新 `pdf_quality.json`。
5. `python src/rebuild/closed_verify.py` —— 应输出 `Closed checks 65/65`（2026-10-30 移除 36 项 SHA-256 哈希检查，检查项由 101 降为 65）。

## 已废弃的旧稿

早期版本（76 项校验的网页端稿、36 页/55 页/60 页/61 页的旧 PDF）不在本目录留存，也没有在支撑包内。任何页数或字段计数一律以生成后的文件为准，不以历史描述为准。
