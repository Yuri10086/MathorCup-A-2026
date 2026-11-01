# Overleaf 可编译版（主论文）

1. Overleaf > New Project > Upload Project，上传 `MathorCup_Overleaf_Ready.zip`，无需二次手工嵌套文件夹。
2. Menu > Settings > Compiler = **XeLaTeX**（不要选择 pdfLaTeX）。
3. Main document = **main.tex**（顶层）；如需调试组件示例可改成 `example.tex`。
4. 点击 Recompile。字体优先使用 Windows 字体（如已安装），否则自动使用 TeX Live 附带字体：英文 TeX Gyre Termes/Latin Modern，中文 Fandol 宋体。
5. 论文主体在 `sections/template-content.tex`，封面信息在 `main.tex`。

**注意**：备用字体与 Windows SimSun/Times New Roman 字形和行宽可能有差异。正式提交前请对照官方 Word 格式人工检查。

AI 使用详情说明是独立提交文件，应当单独上传 `MathorCup_AI_Report_Overleaf_Ready.zip` 新建另一个 Overleaf 项目，防止多个 `main.tex` 冲突。

此处保留排版示例而非参赛实测数据，正式写作须替换。
