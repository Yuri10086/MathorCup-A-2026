from pathlib import Path
import os,shutil,subprocess,sys,json,re
R=Path(__file__).resolve().parents[1];P=R/'paper';B=P/'build_rebuild';B.mkdir(exist_ok=True)
def tool(env,cmd,fallbacks):
 if os.environ.get(env):return os.environ[env]
 found=shutil.which(cmd)
 if found:return found
 for f in fallbacks:
  if f and os.path.exists(f):return f
 raise SystemExit(f'Cannot find {cmd}; set the {env} environment variable to its full path')
pandoc=tool('MC_PANDOC','pandoc',['C:/Anaconda/Library/bin/pandoc.exe',r'C:\Program Files\Pandoc\pandoc.exe'])
xelatex=tool('MC_XELATEX','xelatex',['C:/texlive/2026/bin/windows/xelatex.exe','C:/texlive/2025/bin/windows/xelatex.exe'])
def run(cmd,cwd):
 p=subprocess.run(cmd,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace');(B/('pandoc.log' if 'pandoc' in cmd[0] else 'xelatex_console.log')).write_text(p.stdout,encoding='utf-8')
 if p.returncode:raise RuntimeError(p.stdout[-4000:])
for source,out in [('rebuild_body.md','body.tex'),('rebuild_appendix.md','appendix.tex')]:run([pandoc,str(P/source),'-f','markdown','-t','latex','--top-level-division=section','-o',str(B/out)],P)

for fname in ['body.tex','appendix.tex']:
 pp=B/fname;tt=pp.read_text(encoding='utf-8');tt=tt.replace('.png}', '.pdf}')
 def format_table(match):
  block=match.group(0)
  def spec(m):
   n=len(m.group(1));return r'{@{}*{'+str(n)+r'}{>{\raggedright\arraybackslash}p{\dimexpr\linewidth/'+str(n)+r'-2\tabcolsep\relax}}@{}}'
  block=re.sub(r'\{@\{\}([lrc]+)@\{\}\}',spec,block)
  block=block.replace(r'\_',r'\_\allowbreak{}')
  block=re.sub(r'(?<=[A-Za-z])\.(?=[A-Za-z])',r'.\\allowbreak{}',block)
  block=block.replace('CopulaIndependent',r'\seqsplit{CopulaIndependent}').replace('X2.methyltetrahydrofuran.3.one',r'\seqsplit{X2.methyltetrahydrofuran.3.one}')
  return r'\begingroup\footnotesize'+'\n'+block+'\n'+r'\endgroup'
 tt=re.sub(r'\\begin\{longtable\}[\s\S]*?\\end\{longtable\}',format_table,tt)
 tt=tt.replace(r'top3\_chemical\_annotations.csv',r'\path{top3_chemical_annotations.csv}')
 tt=tt.replace('DMS/Diacetyl/4VG/trans-2-nonenal/Stale hops',
               r'DMS/\allowbreak{}Diacetyl/\allowbreak{}4VG/\allowbreak{}trans-2-nonenal/\allowbreak{}Stale hops')
 tt=tt.replace('DMS/Diacetyl/4VG/trans-2-nonenal/Stale~hops',
               r'DMS/\allowbreak{}Diacetyl/\allowbreak{}4VG/\allowbreak{}trans-2-nonenal/\allowbreak{}Stale~hops')

 for _p in ['defect_threshold_crossing.csv','acetaldehyde_grading.json','selection_criteria_table.csv']:
  tt=tt.replace(_p,_p.replace('.',r'\allowbreak{}.',1))
 pp.write_text(tt,encoding='utf-8')

header=r'''\documentclass[UTF8,a4paper,zihao=-4]{ctexart}
\usepackage[margin=2.3cm]{geometry}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Consolas}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,array,calc,multirow}
\usepackage{hyperref,xurl,float,caption,fancyvrb,seqsplit}
\hypersetup{hidelinks}
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}}
\newcommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setcounter{tocdepth}{1}
\setlength{\emergencystretch}{4em}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.25em}
\renewcommand{\arraystretch}{1.15}
\captionsetup{font=small}
\pagestyle{plain}
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>0.78\textheight0.78\textheight\else\Gin@nat@height\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
\begin{document}
'''
abstract=(P/'rebuild_abstract.txt').read_text(encoding='utf-8');abstract=abstract.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_');tex=header+r'''\pagenumbering{gobble}
\begin{center}{\zihao{3}\bfseries 观测语义约束下的啤酒风味预测、化合物筛选与条件生成}\end{center}
\begin{center}\textbf{摘要}\end{center}
'''+abstract+r'''

\noindent\textbf{关键词：}面板评分；训练内选择；替代组稳定性；条件生成；独立风险

\vfill\noindent\small AI辅助研究参考稿。核心建模与分析尚待参赛队独立完成，不作为独立原创正式提交稿。
\newpage\normalsize\tableofcontents\newpage
\pagenumbering{arabic}
\input{build_rebuild/body.tex}
\clearpage\appendix
\input{build_rebuild/appendix.tex}
\section{复现环境与全部新增程序}
以下为本轮新增计算程序，均含AI辅助说明。原始数据未修改。运行命令：Q2/Q3 封闭实验与论文重生成见 scripts/run\_final.ps1（scripts/run\_rebuild.ps1 为其兼容转发入口）；Q1 从源码完整重建见 scripts/run\_q1.ps1。脚本依赖见 configs/rebuild\_requirements.txt。
'''
# Official rules require the appendix to contain every program actually used. List every src/rebuild
# module plus every script referenced from the text, including the Q1-only entry point run_q1.ps1.
for p in sorted((R/'src/rebuild').glob('*.py'))+[R/'scripts'/n for n in ['run_final.ps1','run_q1.ps1','run_rebuild.ps1','build_rebuild_pdf.py','final_quality.py','package_final.py']]:
 tex+='\n\\subsection{'+p.name.replace('_',r'\_')+'}\n'+r'\VerbatimInput[fontsize=\tiny,breaklines,breakanywhere]{../'+str(p.relative_to(R)).replace(chr(92),'/')+'}\n'
tex+='\n'+r'\end{document}';(P/'competition_draft.tex').write_text(tex,encoding='utf-8')
for _ in range(2):run([xelatex,'-interaction=nonstopmode','-halt-on-error','-output-directory=build_rebuild','competition_draft.tex'],P)
(P/'competition_draft.pdf').write_bytes((B/'competition_draft.pdf').read_bytes());(P/'competition_final.pdf').write_bytes((B/'competition_draft.pdf').read_bytes())
# Explicit current-run AI details, rather than outdated prior PDF.
ai=r'''\section*{AI 工具使用详情说明}
工具：OpenAI Codex，历史GPT-6系列及本轮gpt-6.1-sol，OpenAI，2026年10月28日。

使用目的与环节：本轮数据与代码审计、逐评分员目标重建、基线及结构模型编程、CPU实际实验、统计与消融、生成及迁移分析、图表、中文报告和排版。原始文件未覆盖。

关键交互：用户要求接手既有三问工程，对现有三问独立审计、冻结协议并实际运行，关注重复评分员、共线冗余、风格混杂、最低值堆积和独立评价器。AI据此实施 configs/rebuild\_protocol.json 中的实验；运行中修正深度模型重训样本不等与随机投影不共用两处实现问题，未扩大搜索模型范围。本轮仅执行Q2统一、22风格生成评价、9源酒四策略迁移三组封闭补充，之后停止建模探索；Q1不重训、不改预测，仅在2026-10-28修正评分卡交付的字段过滤缺陷（30字段→3.xlsx summary全部50字段），记录见 outputs/rebuild/audit/protocol\_amendment.md。

实际产出：src/rebuild 的程序、outputs/rebuild 的CSV及训练日志、reports/00至05的分析和 paper/competition\_draft 的参考论文。历史模型、网页端稿件和MC2026A被当作参考，未预设其正确。

采纳与人工修改记录：当前尚无可核实的参赛队人工修改或独立核心建模记录，outputs/rebuild/audit/human\_decision\_record.csv 明确标为待完成。不得把本轮AI主导研究描述为队员独立完成。正式参赛前应由队员独立完成核心建模、算法设计、求解与分析，并记录实际采纳、修改和弃用内容。审阅签字不能替代该要求。

证据追溯：原始哈希、协议、训练日志、结果CSV、代码与PDF检查记录均保留在项目内。关键结论以真实输出为准；不成功的深度模型和不能验证的物理缺陷风险均如实报告。

详细历史交互摘要见 reports/AI\_usage\_log.md；该说明不伪造完整聊天导出，也不伪造队员审核意见。
'''
(P/'ai_details.tex').write_text(header+ai+r'\end{document}',encoding='utf-8');run([xelatex,'-interaction=nonstopmode','-halt-on-error','-output-directory=build_rebuild','ai_details.tex'],P);(R/'reports/AI工具使用详情说明.pdf').write_bytes((B/'ai_details.pdf').read_bytes());print('PDFs built',flush=True)
