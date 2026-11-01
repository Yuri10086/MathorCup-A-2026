from pathlib import Path
import json,hashlib,os,zipfile,shutil
R=Path(__file__).resolve().parents[1];S=R/'submission';S.mkdir(exist_ok=True)
import time
def drop(p):
 for i in range(8):
  try:
   if p.exists():p.unlink()
   return True
  except PermissionError:
   if i==7:return False
   time.sleep(0.8)
def same(src,dst):
 try:return dst.exists() and hashlib.sha256(src.read_bytes()).hexdigest()==hashlib.sha256(dst.read_bytes()).hexdigest()
 except OSError:return False
def copy(src,dst):
 """Copy src over dst. If dst is byte-identical, leave it alone (avoids touching a file a PDF
 viewer has open). Overwriting in place is tried even when unlink fails, because a locked
 destination can often still be rewritten truncating it; a numbered sibling is the last resort."""
 if same(src,dst):
  print('  unchanged',dst.name);return dst
 drop(dst)
 try:
  shutil.copy2(src,dst);return dst
 except PermissionError:
  pass
 for k in range(2,10):
  alt=dst.with_name(dst.stem+'_%d'%k+dst.suffix)
  if same(src,alt):print('  unchanged',alt.name);return alt
  drop(alt)
  try:
   shutil.copy2(src,alt);print('  LOCKED ->',alt.name);return alt
  except PermissionError:
   continue
 raise PermissionError('cannot place '+str(dst))
def write(dst,text):
 for i in range(8):
  try:dst.write_text(text,encoding='utf-8');return
  except PermissionError:
   if i==7:raise
   time.sleep(0.8)
TEAM=os.environ.get('MC_TEAM_ID','').strip().upper()
def official(name):return (TEAM+name) if TEAM else name
copy(R/'paper/competition_final.pdf',S/official('.pdf') if TEAM else S/'论文定稿_待填队号.pdf')
copy(R/'reports/AI工具使用详情说明.pdf',S/'AI工具使用详情说明.pdf')
letters=sorted([p for p in R.rglob('*承诺书*.docx') if S not in p.parents]+[p for p in R.parent.rglob('*承诺书*.docx') if 'MathorCup_A_final_support' not in p.parts])
if letters:
 copy(letters[0],S/(official('承诺书')+'.docx' if TEAM else '承诺书_官方空白模板_待真实签署.docx'))
readiness='''# 提交材料核查

## 一、技术状态

三组封闭实验、Q1 全字段评分卡、论文 PDF、图表、源程序、结果 CSV、复现协议与 AI 使用详情 PDF 均已生成。版式以 outputs/rebuild/closed/pdf_quality.json 为唯一记录（正文页数未超 30 页、摘要第 1 页、目录第 2 页、无页眉、无身份信息，具体页数一律查该文件，本说明不另记以免版本漂移）。内容一致性由 `python src/rebuild/closed_verify.py` 输出 Closed checks 65/65 保证（2026-10-30 移除 36 项 SHA-256 冻结哈希检查，由 101 项降为 65 项）。

Q1 评分卡按 3.xlsx summary 的 50 个字段定义交付：53 列 = 1 列 beer + 50 个由感官监督模型预测的 0—6 字段 + 2 个 acetaldehyde 阈值锚定推断字段（当前均为 0，属非面板评分），250 行、无缺失值。此前只交 30 列是列名前缀过滤造成的实现缺陷，不涉及重训；修复与乙醛判定依据见 outputs/rebuild/audit/protocol_amendment.md 与 outputs/rebuild/q1_complete/acetaldehyde_grading.json。

## 二、上传前必须由队伍完成的事项

1. **真实队号**。官方规范要求论文文件名为「#*******.pdf」（# 为题号，******* 为参赛队号，如 A2600001.pdf）。本包不能虚构队号，当前为 `论文定稿_待填队号.pdf`。确定队号后请设置环境变量 `MC_TEAM_ID`（例如 `A2600001`）后重跑 `python scripts/package_final.py`，脚本会自动生成 `A2600001.pdf`、`A2600001承诺书.docx` 与 `A2600001附件.zip`。
2. **承诺书签署**。本包内为官方空白模板，须每位队员手写或电子签名，再扫描/拍照为 PDF 或图片，命名「#*******承诺书」（如 A2600001承诺书）。不得代签。
3. **队员独立核心工作记录**。现有核心方案与分析由 AI 主导生成，尚无队员独立复算、人工修改与采纳记录。官方《人工智能工具使用规定》第二条要求核心建模、算法设计、求解与分析由队员独立完成，审阅签字不能替代。具体操作清单见随包 `队员独立复核指引.md`（三人分工、每人要读的三处代码、要动手改的一处、要核对的数字），完成后把真实修改与采纳写入 reports/AI_usage_log.md 与 outputs/rebuild/audit/human_decision_record.csv，再决定正式提交版本。
4. **支撑材料命名**。官方要求「#*******附件.rar」或「#*******附件.zip」。本包内 支撑材料.zip 即该压缩包。按官方规范，赛题原始数据可不放入，故 zip 内不含 data/raw；但论文作者公开的逐评分员感官数据是模型必需输入，已随包放在 data/supplementary/（cp1252 编码，MD5 a8f911d8c0061e096abf3b087678e164），属可再分发数据。
5. **AI 声明位置**。论文正文参考文献后已列 AI 使用声明，支撑材料内已含《AI工具使用详情说明》(PDF)，符合规定第三、五条。

## 三、不得在提交时改动的科学边界

- 自动选法未超过固定 2+1；条件生成指标随指标改变。
- 迁移约束按两套口径报告：赛题定义的超阈值口径下，越阈次数由无约束 7 次降到点约束 0 次、不确定性约束 0 次；独立评价器任意微量上升的严格口径下仍有 6/9、5/9、4/9 款酒上升，该残余风险已作为未解决局限写入正文。两种口径差 0.05 分以内，远小于模型在 stale_hops 上的约 0.23 MAE。
- Q1 交付评分卡存在已披露的等级压缩局限：预测等级全部落在 0—4 级，真值 ≥5 的 121 个单元被平均低估 2.07，连续预测上限仅 4.34，故整体 MAE 可用而高强度尾部不可用，见 outputs/rebuild/q1_complete/grade_range_summary.json。
- 观测编码层面的浓度不增加不等于地板以下真实浓度保证；没有真实新酿造与盲评验证。

本文件夹为提交准备材料，未上传竞赛系统。论文与 AI 声明保留真实参与事实。

## 四、本压缩包内含什么、不含什么

为避免"说明里提到的文件其实不在包里"这类误会，这里逐项写清：

- **包内含**：`src/` 全部源程序、`outputs/rebuild` 全部结果 CSV 与图表、`reports/`、`paper/`（含定稿 PDF）、`configs/`、`scripts/`、`data/processed/`、`data/supplementary/Raw sensory data anonymized.csv`（随包必要的补充数据）、本说明文件与 `AI工具使用详情说明.pdf`。
- **包内不含**：赛题官方附件（按规范可不放入，需由使用者自行下载后放入 `data/raw/`）、`outputs/rebuild/closed/delivery_manifest.json`（包自身的清单，避免自引用）。
- **不随包分发、需在本地生成或单独处理**：`submission/` 下的论文 PDF 副本、承诺书、支撑材料 zip 本身（这些是"提交物"，不应该嵌套进自己），以及填好后的承诺书扫描件。`

上面第 1–3 条完成后，用 `MC_TEAM_ID` 重新运行 `python scripts/package_final.py` 即可按官方命名生成这三份提交物。
''';(write(S/'提交说明与待补项.md',readiness))
paths=[]
for folder in ['src/rebuild','outputs/rebuild']:
 paths += [p for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['delivery_manifest.json']]
paths += [R/'README_REBUILD.md',R/'队员独立复核指引.md',R/'configs/rebuild_protocol.json',R/'configs/closed_final_protocol.json',R/'configs/rebuild_requirements.txt']
paths += [R/'scripts'/n for n in ['run_final.ps1','run_q1.ps1','run_rebuild.ps1','build_rebuild_pdf.py','final_quality.py','package_final.py']]
paths += list((R/'reports').glob('0[0-5]_*.md'))+[R/'reports/08_closed_experiments.md',R/'reports/AI_usage_log.md',R/'reports/AI工具使用详情说明.pdf']
# paper/: competition_draft.md is the canonical single source (abstract+body+refs+declaration);
# rebuild_body.md / rebuild_abstract.txt are slices of it, and competition_draft.tex is generated FROM
# them, so shipping all of them is consistent. The PDF is the deliverable the tex produces.
paths += [R/'paper'/n for n in ['competition_draft.md','competition_draft.tex','competition_final.pdf','rebuild_body.md','rebuild_abstract.txt','rebuild_appendix.md']]
paths += [R/'paper/PAPER_VERSION.md']
paths += [R/'data/processed'/n for n in ['chemical_names.csv','transforms.csv','thresholds.csv','template_hierarchy.csv','scale_pair_validation.csv']]
# The author's public per-taster file is redistribution-safe and required by run.py, so it ships.
paths += [p for p in (R/'data/supplementary').glob('*') if p.is_file()]
paths += [p for p in (R/'outputs/review_20261028').glob('*') if p.is_file()]
paths += [S/'提交说明与待补项.md',S/'AI工具使用详情说明.pdf']
files=sorted(set(p for p in paths if p.is_file()));manifest=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files];mp=R/'outputs/rebuild/closed/delivery_manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');files.append(mp)
z=S/(official('附件')+'.zip' if TEAM else '支撑材料.zip')
legacy=S/'支撑材料.zip'
def drop(p):
 for i in range(6):
  try:
   if p.exists():p.unlink()
   return
  except PermissionError:
   if i==5:raise
   time.sleep(0.6)
for p in {z,legacy,R/'outputs/MathorCup_A_final_support.zip'}:drop(p)
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as f:
 for p in files:f.write(p,p.relative_to(R))
with zipfile.ZipFile(z) as f:assert f.testzip() is None
if z!=legacy:copy(z,legacy)
status={'team_id':TEAM or None,'files':len(files),'zip':z.name,'zip_bytes':z.stat().st_size,'support_sha256':hashlib.sha256(z.read_bytes()).hexdigest(),'formal_submission_readiness':'pending real team ID, signatures, independently completed core work','uploaded':False};(S/'submission_manifest.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8');copy(z,R/'outputs/MathorCup_A_final_support.zip');print(status)
