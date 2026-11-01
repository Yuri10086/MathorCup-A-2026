# Rebuilds Question 1 end to end from source, in the required order:
#   1. audit the raw inputs against the recorded SHA-256 manifest
#   2. train Q1 models and write the out-of-fold panel-mean grades
#   3. recompute the threshold-anchored acetaldehyde score
#   4. assemble the 53-column scorecard deliverable
param([string]$Python='',[switch]$SkipAudit)
$ErrorActionPreference='Stop'
$Root=Split-Path $PSScriptRoot -Parent
Set-Location $Root
$env:PYTHONIOENCODING='utf-8'

function Resolve-Tool([string]$Explicit,[string]$Command,[string[]]$Fallbacks){
 if($Explicit){ if(-not (Test-Path $Explicit)){throw "Given path not found: $Explicit"}; return $Explicit }
 $c=Get-Command $Command -ErrorAction SilentlyContinue
 if($c){return $c.Source}
 foreach($f in $Fallbacks){ if($f -and (Test-Path $f)){return $f} }
 throw "Cannot locate '$Command'. Install it or pass it explicitly (see README_REBUILD.md)."
}
if(-not $Python){
 $Python=Resolve-Tool '' 'python' @("$env:CONDA_PREFIX\python.exe","$env:USERPROFILE\anaconda3\python.exe","C:\Anaconda\python.exe")
}
Write-Host "python : $Python"

# Fail fast with an actionable message if an input is missing.
$needed=@('1.csv','4.csv','Raw sensory data anonymized.csv')
foreach($n in $needed){
 $found=$false
 foreach($d in @('data/raw','data/supplementary')){ if(Test-Path (Join-Path $d $n)){ $found=$true } }
 if(-not $found){ throw "Missing input '$n'. Official attachments go in data/raw; the author's public file goes in data/supplementary. See README_REBUILD.md." }
}
foreach($n in @('2.csv','3.xlsx','7.csv','8.xlsx')){
 if(-not (Test-Path (Join-Path 'data/raw' $n)) -and -not (Test-Path (Join-Path 'data/supplementary' $n))){
  Write-Warning "Optional-but-recommended input '$n' not found; audit_extended.py and the 附件7 supplement will be skipped."
 }
}

function Invoke-Stage([string]$Name,[string[]]$StageArgs){
 Write-Host $Name
 $taskPreviousPreference=$ErrorActionPreference
 try {
  $ErrorActionPreference='Continue'
  & $Python @StageArgs *> "outputs/rebuild/logs/$Name.log"
  $taskExitCode=$LASTEXITCODE
 } finally {$ErrorActionPreference=$taskPreviousPreference}
 if($taskExitCode -ne 0){throw "Stage $Name failed; inspect outputs/rebuild/logs/$Name.log"}
}

# 1. Q1 training run.
#    -SkipAudit only skips the redundant standalone audit pass above; it does NOT protect the
#    input baseline. run.py calls audit() unconditionally at import/entry, so the manifest is
#    recomputed either way. That is safe by design: audit() writes raw_manifest.rebuilt.csv and
#    leaves the recorded original baseline untouched whenever the inputs differ from it.
if(-not $SkipAudit){ Invoke-Stage 'q1_audit' @('-u','src/rebuild/q1.py','audit_extended') }
Invoke-Stage 'q1_train' @('-u','src/rebuild/q1.py')
# 2. Threshold-anchored acetaldehyde score (needs 附件1 1.csv and 附件8 thresholds).
Invoke-Stage 'q1_acetaldehyde' @('-u','src/rebuild/q1.py','q1_acetaldehyde')
# 3. Assemble the 53-column deliverable.
Invoke-Stage 'q1_complete' @('-u','src/rebuild/q1.py','q1_complete')
# 4. Extended audits and the consistency gate.
Invoke-Stage 'q1_audit_extended' @('-u','src/rebuild/q1.py','audit_extended')
Invoke-Stage 'q1_verify' @('-u','src/rebuild/closed_verify.py')

Write-Host ''
Write-Host 'Q1 rebuilt. Deliverable: outputs/rebuild/q1/aroma_flavour_template_with_unobserved.csv (250 x 53).'
Write-Host 'Note: the SHA-256 frozen-hash gate was removed on 2026-10-30, so a retrain no longer reports'
Write-Host 'frozen-hash mismatches. closed_verify.py now validates the result tables themselves;'
Write-Host 'it should print "Closed checks 65/65". A non-zero exit means a real regression.'
