param([string]$Python='',[string]$Pandoc='',[string]$XeLaTeX='',[switch]$ReportOnly)
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
# Toolchain is passed down to build_rebuild_pdf.py so the builder never hardcodes a machine path.
if(-not $Pandoc){ $Pandoc=Resolve-Tool '' 'pandoc' @("C:\Anaconda\Library\bin\pandoc.exe","$env:ProgramFiles\Pandoc\pandoc.exe") }
if(-not $XeLaTeX){ $XeLaTeX=Resolve-Tool '' 'xelatex' @('C:\texlive\2026\bin\windows\xelatex.exe','C:\texlive\2025\bin\windows\xelatex.exe') }
$env:MC_PANDOC=$Pandoc
$env:MC_XELATEX=$XeLaTeX

Write-Host "python  : $Python"
Write-Host "pandoc  : $Pandoc"
Write-Host "xelatex : $XeLaTeX"

function Invoke-Stage([string]$Name,[string[]]$StageArgs){
 Write-Host $Name
 $taskPreviousPreference=$ErrorActionPreference
 try {
  $ErrorActionPreference='Continue'
  & $Python @StageArgs *> "outputs/rebuild/logs/$Name.log"
  $taskExitCode=$LASTEXITCODE
 } finally {$ErrorActionPreference=$taskPreviousPreference}
 if($taskExitCode -ne 0){throw "Stage $Name failed; inspect its log"}
}
if(-not $ReportOnly){
 Invoke-Stage 'closed_q2' @('-u','src/rebuild/q2.py')
 Invoke-Stage 'closed_generation' @('-u','src/rebuild/q3.py','generation')
 Invoke-Stage 'closed_migration' @('-u','src/rebuild/q3.py','migration')
}
Invoke-Stage 'closed_tables' @('src/rebuild/tables.py')
Invoke-Stage 'closed_figures' @('src/rebuild/closed_figures.py')
Invoke-Stage 'closed_report' @('src/rebuild/closed_report.py')
Invoke-Stage 'pdf_build' @('scripts/build_rebuild_pdf.py')
Invoke-Stage 'final_quality' @('scripts/final_quality.py')
Invoke-Stage 'closed_verify' @('src/rebuild/closed_verify.py')
Invoke-Stage 'package_final' @('scripts/package_final.py')
