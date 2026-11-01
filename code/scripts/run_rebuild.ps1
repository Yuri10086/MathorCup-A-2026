# Compatibility entry: Q1 is frozen; only the final three closed experiments run.
# No machine-specific interpreter path is stored here: when -Python is omitted, run_final.ps1
# resolves it itself (explicit argument -> PATH -> well-known conda locations), so this file stays
# portable across machines. Pass -Python explicitly only to override that resolution.
param([string]$Python='',[switch]$SkipExperiments)
& (Join-Path $PSScriptRoot 'run_final.ps1') -Python $Python -ReportOnly:$SkipExperiments
