$ErrorActionPreference = "Continue"
$root = "C:\Work\Answer-Aware-parser"
Set-Location $root
$outDir = Join-Path $root "outputs\qspace\overnight_runs"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $outDir ("run_" + $ts + ".log")
"[OVERNIGHT-RUN] started=$ts" | Out-File -FilePath $log -Encoding utf8
"[OVERNIGHT-RUN] cwd=$root" | Out-File -FilePath $log -Encoding utf8 -Append
python -m pytest -q *>&1 | Out-File -FilePath $log -Encoding utf8 -Append
python run_qspace_pipeline.py *>&1 | Out-File -FilePath $log -Encoding utf8 -Append
"[OVERNIGHT-RUN] completed=" + (Get-Date -Format "yyyyMMdd_HHmmss") | Out-File -FilePath $log -Encoding utf8 -Append
