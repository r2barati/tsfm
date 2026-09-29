$ErrorActionPreference = 'Continue'
$project = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $project 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$today = Get-Date -Format 'yyyy-MM-dd'
$log = Join-Path $logDir ("task_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
Start-Transcript -LiteralPath $log -Force | Out-Null
try {
    $cutoffFile = Join-Path $project 'schedule\cutoffs.json'
    if (-not (Test-Path -LiteralPath $cutoffFile)) { throw 'Frozen cutoff list is missing.' }
    $cutoffs = (Get-Content -LiteralPath $cutoffFile -Raw | ConvertFrom-Json).cutoffs
    if ($today -notin $cutoffs) {
        Write-Error "Task Scheduler catch-up arrived on $today, which is not a scheduled cutoff. No forecast was created; the cutoff remains missing."
        exit 2
    }
    $expected = [datetime]::ParseExact("$today 09:00", 'yyyy-MM-dd HH:mm', [Globalization.CultureInfo]::InvariantCulture)
    $delay = (Get-Date) - $expected
    if ($delay.TotalMinutes -lt -10 -or $delay.TotalMinutes -gt 10) {
        Write-Error "Scheduled cutoff was 09:00 local time, but this invocation arrived at $(Get-Date -Format 'HH:mm:ss'). No backfilled forecast was created; the cutoff remains missing."
        exit 2
    }
    $uv = (Get-Command uv.exe -ErrorAction SilentlyContinue).Source
    if (-not $uv) { $uv = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\Scripts\uv.exe' }
    if (-not (Test-Path -LiteralPath $uv)) { throw "uv.exe not found at '$uv'." }
    $env:HF_HOME = Join-Path $project '.cache\huggingface'
    Push-Location $project
    & $uv run --project $project --locked tsfm-prospective run --confirmatory
    if ($LASTEXITCODE -ne 0) { throw "Forecast command failed with exit code $LASTEXITCODE." }
    & $uv run --project $project --locked tsfm-prospective score
    if ($LASTEXITCODE -ne 0) { throw "Scoring command failed with exit code $LASTEXITCODE." }
    & $uv run --project $project --locked tsfm-prospective report
    if ($LASTEXITCODE -ne 0) { throw "Report command failed with exit code $LASTEXITCODE." }
    Pop-Location
    Write-Output "Completed scheduled cutoff $today."
} catch {
    Write-Error $_
    if ($?) { $global:LASTEXITCODE = 1 }
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
