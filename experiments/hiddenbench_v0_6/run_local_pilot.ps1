param([switch]$CheckHostLoadOnly)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $repoRoot

$serverExe = Join-Path $repoRoot '.cache/runtimes/llama.cpp/llama-server.exe'
$modelFile = Join-Path $repoRoot '.cache/models/Qwen3-14B-Q4_K_M.gguf'
$cpuSamples = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 2 -MaxSamples 3).CounterSamples |
    ForEach-Object { $_.CookedValue }
$baselineCpu = ($cpuSamples | Measure-Object -Average).Average
$gpuUtilText = (& nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$baselineGpu = [int]$gpuUtilText
Write-Output "Baseline host load: CPU $([math]::Round($baselineCpu, 1))%, GPU $baselineGpu%."
if ($baselineCpu -ge 40 -or $baselineGpu -ge 70) {
    throw "Host is already busy (CPU $([math]::Round($baselineCpu, 1))%, GPU $baselineGpu%). Wait until it is idle before starting this pilot."
}
if ($CheckHostLoadOnly) {
    Write-Output 'Idle gate passed; no model service was started.'
    return
}

$runStamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$logDir = Join-Path $repoRoot ".cache/pilot_hiddenbench_v0_6/server_$runStamp"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stdoutLog = Join-Path $logDir 'llama_server.stdout.log'
$stderrLog = Join-Path $logDir 'llama_server.stderr.log'
$resourceLog = Join-Path $logDir 'resource_monitor.jsonl'
$monitorJob = $null

$server = Start-Process -WindowStyle Hidden -PassThru `
    -FilePath $serverExe `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -ArgumentList @(
        '--model', $modelFile,
        '--alias', 'Qwen3-14B-Q4_K_M',
        '--host', '127.0.0.1', '--port', '8000',
        '--n-gpu-layers', '24', '--ctx-size', '8192', '--parallel', '1',
        '--threads', '4', '--threads-batch', '4',
        '--temp', '0.6', '--top-k', '20', '--top-p', '0.95', '--min-p', '0',
        '--presence-penalty', '1.5', '--seed', '20260928', '--n-predict', '1536',
        '--reasoning', 'on', '--reasoning-budget', '1024'
    )

try {
    $server.PriorityClass = 'BelowNormal'
    $monitorJob = Start-Job -FilePath (Join-Path $PSScriptRoot 'monitor_resources.ps1') `
        -ArgumentList $server.Id, $resourceLog
    $ready = $false
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        if ($server.HasExited) { throw "llama-server exited with code $($server.ExitCode). See $stderrLog" }
        try {
            $models = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/v1/models' -TimeoutSec 2
            if (@($models.data | ForEach-Object { $_.id }) -contains 'Qwen3-14B-Q4_K_M') {
                $ready = $true
                break
            }
        } catch { }
        Start-Sleep -Seconds 2
    }
    if (-not $ready) { throw "Local model did not become ready. See $stderrLog" }

    $env:PYTHONPATH = (Join-Path $repoRoot '.cache/research/HiddenBench_ICML/src')
    python experiments/hiddenbench_v0_6/run_protocol_pilot_v0_6.py --prepare-only
    if ($LASTEXITCODE -ne 0) { throw 'v0.6 prepare-only verification failed.' }
    python experiments/hiddenbench_v0_6/run_protocol_pilot_v0_6.py
    if ($LASTEXITCODE -ne 0) { throw "v0.6 runner exited with code $LASTEXITCODE." }
} finally {
    if ($monitorJob) {
        Stop-Job -Job $monitorJob -ErrorAction SilentlyContinue
        Remove-Job -Job $monitorJob -Force -ErrorAction SilentlyContinue
    }
    if (-not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
        Wait-Process -Id $server.Id -Timeout 10 -ErrorAction SilentlyContinue
    }
}
