param([switch]$PrepareOnly)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $repoRoot

$pythonExe = (Get-Command python -ErrorAction Stop).Source
$serverScript = Join-Path $repoRoot 'research/local_chat_server.py'
$modelDir = Join-Path $repoRoot '.cache/models/Qwen3-1.7B'
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$runDir = Join-Path $repoRoot ".cache/pilot_hiddenbench_v0_9/$stamp"
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$serverLog = Join-Path $runDir 'server.log'
$serverErrorLog = Join-Path $runDir 'server.stderr.log'
$runnerOut = Join-Path $runDir 'runner.stdout.log'
$runnerErr = Join-Path $runDir 'runner.stderr.log'
$usageLog = Join-Path $runDir 'usage.jsonl'
$resourceLog = Join-Path $runDir 'resources.jsonl'
$server = $null
$runner = $null
$cpuHighSamples = 0
$gpuHighSamples = 0
$priorServerCpu = 0.0
$priorSampleAt = Get-Date
$logicalProcessors = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors

function Get-ResourceSample {
    param([switch]$CheckStop)
    $sampleAt = Get-Date
    $cpu = [double](Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
    $gpuValues = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | Select-Object -First 1).Split(',') |
        ForEach-Object { [int]$_.Trim() }
    $process = if ($server) { Get-Process -Id $server.Id -ErrorAction SilentlyContinue } else { $null }
    $elapsed = ($sampleAt - $script:priorSampleAt).TotalSeconds
    $serverCores = if ($process -and $elapsed -gt 0) { ($process.CPU - $script:priorServerCpu) / $elapsed } else { 0.0 }
    $availableRamMiB = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024, 0)
    $row = [ordered]@{
        timestamp_local = $sampleAt.ToString('o')
        host_cpu_percent = [math]::Round($cpu, 2)
        server_cpu_cores = [math]::Round($serverCores, 3)
        server_working_set_mib = if ($process) { [math]::Round($process.WorkingSet64 / 1MB, 1) } else { 0 }
        gpu_util_percent = $gpuValues[0]
        gpu_memory_used_mib = $gpuValues[1]
        gpu_memory_total_mib = $gpuValues[2]
        available_system_memory_mib = $availableRamMiB
    }
    $row | ConvertTo-Json -Compress | Add-Content -LiteralPath $resourceLog -Encoding utf8
    if ($process) { $script:priorServerCpu = $process.CPU }
    $script:priorSampleAt = $sampleAt

    if ($CheckStop) {
        $script:cpuHighSamples = if ($cpu -ge 65) { $script:cpuHighSamples + 1 } else { 0 }
        $script:gpuHighSamples = if ($gpuValues[0] -ge 90) { $script:gpuHighSamples + 1 } else { 0 }
        if ($script:cpuHighSamples -ge 2 -or $script:gpuHighSamples -ge 2) {
            throw "Automatic resource stop: CPU $([math]::Round($cpu, 1))%, GPU $($gpuValues[0])%."
        }
    }
    return $row
}

try {
    if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
        throw 'Port 8000 is already in use; no process was changed.'
    }
    $baselineCpu = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 2 -MaxSamples 3).CounterSamples |
        ForEach-Object { $_.CookedValue }
    $meanCpu = ($baselineCpu | Measure-Object -Average).Average
    $gpu = (& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | Select-Object -First 1).Split(',') |
        ForEach-Object { [int]$_.Trim() }
    $freeRam = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024, 0)
    Write-Output "Idle gate: CPU $([math]::Round($meanCpu,1))%, GPU $($gpu[0])%, GPU memory $($gpu[1]) MiB, free system memory $freeRam MiB."
    if ($meanCpu -ge 25 -or $gpu[0] -ge 50 -or $gpu[1] -ge 2500 -or $freeRam -lt 6000) {
        throw 'Idle resource gate rejected launch. No model service was started.'
    }

    $env:PYTHONPATH = Join-Path $repoRoot '.cache/research/HiddenBench_ICML/src'
    $env:TLU_MODEL_PATH = $modelDir
    $env:TLU_MODEL_NAME = 'Qwen3-1.7B'
    $env:TLU_TORCH_THREADS = '2'
    $env:TLU_USAGE_LOG = $usageLog
    $env:CUDA_CACHE_PATH = Join-Path $repoRoot '.cache/cuda'
    $env:OMP_NUM_THREADS = '2'
    $env:MKL_NUM_THREADS = '2'
    $env:TOKENIZERS_PARALLELISM = 'false'
    $env:OPENAI_API_KEY = 'local-experiment'
    $env:PYTHONIOENCODING = 'utf-8'

    & $pythonExe experiments/hiddenbench_v0_9/run_capability_screen.py --artifacts-only
    if ($LASTEXITCODE -ne 0) { throw 'Pinned artifact preflight failed.' }
    if ($PrepareOnly) {
        Write-Output 'Prepare-only completed; no model service was started.'
        return
    }

    $server = Start-Process -WindowStyle Hidden -PassThru -FilePath $pythonExe `
        -WorkingDirectory $repoRoot -RedirectStandardOutput $serverLog -RedirectStandardError $serverErrorLog `
        -ArgumentList @('-m','uvicorn','local_chat_server:app','--app-dir','research','--host','127.0.0.1','--port','8000','--workers','1','--log-level','warning')
    $server.PriorityClass = 'BelowNormal'
    $ready = $false
    $readyClock = [System.Diagnostics.Stopwatch]::StartNew()
    while (-not $ready -and $readyClock.Elapsed.TotalSeconds -lt 120) {
        Start-Sleep -Seconds 5
        $server.Refresh()
        if ($server.HasExited) { throw "Local Transformers endpoint exited ($($server.ExitCode)); see $serverLog" }
        if (($readyClock.Elapsed.TotalSeconds -ge 15) -and ([int]$readyClock.Elapsed.TotalSeconds % 15 -lt 5)) {
            Get-ResourceSample -CheckStop | Out-Null
        }
        try {
            $models = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/v1/models' -TimeoutSec 2
            if (@($models.data | ForEach-Object { $_.id }) -contains 'Qwen3-1.7B') { $ready = $true }
        } catch { }
    }
    if (-not $ready) { throw 'Local model service did not become ready within 120 seconds.' }

    $runner = Start-Process -WindowStyle Hidden -PassThru -FilePath $pythonExe `
        -WorkingDirectory $repoRoot -RedirectStandardOutput $runnerOut -RedirectStandardError $runnerErr `
        -ArgumentList @('experiments/hiddenbench_v0_9/run_capability_screen.py')
    $runner.PriorityClass = 'BelowNormal'
    $runClock = [System.Diagnostics.Stopwatch]::StartNew()
    while ($true) {
        Start-Sleep -Seconds 15
        $runner.Refresh()
        if ($runner.HasExited) { break }
        $server.Refresh()
        if ($server.HasExited) { throw 'Local model service exited during voting.' }
        Get-ResourceSample -CheckStop | Out-Null
        if ($runClock.Elapsed.TotalSeconds -ge 600) { throw 'Runner exceeded the 600-second no-progress/overall time cap.' }
    }
    if ($runner.ExitCode -ne 0) { throw "Capability runner exited with code $($runner.ExitCode); see $runnerErr" }
    $runnerText = Get-Content -LiteralPath $runnerOut -Raw
    $jsonStart = $runnerText.IndexOf('{')
    if ($jsonStart -lt 0) { throw "Runner completed without a sanitized summary; see $runnerOut" }
    $summary = $runnerText.Substring($jsonStart) | ConvertFrom-Json
    Write-Output "Capability screen completed. Sanitized result and private raw records are under $($summary.raw_result_local_path)."
    Write-Output "Runtime telemetry and token-only request usage log: $runDir"
} finally {
    if ($runner) {
        $runner.Refresh()
        if (-not $runner.HasExited) { Stop-Process -Id $runner.Id -Force -ErrorAction SilentlyContinue }
        Wait-Process -Id $runner.Id -Timeout 10 -ErrorAction SilentlyContinue
    }
    if ($server) {
        $server.Refresh()
        if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue }
        Wait-Process -Id $server.Id -Timeout 10 -ErrorAction SilentlyContinue
    }
}
