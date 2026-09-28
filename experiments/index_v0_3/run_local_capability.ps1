param([switch]$PrepareOnly)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $repoRoot
$pythonExe = (Get-Command python -ErrorAction Stop).Source
$modelPath = Join-Path $repoRoot '.cache/models/Qwen3-1.7B'
$tasksPath = Join-Path $repoRoot '.cache/index_v0_2/tasks.jsonl'
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$runDir = Join-Path $repoRoot ".cache/index_v0_3/$stamp"
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$serverStdout = Join-Path $runDir 'server.stdout.log'
$serverStderr = Join-Path $runDir 'server.stderr.log'
$runnerStdout = Join-Path $runDir 'runner.stdout.log'
$runnerStderr = Join-Path $runDir 'runner.stderr.log'
$usageLog = Join-Path $runDir 'usage.jsonl'
$resourceLog = Join-Path $runDir 'resources.jsonl'
$server = $null
$runner = $null
$cpuHighSamples = 0
$gpuHighSamples = 0
$priorServerCpu = 0.0
$priorSampleAt = Get-Date

function Get-ResourceSample {
    param([switch]$CheckStop)
    $sampleAt = Get-Date
    $cpu = [double](Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
    $gpuFields = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | Select-Object -First 1).Split(',') |
        ForEach-Object { [int]$_.Trim() }
    $freeRamMiB = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024, 0)
    $process = if ($server) { Get-Process -Id $server.Id -ErrorAction SilentlyContinue } else { $null }
    $elapsed = ($sampleAt - $script:priorSampleAt).TotalSeconds
    $serverCores = if ($process -and $elapsed -gt 0) { ($process.CPU - $script:priorServerCpu) / $elapsed } else { 0.0 }
    $row = [ordered]@{
        timestamp_local = $sampleAt.ToString('o')
        host_cpu_percent = [math]::Round($cpu, 2)
        server_cpu_cores = [math]::Round($serverCores, 3)
        server_working_set_mib = if ($process) { [math]::Round($process.WorkingSet64 / 1MB, 1) } else { 0 }
        gpu_util_percent = $gpuFields[0]
        gpu_memory_used_mib = $gpuFields[1]
        gpu_memory_total_mib = $gpuFields[2]
        available_system_memory_mib = $freeRamMiB
    }
    $row | ConvertTo-Json -Compress | Add-Content -LiteralPath $resourceLog -Encoding utf8
    if ($process) { $script:priorServerCpu = $process.CPU }
    $script:priorSampleAt = $sampleAt

    if ($CheckStop) {
        $script:cpuHighSamples = if ($cpu -ge 45) { $script:cpuHighSamples + 1 } else { 0 }
        $script:gpuHighSamples = if ($gpuFields[0] -ge 85) { $script:gpuHighSamples + 1 } else { 0 }
        if ($script:cpuHighSamples -ge 2 -or $script:gpuHighSamples -ge 2 -or $freeRamMiB -lt 4000) {
            throw "Automatic resource stop: CPU $([math]::Round($cpu,1))%, GPU $($gpuFields[0])%, free RAM $freeRamMiB MiB."
        }
    }
    return $row
}

try {
    if (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue) {
        throw 'Port 8001 is already in use; no process was changed.'
    }

    $cpuSamples = @()
    for ($i = 0; $i -lt 3; $i++) {
        $cpuSamples += [double](Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
        if ($i -lt 2) { Start-Sleep -Seconds 2 }
    }
    $meanCpu = ($cpuSamples | Measure-Object -Average).Average
    $maxCpu = ($cpuSamples | Measure-Object -Maximum).Maximum
    $gpuFields = (& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | Select-Object -First 1).Split(',') |
        ForEach-Object { [int]$_.Trim() }
    $freeRamMiB = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024, 0)
    Write-Output "Idle gate: CPU mean $([math]::Round($meanCpu,1))% / max $([math]::Round($maxCpu,1))%, GPU $($gpuFields[0])%, GPU memory $($gpuFields[1]) MiB, free system memory $freeRamMiB MiB."
    if ($meanCpu -ge 20 -or $maxCpu -ge 30 -or $gpuFields[0] -ge 25 -or $gpuFields[1] -ge 1800 -or $freeRamMiB -lt 6000) {
        throw 'Idle resource gate rejected this attempt. No model artifact was hashed and no service was started.'
    }

    $env:TLU_MODEL_PATH = $modelPath
    $env:TLU_MODEL_NAME = 'Qwen3-1.7B'
    $env:TLU_TORCH_THREADS = '1'
    $env:TLU_USAGE_LOG = $usageLog
    $env:CUDA_CACHE_PATH = Join-Path $repoRoot '.cache/cuda'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:TOKENIZERS_PARALLELISM = 'false'
    $env:OPENAI_API_KEY = 'local-experiment'
    $env:PYTHONIOENCODING = 'utf-8'

    & $pythonExe experiments/index_v0_3/run_capability_pilot.py --tasks $tasksPath --output-dir $runDir --artifacts-only
    if ($LASTEXITCODE -ne 0) { throw 'Pinned local task/model/tokenizer preflight failed.' }
    if ($PrepareOnly) {
        Write-Output 'Prepare-only complete; no model service was started or model weights loaded.'
        return
    }

    $server = Start-Process -WindowStyle Hidden -PassThru -FilePath $pythonExe `
        -WorkingDirectory $repoRoot -RedirectStandardOutput $serverStdout -RedirectStandardError $serverStderr `
        -ArgumentList @('-m','uvicorn','local_chat_server:app','--app-dir','research','--host','127.0.0.1','--port','8001','--workers','1','--log-level','warning')
    $server.PriorityClass = 'BelowNormal'
    $ready = $false
    $readyClock = [System.Diagnostics.Stopwatch]::StartNew()
    while (-not $ready -and $readyClock.Elapsed.TotalSeconds -lt 120) {
        Start-Sleep -Seconds 4
        $server.Refresh()
        if ($server.HasExited) { throw "Local model service exited ($($server.ExitCode)); see $serverStderr" }
        if ($readyClock.Elapsed.TotalSeconds -ge 10 -and ([int]$readyClock.Elapsed.TotalSeconds % 10 -lt 4)) {
            Get-ResourceSample -CheckStop | Out-Null
        }
        try {
            $models = Invoke-RestMethod -Uri 'http://127.0.0.1:8001/v1/models' -TimeoutSec 2
            if (@($models.data | ForEach-Object { $_.id }) -contains 'Qwen3-1.7B') { $ready = $true }
        } catch { }
    }
    if (-not $ready) { throw 'Model endpoint did not become ready within 120 seconds.' }

    $runner = Start-Process -WindowStyle Hidden -PassThru -FilePath $pythonExe `
        -WorkingDirectory $repoRoot -RedirectStandardOutput $runnerStdout -RedirectStandardError $runnerStderr `
        -ArgumentList @('experiments/index_v0_3/run_capability_pilot.py','--tasks',$tasksPath,'--output-dir',$runDir)
    $runner.PriorityClass = 'BelowNormal'
    $runClock = [System.Diagnostics.Stopwatch]::StartNew()
    while ($true) {
        Start-Sleep -Seconds 10
        $runner.Refresh()
        if ($runner.HasExited) { break }
        $server.Refresh()
        if ($server.HasExited) { throw 'Local model service exited during the INDEX_m pilot.' }
        Get-ResourceSample -CheckStop | Out-Null
        if ($runClock.Elapsed.TotalSeconds -ge 300) { throw 'INDEX_m pilot exceeded its 300-second cap.' }
    }
    if ($runner.ExitCode -ne 0) { throw "INDEX_m runner exited with code $($runner.ExitCode); see $runnerStderr" }
    $runnerText = Get-Content -LiteralPath $runnerStdout -Raw
    $jsonStart = $runnerText.IndexOf('{')
    if ($jsonStart -lt 0) { throw "Runner completed without a sanitized summary; see $runnerStdout" }
    $summary = $runnerText.Substring($jsonStart) | ConvertFrom-Json
    Write-Output "INDEX_m pilot completed: $($summary.status), $($summary.model_requests) model requests. Private raw records and telemetry are under $runDir."
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
