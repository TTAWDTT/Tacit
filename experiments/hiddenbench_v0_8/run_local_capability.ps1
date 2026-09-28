param([switch]$PrepareOnly)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $repoRoot

$serverExe = Join-Path $repoRoot '.cache/runtimes/llama.cpp/llama-server.exe'
$modelFile = Join-Path $repoRoot '.cache/models/Qwen3-8B-Q4_K_M.gguf'
$pythonExe = (Get-Command python -ErrorAction Stop).Source
$runStamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$runDir = Join-Path $repoRoot ".cache/pilot_hiddenbench_v0_8/$runStamp"
$logDir = Join-Path $repoRoot ".cache/pilot_hiddenbench_v0_8/server_$runStamp"
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stdoutLog = Join-Path $logDir 'llama_server.stdout.log'
$stderrLog = Join-Path $logDir 'llama_server.stderr.log'
$resourceLog = Join-Path $logDir 'resource_monitor.jsonl'
$runnerOutLog = Join-Path $logDir 'runner.stdout.log'
$runnerErrLog = Join-Path $logDir 'runner.stderr.log'
$server = $null
$runner = $null

try {
    $baselineCpuSamples = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 2 -MaxSamples 3).CounterSamples |
        ForEach-Object { $_.CookedValue }
    $baselineCpu = ($baselineCpuSamples | Measure-Object -Average).Average
    $baselineGpu = [int]((& nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | Select-Object -First 1).Trim())
    Write-Output "Baseline load: CPU $([math]::Round($baselineCpu, 1))%, GPU $baselineGpu%."
    if ($baselineCpu -ge 35 -or $baselineGpu -ge 60) {
        throw "Resource gate rejected launch at CPU $([math]::Round($baselineCpu, 1))%, GPU $baselineGpu%. No model was started."
    }
    if ($PrepareOnly) {
        $env:PYTHONPATH = Join-Path $repoRoot '.cache/research/HiddenBench_ICML/src'
        python experiments/hiddenbench_v0_8/run_capability_screen.py --artifacts-only
        if ($LASTEXITCODE -ne 0) { throw 'Pinned artifact preflight failed.' }
        Write-Output 'Prepare-only completed; no model service was started.'
        return
    }

    $server = Start-Process -WindowStyle Hidden -PassThru `
        -FilePath $serverExe -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog `
        -ArgumentList @(
            '--model', $modelFile, '--alias', 'Qwen3-8B-Q4_K_M',
            '--host', '127.0.0.1', '--port', '8000',
            '--n-gpu-layers', '20', '--ctx-size', '8192', '--parallel', '1',
            '--threads', '4', '--threads-batch', '4',
            '--temp', '0.6', '--top-k', '20', '--top-p', '0.95', '--min-p', '0',
            '--presence-penalty', '1.5', '--seed', '20260928', '--n-predict', '1536',
            '--reasoning', 'on', '--reasoning-budget', '1024'
        )
    $server.PriorityClass = 'BelowNormal'

    $ready = $false
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        $server.Refresh()
        if ($server.HasExited) { throw "llama-server exited ($($server.ExitCode)); see $stderrLog" }
        try {
            $models = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/v1/models' -TimeoutSec 2
            if (@($models.data | ForEach-Object { $_.id }) -contains 'Qwen3-8B-Q4_K_M') { $ready = $true; break }
        } catch { }
        Start-Sleep -Seconds 2
    }
    if (-not $ready) { throw "Qwen3-8B service did not become ready; see $stderrLog" }

    $env:PYTHONPATH = Join-Path $repoRoot '.cache/research/HiddenBench_ICML/src'
    $env:OPENAI_API_KEY = 'local-experiment'
    $runner = Start-Process -WindowStyle Hidden -PassThru -FilePath $pythonExe `
        -WorkingDirectory $repoRoot -RedirectStandardOutput $runnerOutLog -RedirectStandardError $runnerErrLog `
        -ArgumentList @('experiments/hiddenbench_v0_8/run_capability_screen.py')
    $cpuHighSamples = 0
    $gpuHighSamples = 0
    $logicalProcessors = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
    $priorServerCpu = (Get-Process -Id $server.Id).CPU
    $priorSampleAt = Get-Date
    while ($true) {
        Start-Sleep -Seconds 15
        $runner.Refresh()
        if ($runner.HasExited) { break }
        $sampleAt = Get-Date
        $process = Get-Process -Id $server.Id -ErrorAction SilentlyContinue
        if (-not $process) { throw 'Model server exited while capability votes were running.' }
        $elapsed = ($sampleAt - $priorSampleAt).TotalSeconds
        $serverCores = if ($elapsed -gt 0) { ($process.CPU - $priorServerCpu) / $elapsed } else { 0 }
        $hostCpu = [double](Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
        $gpu = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | Select-Object -First 1).Split(',') |
            ForEach-Object { [int]$_.Trim() }
        $gpuUtil = $gpu[0]
        $gpuMemoryUsed = $gpu[1]
        $gpuMemoryTotal = $gpu[2]
        $cpuHighSamples = if ($hostCpu -ge 75) { $cpuHighSamples + 1 } else { 0 }
        $gpuHighSamples = if ($gpuUtil -ge 90) { $gpuHighSamples + 1 } else { 0 }
        [ordered]@{
            timestamp_local = $sampleAt.ToString('o')
            host_cpu_percent = [math]::Round($hostCpu, 2)
            server_cpu_cores = [math]::Round($serverCores, 3)
            server_working_set_mib = [math]::Round($process.WorkingSet64 / 1MB, 1)
            gpu_util_percent = $gpuUtil
            gpu_memory_used_mib = $gpuMemoryUsed
            gpu_memory_total_mib = $gpuMemoryTotal
        } | ConvertTo-Json -Compress | Add-Content -LiteralPath $resourceLog -Encoding utf8
        $priorServerCpu = $process.CPU
        $priorSampleAt = $sampleAt
        if ($cpuHighSamples -ge 3 -or $gpuHighSamples -ge 2) {
            Set-Content -LiteralPath (Join-Path $runDir 'STOP_REASON.txt') -Value "Automatic resource stop at $($sampleAt.ToString('o')); host CPU=$([math]::Round($hostCpu,1))%, GPU=$gpuUtil%." -Encoding utf8
            Stop-Process -Id $runner.Id -Force -ErrorAction SilentlyContinue
            throw 'Automatic resource stop triggered; the partial condition is incomplete and must not be scored.'
        }
    }
    if ($runner.ExitCode -ne 0) { throw "Capability runner exited with code $($runner.ExitCode); see $runnerErrLog" }
    Write-Output "Capability screen completed. Local logs: $logDir"
    Write-Output "The sanitized aggregate and private raw output are under $runDir."
} finally {
    if ($runner) {
        $runner.Refresh()
        if (-not $runner.HasExited) { Stop-Process -Id $runner.Id -Force -ErrorAction SilentlyContinue }
    }
    if ($server) {
        $server.Refresh()
        if (-not $server.HasExited) {
            Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
            Wait-Process -Id $server.Id -Timeout 10 -ErrorAction SilentlyContinue
        }
    }
}
