$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $repoRoot

$serverExe = Join-Path $repoRoot '.cache/runtimes/llama.cpp/llama-server.exe'
$modelFile = Join-Path $repoRoot '.cache/models/Qwen3-14B-Q4_K_M.gguf'
$runStamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$logDir = Join-Path $repoRoot ".cache/pilot_hiddenbench_v0_6/server_$runStamp"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stdoutLog = Join-Path $logDir 'llama_server.stdout.log'
$stderrLog = Join-Path $logDir 'llama_server.stderr.log'

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
    if (-not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
        Wait-Process -Id $server.Id -Timeout 10 -ErrorAction SilentlyContinue
    }
}
