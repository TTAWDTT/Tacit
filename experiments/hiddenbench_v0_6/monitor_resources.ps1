param(
    [Parameter(Mandatory = $true)][int]$TargetPid,
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [int]$IntervalSeconds = 15
)

$logicalProcessorCount = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$previous = Get-Process -Id $TargetPid -ErrorAction Stop
$previousCpuSeconds = $previous.CPU
$previousAt = Get-Date

while ($true) {
    Start-Sleep -Seconds $IntervalSeconds
    $sampleAt = Get-Date
    $process = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    if (-not $process) { break }
    $elapsed = ($sampleAt - $previousAt).TotalSeconds
    $cpuCores = if ($elapsed -gt 0) { ($process.CPU - $previousCpuSeconds) / $elapsed } else { 0 }
    $systemCpu = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
    $gpuText = & nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | Select-Object -First 1
    $gpuUtil = [int]$gpuText.Trim()
    [ordered]@{
        timestamp_local = $sampleAt.ToString('o')
        server_cpu_cores = [math]::Round($cpuCores, 3)
        server_cpu_percent_system = [math]::Round(100 * $cpuCores / $logicalProcessorCount, 2)
        server_working_set_mib = [math]::Round($process.WorkingSet64 / 1MB, 1)
        host_cpu_percent = [math]::Round($systemCpu, 2)
        gpu_util_percent = $gpuUtil
    } | ConvertTo-Json -Compress | Add-Content -LiteralPath $OutputPath -Encoding utf8
    $previousCpuSeconds = $process.CPU
    $previousAt = $sampleAt
}
