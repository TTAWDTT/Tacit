param(
    [string]$Output = '.cache/emergent_ood_v0_3/resource_preflight.json',
    [int[]]$Ports = @(8000)
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$target = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Output))
$rootPrefix = $repoRoot.TrimEnd('\') + '\'
if (-not $target.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Output must stay inside the project directory.'
}
$relativeParts = $target.Substring($rootPrefix.Length).Split([char[]]@('\', '/'), [System.StringSplitOptions]::RemoveEmptyEntries)
$walk = $repoRoot
foreach ($part in $relativeParts) {
    $walk = Join-Path $walk $part
    if (Test-Path -LiteralPath $walk) {
        $item = Get-Item -Force -LiteralPath $walk
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Output path must not traverse a symbolic link or other reparse point.'
        }
    }
}
if (-not $Ports -or @($Ports | Where-Object { $_ -lt 1 -or $_ -gt 65535 }).Count -gt 0) {
    throw 'Ports must contain valid TCP port numbers.'
}
$Ports = @($Ports | Sort-Object -Unique)

$cpuSamples = @()
for ($i = 0; $i -lt 3; $i++) {
    $cpuSamples += [double](Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
    if ($i -lt 2) { Start-Sleep -Seconds 2 }
}
$gpuLines = @(& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
if ($LASTEXITCODE -ne 0 -or $gpuLines.Count -eq 0) { throw 'Could not read NVIDIA GPU utilization and memory.' }
$gpuFields = $gpuLines[0].Split(',') | ForEach-Object { [int]$_.Trim() }
$freeRamMiB = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024, 0)
$listeningPorts = @(
    Get-NetTCPConnection -State Listen -ErrorAction Stop |
        Where-Object { $Ports -contains $_.LocalPort } |
        Select-Object -ExpandProperty LocalPort -Unique
)
$meanCpu = ($cpuSamples | Measure-Object -Average).Average
$maxCpu = ($cpuSamples | Measure-Object -Maximum).Maximum
$limits = [ordered]@{
    host_cpu_mean_below_percent = 20
    host_cpu_each_sample_below_percent = 30
    gpu_utilization_below_percent = 25
    gpu_memory_below_mib = 1800
    free_system_memory_at_least_mib = 6000
}
$reasons = @()
if ($meanCpu -ge $limits.host_cpu_mean_below_percent) { $reasons += 'Host CPU mean is at or above the frozen limit.' }
if ($maxCpu -ge $limits.host_cpu_each_sample_below_percent) { $reasons += 'At least one host CPU sample is at or above the frozen limit.' }
if ($gpuFields[0] -ge $limits.gpu_utilization_below_percent) { $reasons += 'GPU utilization is at or above the frozen limit.' }
if ($gpuFields[1] -ge $limits.gpu_memory_below_mib) { $reasons += 'GPU memory use is at or above the frozen limit.' }
if ($freeRamMiB -lt $limits.free_system_memory_at_least_mib) { $reasons += 'Free system memory is below the frozen minimum.' }
if ($listeningPorts.Count -gt 0) { $reasons += 'One or more requested endpoint ports are already in use.' }

$report = [ordered]@{
    schema = 'tlu.local_resource_preflight.v1'
    status = if ($reasons.Count -eq 0) { 'eligible' } else { 'rejected' }
    sampled_at_utc = [DateTime]::UtcNow.ToString('o')
    machine_name = $env:COMPUTERNAME
    requested_ports = $Ports
    listening_requested_ports = $listeningPorts
    limits = $limits
    observed = [ordered]@{
        host_cpu_samples_percent = @($cpuSamples | ForEach-Object { [math]::Round($_, 2) })
        host_cpu_mean_percent = [math]::Round($meanCpu, 2)
        host_cpu_max_percent = [math]::Round($maxCpu, 2)
        gpu_utilization_percent = $gpuFields[0]
        gpu_memory_used_mib = $gpuFields[1]
        gpu_memory_total_mib = $gpuFields[2]
        free_system_memory_mib = $freeRamMiB
    }
    rejection_reasons = $reasons
    model_artifact_hashed = $false
    model_artifact_read = $false
    model_loaded = $false
    service_started = $false
    inference_requests = 0
}

$parent = Split-Path -Parent $target
New-Item -ItemType Directory -Path $parent -Force | Out-Null
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $target -Encoding utf8
Write-Output "Resource preflight $($report.status): $target"
if ($reasons.Count -gt 0) {
    $reasons | ForEach-Object { Write-Output "- $_" }
    exit 2
}
