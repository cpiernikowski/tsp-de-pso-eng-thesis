param (
    [Parameter(Mandatory = $true)]
    [string]$Program,

    [Parameter(Mandatory = $false)]
    [string[]]$Arguments = @()
)

$cpuSamples = @()
$ramSamples = @()

$sampleIntervalMs = 20

$time = Measure-Command {

    $proc = Start-Process `
        -FilePath $Program `
        -ArgumentList $Arguments `
        -NoNewWindow `
        -PassThru

    $prevCpuTime = [TimeSpan]::Zero
    $prevSampleTime = Get-Date

    while (-not $proc.HasExited) {
        try {
            $p = Get-Process -Id $proc.Id -ErrorAction Stop
            $now = Get-Date

            $cpuTimeDelta = $p.TotalProcessorTime - $prevCpuTime
            $timeDelta = ($now - $prevSampleTime).TotalSeconds

            if ($timeDelta -gt 0) {
                $cpuPercent = $cpuPercent = ($cpuTimeDelta.TotalSeconds / $timeDelta) / [Environment]::ProcessorCount * 100
                $cpuSamples += $cpuPercent
            }

            $prevCpuTime = $p.TotalProcessorTime
            $prevSampleTime = $now

            $ramSamples += ($p.WorkingSet64 / 1MB)

            Start-Sleep -Milliseconds $sampleIntervalMs
        }
        catch {
            break
        }
    }
}

[PSCustomObject]@{
    Program      = $Program
    Arguments    = ($Arguments -join ' ')
    TimeSeconds  = [math]::Round($time.TotalSeconds, 3)

    CPU_Avg_pct  = [math]::Round(($cpuSamples | Measure-Object -Average).Average, 2)
    CPU_Max_pct  = [math]::Round(($cpuSamples | Measure-Object -Maximum).Maximum, 2)

    RAM_Avg_MB   = [math]::Round(($ramSamples | Measure-Object -Average).Average, 2)
    RAM_Max_MB   = [math]::Round(($ramSamples | Measure-Object -Maximum).Maximum, 2)

    Samples      = $cpuSamples.Count
}