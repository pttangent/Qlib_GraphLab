param(
    [Parameter(Mandatory = $true)] [string]$RunRoot,
    [Parameter(Mandatory = $true)] [string]$BenchmarkRoot,
    [Parameter(Mandatory = $true)] [string]$RepoRoot,
    [string]$Branch = "agent/nff-research-atomic-fastpath-v2-7",
    [int]$IntervalSeconds = 300
)

$ErrorActionPreference = "Stop"
$run = [System.IO.Path]::GetFullPath($RunRoot)
$repo = [System.IO.Path]::GetFullPath($RepoRoot)
$log = Join-Path $run "watchdog.log"
$statePath = Join-Path $run "watchdog_state.json"
$statusPath = Join-Path $run "status.json"

function Write-WatchdogLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $log -Value $line -Encoding utf8
}

function Commit-ResourceSnapshot {
    & C:\Python314\python.exe (Join-Path $repo "tools/collect_v2_7_resource_snapshot.py") `
        --run-root $run --benchmark-root $BenchmarkRoot | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "resource snapshot collector failed: $LASTEXITCODE" }
    $snapshot = Join-Path $repo ("reports/resource_snapshots/" + (Split-Path $run -Leaf))
    & git -C $repo add -- $snapshot
    if ($LASTEXITCODE -ne 0) { throw "git add failed: $LASTEXITCODE" }
    & git -C $repo diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-WatchdogLog "no report changes to commit"
        return
    }
    & git -C $repo commit -m "Archive terminal v2.7 resource snapshot"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed: $LASTEXITCODE" }
    & git -C $repo push origin $Branch
    if ($LASTEXITCODE -ne 0) { throw "git push failed: $LASTEXITCODE" }
    Write-WatchdogLog "terminal resource snapshot committed and pushed"
}

Write-WatchdogLog "watchdog started run=$run interval_seconds=$IntervalSeconds"
while ($true) {
    try {
        if (-not (Test-Path -LiteralPath $statusPath)) {
            Write-WatchdogLog "status.json missing"
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }
        $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        $processAlive = [bool](Get-Process -Id ([int]$status.pid) -ErrorAction SilentlyContinue)
        $heartbeatAge = ((Get-Date) - (Get-Item -LiteralPath $statusPath).LastWriteTime).TotalMinutes
        $resources = $status.resources
        $state = @{
            observed_utc = (Get-Date).ToUniversalTime().ToString("o")
            status = $status.status
            stage = $status.stage
            completed_units = $status.completed_units
            total_units = $status.total_units
            running_workers = $status.running_workers
            pending_units = $status.pending_units
            failure_count = $status.failure_count
            process_alive = $processAlive
            status_file_age_minutes = [math]::Round($heartbeatAge, 3)
            memory_percent = $resources.memory_percent
            memory_available_gb = $resources.memory_available_gb
            worker_rss_total_gb = $resources.worker_rss_total_gb
            disk_free_gb = $resources.disk_free_gb
        }
        $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statePath -Encoding utf8
        Write-WatchdogLog ("state completed={0}/{1} running={2} pending={3} failure={4} mem={5} avail={6} rss={7} alive={8}" -f `
            $status.completed_units, $status.total_units, $status.running_workers, $status.pending_units,
            $status.failure_count, $resources.memory_percent, $resources.memory_available_gb,
            $resources.worker_rss_total_gb, $processAlive)

        if ($status.status -in @("complete", "partial_success", "blocked_disk_free_floor", "failed")) {
            Commit-ResourceSnapshot
            Write-WatchdogLog "watchdog terminal status=$($status.status)"
            exit 0
        }
        if (-not $processAlive -and $status.status -eq "running") {
            Write-WatchdogLog "alert parent process disappeared while status is running"
            exit 2
        }
    } catch {
        Write-WatchdogLog "watchdog error: $($_.Exception.Message)"
    }
    Start-Sleep -Seconds $IntervalSeconds
}
