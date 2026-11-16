# Runs the frozen pipeline end to end into a fresh folder and stops at the first failing step.
# Usage, from the project root with the sip environment active:
#   .\scripts\run_pipeline.ps1 -RunDir data\runs\check -RandomScenes 10
param(
    [string]$RunDir = "data\runs\check",
    [int]$RandomScenes = 10,
    [string]$Dataset = "data\raw\tless"
)
$ErrorActionPreference = "Stop"
$results = Join-Path $RunDir "results"
New-Item -ItemType Directory -Force -Path $RunDir, $results | Out-Null
$log = Join-Path $RunDir "run.log"
"run started $(Get-Date -Format s)" | Out-File -FilePath $log -Encoding utf8

function Step([string]$name, [scriptblock]$cmd) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    $t0 = Get-Date
    & $cmd | ForEach-Object { $_; $_ | Out-File -FilePath $log -Append -Encoding utf8 }
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $name (exit code $LASTEXITCODE). See $log" }
    $msg = "{0}: {1:N0} s" -f $name, ((Get-Date) - $t0).TotalSeconds
    Write-Host $msg -ForegroundColor Green
    $msg | Out-File -FilePath $log -Append -Encoding utf8
}

$gen = Join-Path $RunDir "gen_v3"
Step "unit tests" { python -m pytest -q }
Step "replica jobs, scene 1" { python scripts/make_replica_job.py $Dataset --all --out-root $RunDir }
Step "render replicas" { blenderproc run scripts/render_jobs.py $RunDir "replica_s01_im*" }
Step "replica statistics" { python scripts/scene_replica_stats.py $RunDir "replica_s01_im*" $Dataset --out $results --samples-out $RunDir }
Step "residual analysis" { python scripts/analyze_replica_residuals.py (Join-Path $RunDir "scene01_replica_pixels.npz") }
Step "support planes (RANSAC)" { python scripts/fit_support_planes.py $Dataset --out $results }
Step "consensus plane" { python scripts/consensus_plane.py $Dataset (Join-Path $results "scene01_support_planes.json") }
Step "random scene jobs" { python scripts/make_random_jobs.py configs/generation_v3.yaml --count $RandomScenes --out-root $gen }
Step "render random scenes" { blenderproc run scripts/render_random_scenes.py $gen "scene_*" }
Step "validate random scenes" { python scripts/validate_scenes.py $gen "scene_*" }
Step "brightness vs real" { python scripts/compare_brightness.py $Dataset $gen "scene_*" }
Step "compare with committed results" { python scripts/check_reproduction.py results $results }
"run finished $(Get-Date -Format s)" | Out-File -FilePath $log -Append -Encoding utf8
Write-Host "`nAll steps passed. Log: $log" -ForegroundColor Green