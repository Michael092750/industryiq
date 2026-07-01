#!/usr/bin/env pwsh
# Start the IndustryIQ local stack in Docker.
#
#   ./scripts/start.ps1
#
# Brings up Postgres (+ the Milvus infra) and the API. The app image is built
# with the `local` extra (fastembed) and reads ANTHROPIC_API_KEY and friends
# straight from your root .env via docker-compose.override.yml -- so there are no
# environment variables to type each time. Data lives in Docker named volumes and
# survives restarts.
#
# Stop with:            docker compose down          (keeps data)
# Stop and wipe data:   docker compose down -v       (deletes the DB volume!)

$ErrorActionPreference = 'Stop'

# Run from the repo root regardless of where the script is invoked from.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path (Join-Path $root '.env'))) {
    Write-Error "No .env found in $root. Copy .env.example to .env and fill it in first."
    exit 1
}

# A non-Docker process on port 8000 (e.g. a manually-run `uvicorn`) will stop the
# app container from binding. Catch that early with a clear message.
$busy = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($busy) {
    $proc = Get-Process -Id $busy.OwningProcess -ErrorAction SilentlyContinue
    if ($proc -and $proc.ProcessName -match 'python|uvicorn') {
        Write-Error "Port 8000 is held by $($proc.ProcessName) (PID $($proc.Id)) -- likely a host-run backend. Stop it, then re-run this script."
        exit 1
    }
}

Write-Host "Building and starting containers (first build downloads fastembed)..." -ForegroundColor Cyan
docker compose up -d --build
if ($LASTEXITCODE -ne 0) { Write-Error "docker compose up failed."; exit 1 }

Write-Host "Waiting for the API at http://localhost:8000 ..." -ForegroundColor Cyan
$ok = $false
foreach ($i in 1..60) {
    try {
        $r = Invoke-RestMethod -Uri 'http://localhost:8000/health' -TimeoutSec 3
        if ($r.status -eq 'ok') { $ok = $true; break }
    } catch { Start-Sleep -Seconds 2 }
}

if (-not $ok) {
    Write-Warning "API did not become healthy in time. Recent app logs:"
    docker logs --tail 40 industryiq_app
    exit 1
}

Write-Host ""
Write-Host "API is up:  http://localhost:8000" -ForegroundColor Green
Write-Host "Frontend reaches it via frontend/.env.local (VITE_API_URL=http://localhost:8000)."
Write-Host "Note: the first chat is slow while the embedding model downloads inside the container."
