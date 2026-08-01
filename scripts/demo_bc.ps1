<#
.SYNOPSIS
    Option B -> Option C multi-agent demo: "kill a worker, watch C resume".

.DESCRIPTION
    Drives the three demo beats against a running IndustryIQ API:
      1. Option B (in-process), happy path -> a clean fan-out answer.
      2. Option B breaks -> a node crashes and the whole run is LOST (no recovery).
      3. Option C (distributed) -> the same crash is survived: a flaky worker fails
         a node, a healthy worker reclaims it, and the run completes. The run
         ledger (/debug/runs/{id}) shows the reclaim.

.PREREQUISITES
    - The API is running (e.g. uvicorn industryiq.api.app:app) at -BaseUrl.
    - REDIS_URL is set (Option C needs Redis) and RAG_PROVIDER=anthropic with
      ANTHROPIC_API_KEY, in the .env the app + workers load.
    - DEBUG_API_KEY is set (so /debug/runs/{id} is enabled) and passed as -DebugKey.

.EXAMPLE
    ./scripts/demo_bc.ps1 -DebugKey $env:DEBUG_API_KEY -Question "Compare the AI and semiconductor markets"
#>

param(
    [string]$BaseUrl = "http://localhost:8000",
    [string]$DebugKey = $env:DEBUG_API_KEY,
    [string]$Email = "demo@industryiq.local",
    [string]$Password = "demo-password-123",
    [string]$Question = "Compare the AI and semiconductor industries by market size and growth.",
    [string]$Python = ".venv/Scripts/python.exe"
)

$ErrorActionPreference = "Stop"

function Pause-Beat($label) {
    Write-Host ""
    Read-Host ">> $label (press Enter)" | Out-Null
}

function Get-Token {
    $body = @{ email = $Email; password = $Password } | ConvertTo-Json
    try {
        $r = Invoke-RestMethod -Method Post -Uri "$BaseUrl/auth/register" -ContentType "application/json" -Body $body
    } catch {
        $r = Invoke-RestMethod -Method Post -Uri "$BaseUrl/auth/login" -ContentType "application/json" -Body $body
    }
    return $r.access_token
}

function Invoke-AgentRun($token, $executor, $injectFailure) {
    $headers = @{ Authorization = "Bearer $token" }
    $body = @{ question = $Question; executor = $executor; inject_failure = $injectFailure } | ConvertTo-Json
    return Invoke-RestMethod -Method Post -Uri "$BaseUrl/agents/run" -Headers $headers -ContentType "application/json" -Body $body
}

$token = Get-Token
Write-Host "Authenticated as $Email" -ForegroundColor Green

# --- Beat 1: Option B, happy path ------------------------------------------------
Pause-Beat "Beat 1 - Option B (in-process), happy path"
$b1 = Invoke-AgentRun $token "local" $false
Write-Host "completed nodes: $($b1.completed -join ', ')" -ForegroundColor Cyan
Write-Host $b1.answer

# --- Beat 2: Option B breaks -----------------------------------------------------
Pause-Beat "Beat 2 - Option B with a mid-run crash (no recovery)"
$b2 = Invoke-AgentRun $token "local" $true
Write-Host "failed nodes: $($b2.failed -join ', ')" -ForegroundColor Red
Write-Host $b2.answer -ForegroundColor Red

# --- Beat 3: Option C survives the crash -----------------------------------------
Pause-Beat "Beat 3 - Option C (distributed): starting 1 healthy + 1 flaky worker"
$healthy = Start-Process -FilePath $Python -ArgumentList "scripts/run_worker.py","w-healthy" -PassThru
$env:AGENT_FAILURE_MODE = "crash_once"
$flaky = Start-Process -FilePath $Python -ArgumentList "scripts/run_worker.py","w-flaky" -PassThru
Remove-Item Env:\AGENT_FAILURE_MODE
Write-Host "workers started (pids $($healthy.Id), $($flaky.Id)); giving them a moment..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

try {
    $c = Invoke-AgentRun $token "distributed" $false
    Write-Host "completed nodes: $($c.completed -join ', ')  failed: $($c.failed -join ', ')" -ForegroundColor Green
    Write-Host $c.answer

    if ($DebugKey) {
        Pause-Beat "Show the run ledger (the reclaim is in here)"
        $ledger = Invoke-RestMethod -Method Get -Uri "$BaseUrl/debug/runs/$($c.run_id)" -Headers @{ "X-Debug-Key" = $DebugKey }
        $ledger.events | ForEach-Object { "{0,-18} {1}" -f $_.event, ($_ | ConvertTo-Json -Compress) }
    }
} finally {
    Write-Host "stopping workers..." -ForegroundColor Yellow
    Stop-Process -Id $healthy.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $flaky.Id -ErrorAction SilentlyContinue
}

Write-Host "`nDemo complete: B lost the run on a crash; C reclaimed the task and finished." -ForegroundColor Green
