#!/usr/bin/env pwsh
<#
.SYNOPSIS
  One-command app deploy to the IndustryIQ EC2 box (Procedure B, automated).

.DESCRIPTION
  Does the whole app deploy with nothing to type each run:
    1. Auto-detects the running instance's public IP (or use -Ip).
    2. Ensures the SSH key locally (re-fetches it from the stack if missing).
    3. Resolves the three server secrets without prompting: an env var if set,
       else the value already on the server (so logins stay valid), else a fresh
       random one. Force new ones with -RegenerateSecrets.
    4. Bundles committed code (git archive HEAD), uploads it, writes .env, and
       (re)starts the containers via docker compose.
    5. Waits for /health and prints the secrets it used.

  Secrets precedence per key (DEBUG_API_KEY, ADMIN_API_KEY, JWT_SECRET):
    environment variable  >  existing server .env  >  generated random hex

.PARAMETER Backend
  'pgvector' starts only db + app (the lean production default).
  'both' starts all services incl. Milvus and sets VECTOR_BACKEND=both
  (needs a >=8 GB instance, e.g. t3.large). Default: both.

.PARAMETER Ip           Override the instance IP (else auto-detected).
.PARAMETER StackName    CloudFormation stack (for IP + key fetch). Default: IndustryIqStack.
.PARAMETER KeyPath      SSH private key path. Default: ~/.ssh/industryiq-cdk-key.pem.
.PARAMETER Region       AWS region. Default: us-east-1.
.PARAMETER MigrateMilvus  After deploy, copy the pgvector corpus into Milvus (needs -Backend both).
.PARAMETER RegenerateSecrets  Ignore existing server secrets and generate fresh ones.
.PARAMETER Start  Start the instance first if it's stopped, then deploy. Pairs with -Stop.
.PARAMETER Stop   Stop the instance to halt compute billing, then exit. The EBS volume
                  (and your corpus on it) is preserved; resume later with -Start.

.EXAMPLE
  ./scripts/deploy.ps1
.EXAMPLE
  ./scripts/deploy.ps1 -Backend both -MigrateMilvus
.EXAMPLE
  $env:JWT_SECRET = "my-fixed-secret"; ./scripts/deploy.ps1
.EXAMPLE
  ./scripts/deploy.ps1 -Stop        # park the box to pause compute billing (keeps data)
.EXAMPLE
  ./scripts/deploy.ps1 -Start       # resume the box, then deploy
#>
[CmdletBinding()]
param(
    [ValidateSet('pgvector', 'both')]
    [string]$Backend = 'both',
    [string]$Ip,
    [string]$StackName = 'IndustryIqStack',
    [string]$KeyPath = "$HOME\.ssh\industryiq-cdk-key.pem",
    [string]$Region = 'us-east-1',
    [switch]$MigrateMilvus,
    [switch]$RegenerateSecrets,
    [switch]$Start,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'

# Run from the repo root regardless of where the script is invoked from.
Set-Location (Split-Path -Parent $PSScriptRoot)

# StrictHostKeyChecking=no: don't prompt on first connect. UserKnownHostsFile=NUL:
# don't fail if a recycled IP now has a different host key. Remove the second -o if
# your OpenSSH build rejects NUL.
$sshOpts = @('-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=NUL', '-i', $KeyPath)

function New-Hex([int]$n) {
    $b = [byte[]]::new($n)
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b)
    ($b | ForEach-Object { $_.ToString('x2') }) -join ''
}

# Read this stack's instance id from its CloudFormation outputs (survives IP churn).
function Get-InstanceId {
    $id = (aws cloudformation describe-stacks --stack-name $StackName --region $Region `
            --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text).Trim()
    if (-not $id -or $id -eq 'None') { throw "Could not read InstanceId from stack $StackName. Is it deployed?" }
    return $id
}

# --- 0. Park the box to stop compute billing (data on EBS is preserved) ------ #
if ($Stop) {
    $id = Get-InstanceId
    Write-Host "Stopping instance $id to halt compute billing..." -ForegroundColor Cyan
    aws ec2 stop-instances --region $Region --instance-ids $id | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to issue stop for $id." }
    Write-Host "Waiting for it to reach 'stopped'..." -ForegroundColor Cyan
    aws ec2 wait instance-stopped --region $Region --instance-ids $id
    Write-Host "Instance stopped. Compute billing paused; the EBS volume (~`$1.60/mo) keeps your corpus." -ForegroundColor Green
    Write-Host "Resume later with:  ./scripts/deploy.ps1 -Start" -ForegroundColor Green
    exit 0
}

# --- 1. Resolve the target instance IP -------------------------------------- #
if ($Start) {
    $id = Get-InstanceId
    $state = (aws ec2 describe-instances --region $Region --instance-ids $id `
            --query "Reservations[0].Instances[0].State.Name" --output text).Trim()
    if ($state -ne 'running') {
        Write-Host "Instance $id is '$state'; starting it..." -ForegroundColor Cyan
        aws ec2 start-instances --region $Region --instance-ids $id | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Failed to start $id." }
        aws ec2 wait instance-running --region $Region --instance-ids $id
        Write-Host "Instance running (Docker may need ~a minute more to come up)." -ForegroundColor Green
    }
    else {
        Write-Host "Instance $id already running." -ForegroundColor Green
    }
    if (-not $Ip) {
        $Ip = (aws ec2 describe-instances --region $Region --instance-ids $id `
                --query "Reservations[0].Instances[0].PublicIpAddress" --output text).Trim()
    }
}
if (-not $Ip) {
    Write-Host "Looking up the running instance IP..." -ForegroundColor Cyan
    $Ip = (aws ec2 describe-instances --region $Region `
            --filters "Name=instance-state-name,Values=running" `
            --query "Reservations[0].Instances[0].PublicIpAddress" --output text).Trim()
}
if (-not $Ip -or $Ip -eq 'None') {
    throw "No running instance found. Is the stack deployed and the instance started?"
}
$Remote = "ec2-user@$Ip"
Write-Host "Target: $Remote  (backend=$Backend)" -ForegroundColor Green

# --- 2. Ensure the SSH key locally (fetch if missing OR stale) -------------- #
# A destroy/recreate mints a brand-new key pair, so a leftover local key from a
# prior stack would silently fail auth. Compare against the stack's current key
# and rewrite when they differ.
Write-Host "Checking the SSH key against stack $StackName..." -ForegroundColor Cyan
$getKeyCmd = aws cloudformation describe-stacks --stack-name $StackName --region $Region `
    --query "Stacks[0].Outputs[?OutputKey=='GetSshKeyCommand'].OutputValue" --output text
if (-not $getKeyCmd -or $getKeyCmd -eq 'None') { throw "Could not read GetSshKeyCommand output from stack $StackName." }
$keyLines = Invoke-Expression $getKeyCmd
$keyText = ($keyLines -join "`n")
if (-not $keyText.Trim()) { throw "Fetched an empty SSH key from stack $StackName (was it deployed?)." }
$localKey = if (Test-Path $KeyPath) { Get-Content $KeyPath -Raw } else { '' }
if (($localKey -replace "`r", '').Trim() -ne ($keyText -replace "`r", '').Trim()) {
    Write-Host "  Local key missing or stale (stack likely recreated); rewriting it." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force (Split-Path $KeyPath) | Out-Null
    $keyLines | Out-File -FilePath $KeyPath -Encoding ascii
    icacls "$KeyPath" /inheritance:r /grant:r "$($env:USERNAME):(R)" | Out-Null
}
else {
    Write-Host "  SSH key is current." -ForegroundColor Green
}

# --- 3. Wait until Docker is reachable on the box --------------------------- #
Write-Host "Waiting for Docker on the box..." -ForegroundColor Cyan
$ready = $false
foreach ($i in 1..30) {
    ssh @sshOpts $Remote "docker buildx version" *> $null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 5
}
if (-not $ready) {
    throw "Can't reach Docker on $Remote. If SSH timed out, your IP may not be allowed on port 22 (re-deploy the stack with -c my_ip=<your-ip>)."
}

# --- 4. Resolve secrets: env var > existing server value > generated -------- #
$existing = @{}
if (-not $RegenerateSecrets) {
    $envLines = ssh @sshOpts $Remote "cat industryiq/.env 2>/dev/null"
    if ($LASTEXITCODE -eq 0) {
        foreach ($line in @($envLines)) {
            if ($line -match '^\s*([A-Z_]+)=(.*)$') { $existing[$Matches[1]] = $Matches[2].Trim() }
        }
    }
}
function Resolve-Secret([string]$name, [int]$bytes) {
    $fromEnv = [Environment]::GetEnvironmentVariable($name)
    if ($fromEnv) { Write-Host "  $name  <- environment"; return $fromEnv }
    if ($existing.ContainsKey($name) -and $existing[$name]) { Write-Host "  $name  <- reused from server"; return $existing[$name] }
    Write-Host "  $name  <- generated"
    return (New-Hex $bytes)
}
Write-Host "Resolving secrets:" -ForegroundColor Cyan
$DKEY = Resolve-Secret 'DEBUG_API_KEY' 16
$AKEY = Resolve-Secret 'ADMIN_API_KEY' 16
$JWT = Resolve-Secret 'JWT_SECRET' 32

# --- 5. Bundle committed code (HEAD) and upload ----------------------------- #
if ($Backend -eq 'both') { $vectorBackend = 'both'; $services = '' }
else { $vectorBackend = 'pgvector'; $services = 'db app' }

$tar = Join-Path $env:TEMP 'industryiq-app.tar.gz'
Write-Host "Bundling committed code (git archive HEAD)..." -ForegroundColor Cyan
git archive --format=tar.gz -o $tar HEAD
if ($LASTEXITCODE -ne 0) { throw "git archive failed (is HEAD a valid commit?)." }
Write-Host "Uploading..." -ForegroundColor Cyan
scp @sshOpts $tar "${Remote}:/home/ec2-user/app.tar.gz"
if ($LASTEXITCODE -ne 0) { throw "scp failed." }

# --- 6. Unpack, write .env, (re)start the containers ------------------------ #
$writeEnv = "printf 'AWS_REGION=$Region\nVECTOR_BACKEND=$vectorBackend\nDEBUG_API_KEY=%s\nADMIN_API_KEY=%s\nJWT_SECRET=%s\n' '$DKEY' '$AKEY' '$JWT' > industryiq/.env"
$composeUp = "docker compose -f docker-compose.yml -f compose.prod.yml up -d --build $services"
$remoteCmd = "rm -rf industryiq && mkdir industryiq && tar xzf app.tar.gz -C industryiq && $writeEnv && cd industryiq && $composeUp"
Write-Host "Deploying ($Backend)..." -ForegroundColor Cyan
ssh @sshOpts $Remote $remoteCmd
if ($LASTEXITCODE -ne 0) { throw "Remote deploy failed." }

# --- 7. Optional: copy the pgvector corpus into Milvus ---------------------- #
if ($MigrateMilvus) {
    if ($Backend -ne 'both') {
        Write-Warning "-MigrateMilvus needs -Backend both; skipping."
    }
    else {
        Write-Host "Waiting ~90s for Milvus to come up, then migrating..." -ForegroundColor Cyan
        Start-Sleep -Seconds 90
        ssh @sshOpts $Remote "docker cp industryiq/scripts/migrate_pg_to_milvus.py industryiq_app:/tmp/m.py && docker exec industryiq_app python /tmp/m.py"
        if ($LASTEXITCODE -ne 0) { Write-Warning "Migration failed (is there data in pgvector, and is Milvus healthy yet?)." }
    }
}

# --- 8. Health check -------------------------------------------------------- #
Write-Host "Waiting for the API health check..." -ForegroundColor Cyan
$healthy = $false
foreach ($i in 1..30) {
    try {
        $r = Invoke-RestMethod "http://${Ip}:8000/health" -TimeoutSec 5
        if ($r.status -eq 'ok') { $healthy = $true; break }
    }
    catch { Start-Sleep -Seconds 5 }
}
Write-Host ""
if ($healthy) {
    Write-Host "Deployed. API healthy at http://${Ip}:8000" -ForegroundColor Green
}
else {
    Write-Warning "API did not report healthy in time. Recent app logs:"
    ssh @sshOpts $Remote "docker logs --tail 40 industryiq_app"
}
Write-Host "Secrets in use (save these):" -ForegroundColor Cyan
Write-Host "  DEBUG_API_KEY=$DKEY"
Write-Host "  ADMIN_API_KEY=$AKEY"
Write-Host "  JWT_SECRET=$JWT"
