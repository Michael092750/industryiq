#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Stop all IndustryIQ compute billing (reversible; your data is preserved).

.DESCRIPTION
  The bill is driven by the EC2 instance -- the app, pgvector and Milvus all run
  on it, so stopping the instance halts the compute charge while the EBS volume
  (and the corpus on it) stays intact. Resume later with:

    ./scripts/deploy.ps1 -Start

  What this script does:
    1. Stops this stack's EC2 instance (found via the InstanceId stack output).
    2. With -AllInstances, also stops every OTHER running instance in the region.
    3. Prints a read-only billing audit of anything still able to cost money
       (running instances, unassociated Elastic IPs, EBS volumes, NAT gateways,
       load balancers, RDS) so nothing keeps billing without you knowing.

  It never terminates or deletes anything. To fully remove the stack -- which
  DESTROYS the corpus and forces a re-ingest -- run instead:

    cdk destroy IndustryIqStack

  Note: resources are regional, so this audits -Region only (default us-east-1).

.PARAMETER StackName     CloudFormation stack. Default: IndustryIqStack.
.PARAMETER Region        AWS region to act on and audit. Default: us-east-1.
.PARAMETER AllInstances  Also stop every other running EC2 instance in the region.
.PARAMETER Audit         Skip stopping; only print the billing audit.

.EXAMPLE
  ./scripts/stop-billing.ps1
.EXAMPLE
  ./scripts/stop-billing.ps1 -AllInstances
.EXAMPLE
  ./scripts/stop-billing.ps1 -Audit
#>
[CmdletBinding()]
param(
    [string]$StackName = 'IndustryIqStack',
    [string]$Region = 'us-east-1',
    [switch]$AllInstances,
    [switch]$Audit
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

# Run an aws query and return its output split into a clean list of tokens.
# Args are passed as one array so '--flags' are never parsed by PowerShell.
function Get-AwsItems([string[]]$AwsArgs) {
    $out = & aws @AwsArgs 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) { return @() }
    return @(($out -join "`n") -split '\s+' | Where-Object { $_ -ne '' })
}

# This stack's instance id from its CloudFormation outputs; $null if no stack.
function Get-StackInstanceId {
    $out = & aws cloudformation describe-stacks --stack-name $StackName --region $Region `
        --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
    $id = ([string]$out).Trim()
    if (-not $id -or $id -eq 'None') { return $null }
    return $id
}

$stackId = Get-StackInstanceId

# --- 1. Stop this stack's instance ------------------------------------------ #
if (-not $Audit) {
    if (-not $stackId) {
        Write-Warning "No InstanceId for stack $StackName (already destroyed or not deployed?). Skipping the primary stop."
    }
    else {
        $state = (aws ec2 describe-instances --region $Region --instance-ids $stackId `
                --query "Reservations[0].Instances[0].State.Name" --output text).Trim()
        if ($state -eq 'running') {
            Write-Host "Stopping $StackName instance $stackId ..." -ForegroundColor Cyan
            aws ec2 stop-instances --region $Region --instance-ids $stackId | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "Failed to issue stop for $stackId." }
            Write-Host "Waiting for it to reach 'stopped'..." -ForegroundColor Cyan
            aws ec2 wait instance-stopped --region $Region --instance-ids $stackId
            Write-Host "  Stopped. Compute billing paused; the EBS volume keeps your corpus." -ForegroundColor Green
        }
        else {
            Write-Host "$StackName instance $stackId is already '$state'." -ForegroundColor Green
        }
    }

    # --- 2. Optionally stop every other running instance in the region ------ #
    if ($AllInstances) {
        $others = @(Get-AwsItems @('ec2', 'describe-instances', '--region', $Region,
                '--filters', 'Name=instance-state-name,Values=running',
                '--query', 'Reservations[].Instances[].InstanceId', '--output', 'text') |
            Where-Object { $_ -ne $stackId })
        if ($others.Count -gt 0) {
            Write-Host "Stopping $($others.Count) other running instance(s): $($others -join ', ')" -ForegroundColor Cyan
            aws ec2 stop-instances --region $Region --instance-ids $others | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "Failed to stop other instances." }
            aws ec2 wait instance-stopped --region $Region --instance-ids $others
            Write-Host "  Stopped." -ForegroundColor Green
        }
        else {
            Write-Host "No other running instances in $Region." -ForegroundColor Green
        }
    }
}

# --- 3. Billing audit (read-only) ------------------------------------------- #
Write-Host ""
Write-Host "Billing audit for region ${Region}:" -ForegroundColor Cyan

function Show-Flag([string]$label, [string[]]$items, [string]$hint) {
    if ($items.Count -gt 0) {
        Write-Host "  [!] $label ($($items.Count)): $($items -join ', ')" -ForegroundColor Yellow
        if ($hint) { Write-Host "      $hint" -ForegroundColor DarkYellow }
    }
    else {
        Write-Host "  [ok] ${label}: none" -ForegroundColor Green
    }
}

$running = Get-AwsItems @('ec2', 'describe-instances', '--region', $Region,
    '--filters', 'Name=instance-state-name,Values=running',
    '--query', 'Reservations[].Instances[].InstanceId', '--output', 'text')
Show-Flag 'Running EC2 instances (billing compute)' $running 'stop with: aws ec2 stop-instances --instance-ids <id>'

$eips = Get-AwsItems @('ec2', 'describe-addresses', '--region', $Region,
    '--query', 'Addresses[?AssociationId==`null`].AllocationId', '--output', 'text')
Show-Flag 'Unassociated Elastic IPs (bill while idle)' $eips 'release with: aws ec2 release-address --allocation-id <id>'

$nats = Get-AwsItems @('ec2', 'describe-nat-gateways', '--region', $Region,
    '--filter', 'Name=state,Values=available',
    '--query', 'NatGateways[].NatGatewayId', '--output', 'text')
Show-Flag 'NAT gateways (~$32/mo each)' $nats 'not used by this stack -- investigate if present'

$lbs = Get-AwsItems @('elbv2', 'describe-load-balancers', '--region', $Region,
    '--query', 'LoadBalancers[].LoadBalancerName', '--output', 'text')
Show-Flag 'Load balancers (bill continuously)' $lbs 'not used by this stack -- investigate if present'

$rds = Get-AwsItems @('rds', 'describe-db-instances', '--region', $Region,
    '--query', 'DBInstances[].DBInstanceIdentifier', '--output', 'text')
Show-Flag 'RDS databases (bill continuously)' $rds 'not used by this stack -- investigate if present'

# EBS is expected and cheap; stopping the instance does not remove it.
$vols = Get-AwsItems @('ec2', 'describe-volumes', '--region', $Region,
    '--query', 'Volumes[].VolumeId', '--output', 'text')
if ($vols.Count -gt 0) {
    Write-Host "  [--] EBS volumes ($($vols.Count)): expected; ~`$0.08/GB-mo, keeps your corpus. Removed only by 'cdk destroy'." -ForegroundColor Gray
}

Write-Host ""
Write-Host "Done. Resume the app any time with:  ./scripts/deploy.ps1 -Start" -ForegroundColor Green
