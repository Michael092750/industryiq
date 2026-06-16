# Deploying ragproject to AWS

The deployment has **two independent parts**:

- **Infrastructure** — the EC2 server, network, IAM. Managed by **AWS CDK** (`infra/`).
- **Application** — your code + Docker containers. Shipped by a **deploy script** (bundle → scp → `docker compose up`).

You almost always need **both**: `cdk deploy` creates an *empty server*; the app
script puts your code on it.

## Which section do I need?

| Your situation | Go to |
| --- | --- |
| Deploying for the very first time | [1. First-time deployment](#1-first-time-deployment) |
| Redeploying after `cdk destroy` | [2. Redeploy from scratch](#2-redeploy-from-scratch) |
| Changed only app code, server is up | [3. Update the app](#3-update-the-app-only) |
| Changed the infra (stack code) | [4. Update the infrastructure](#4-update-the-infrastructure) |
| Need the debug key / IP / to SSH in | [5. Common tasks](#5-common-tasks) |
| Something broke | [6. Troubleshooting](#6-troubleshooting) |
| Stop the bill | [7. Tear down](#7-tear-down) |

The actual command sequences live in [Procedures](#procedures) (A = infra, B = app);
the sections below tell you which to run.

---

## 1. First-time deployment

Do the one-time setup, then deploy.

### 1a. Prerequisites (install once)

| Tool | Check | Install |
| --- | --- | --- |
| Node.js | `node --version` | <https://nodejs.org> |
| CDK CLI | `cdk --version` | `npm install -g aws-cdk` |
| AWS CLI | `aws --version` | <https://aws.amazon.com/cli/> |
| Python 3.11+ | `python --version` | <https://python.org> |
| Docker Desktop | `docker --version` | for local testing |

> After installing a CLI, open a **fresh terminal** or it won't be found. If `cdk`
> is still "not recognized," see [Troubleshooting](#6-troubleshooting).

### 1b. AWS account setup (once)

```powershell
aws configure                 # access key, secret, region us-east-1
aws sts get-caller-identity   # prints your account ARN if OK
```

- Attach **`AdministratorAccess`** to your IAM user (CDK needs CloudFormation, S3, SSM, IAM, EC2).
- Enable **Bedrock model access**: Console → Bedrock → Model access → grant Anthropic
  Claude + Amazon Titan, and submit the Anthropic use-case form.

### 1c. Bootstrap CDK (once per account + region)

```powershell
cd infra
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cdk bootstrap aws://<ACCOUNT_ID>/us-east-1
```

### 1d. Deploy

1. Run **[Procedure A — Deploy the infrastructure](#procedure-a--deploy-the-infrastructure)**
2. Run **[Procedure B — Deploy the app](#procedure-b--deploy-the-app)**
3. [Test](#test-any-deployment)

---

## 2. Redeploy from scratch

After a `cdk destroy`, the one-time setup (1a–1c) is **already done** and persists.
Just:

1. Run **[Procedure A — Deploy the infrastructure](#procedure-a--deploy-the-infrastructure)**
2. Run **[Procedure B — Deploy the app](#procedure-b--deploy-the-app)**
3. [Test](#test-any-deployment)

A fresh deploy creates a **new** public IP and SSH key — don't reuse the old ones.

---

## 3. Update the app only

The server is already running and you changed app code. **No CDK needed** — just
re-run **[Procedure B — Deploy the app](#procedure-b--deploy-the-app)**. It rebuilds
the image and restarts the containers in place.

---

## 4. Update the infrastructure

You edited `infra/ragproject_infra/ragproject_stack.py` (e.g. instance size).
Re-run **[Procedure A](#procedure-a--deploy-the-infrastructure)** — CDK computes the
diff and applies only what changed. Run `cdk diff -c my_ip=$myip` first to preview.

---

## 5. Common tasks

> First set these in a **Git Bash** shell (used by several tasks):
> ```bash
> KEY=~/.ssh/ragproject-cdk-key.pem
> HOST=ec2-user@<PUBLIC_IP>
> SSHO="-o StrictHostKeyChecking=no -i $KEY"
> ```

**Find the running instance's public IP** (PowerShell):
```powershell
aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" `
  --query "Reservations[].Instances[].PublicIpAddress" --output text
```

**Re-fetch the SSH key** (if you deleted the `.pem`) — get the param name from the
stack outputs, then (PowerShell):
```powershell
$cmd = aws cloudformation describe-stacks --stack-name RagprojectStack `
  --query "Stacks[0].Outputs[?OutputKey=='GetSshKeyCommand'].OutputValue" --output text
$keyPath = "$HOME\.ssh\ragproject-cdk-key.pem"
Invoke-Expression $cmd | Out-File -FilePath $keyPath -Encoding ascii
icacls $keyPath /inheritance:r /grant:r "$($env:USERNAME):(R)"
```

**Retrieve the debug key** (the `X-Debug-Key` for `/debug/*`) — Git Bash:
```bash
ssh $SSHO $HOST "grep DEBUG_API_KEY ragproject/.env"
```
No output / no file? The app isn't deployed — run Procedure B.

**SSH into the box** — Git Bash:
```bash
ssh $SSHO $HOST
```

---

## 6. Troubleshooting

**`cdk : not recognized`** — npm's folder isn't on PATH. Quick fix (this terminal):
```powershell
$env:Path += ";$env:APPDATA\npm"
```
Permanent: add `%APPDATA%\npm` to your user PATH, then **fully restart VS Code**
(a new terminal tab inherits VS Code's old PATH).

**`ModuleNotFoundError: No module named 'aws_cdk'`** — you ran `cdk` without the
**infra** venv active. `cd infra; .\.venv\Scripts\Activate.ps1` first. (Note: this is
the *infra* venv, not the app's root `.venv`.)

**SSH `Connection timed out`** — port 22 isn't open to your IP. Happens if you ran a
bare `cdk deploy` (SSH locked to `0.0.0.0/32`) or your IP changed. Open it:
```powershell
$myip = (Invoke-WebRequest https://checkip.amazonaws.com -UseBasicParsing).Content.Trim()
$id  = aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" --query "Reservations[0].Instances[0].InstanceId" --output text
$sg  = aws ec2 describe-instances --instance-ids $id --query "Reservations[0].Instances[0].SecurityGroups[0].GroupId" --output text
aws ec2 authorize-security-group-ingress --group-id $sg --protocol tcp --port 22 --cidr "$myip/32"
```
Proper fix: redeploy with `-c my_ip=$myip` (CDK reverts manual SG edits next deploy).

**No `.env` on the server / no `/health`** — you ran only `cdk deploy` (infra) and
skipped the app. Run **[Procedure B](#procedure-b--deploy-the-app)**.

**`compose build requires buildx ...`** — old Docker plugin. The CDK user-data
installs a current buildx, so a fresh instance is fine; only relevant on hand-built boxes.

---

## 7. Tear down

```powershell
cd infra
.\.venv\Scripts\Activate.ps1
cdk destroy          # deletes the instance, security group, IAM role, key pair
```

Optional extra cleanup (only if fully done with AWS):
```powershell
aws iam detach-user-policy --user-name <you> --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
Remove-Item $HOME\.ssh\ragproject-cdk-key.pem -Force
```
Leave the `CDKToolkit` bootstrap stack — it's free and reused next time.

---

## Procedures

### Procedure A — Deploy the infrastructure

In `infra/`, with the **infra venv active** (`.\.venv\Scripts\Activate.ps1`):

```powershell
$myip = (Invoke-WebRequest https://checkip.amazonaws.com -UseBasicParsing).Content.Trim()
cdk deploy -c my_ip=$myip --require-approval never
```

- **Always pass `-c my_ip=$myip`** — it locks SSH to your IP. A bare `cdk deploy`
  locks SSH to nobody (`0.0.0.0/32`).
- Note the outputs: **`PublicIp`**, **`InstanceId`**, **`GetSshKeyCommand`**.
- This creates an **empty server** — you still need Procedure B.

Then fetch the SSH key (run the printed `GetSshKeyCommand`, saving it):
```powershell
$keyPath = "$HOME\.ssh\ragproject-cdk-key.pem"
aws ssm get-parameter --name /ec2/keypair/<KEY_PAIR_ID> --with-decryption `
  --query Parameter.Value --output text | Out-File -FilePath $keyPath -Encoding ascii
icacls $keyPath /inheritance:r /grant:r "$($env:USERNAME):(R)"
```
> `-Encoding ascii` is required — PowerShell's default UTF-16 corrupts the key.

### Procedure B — Deploy the app

The instance installs Docker on first boot (~1-2 min). From the **repo root** in
**Git Bash**, with `<PUBLIC_IP>` filled in:

```bash
KEY=~/.ssh/ragproject-cdk-key.pem
HOST=ec2-user@<PUBLIC_IP>
SSHO="-o StrictHostKeyChecking=no -i $KEY"

# wait until Docker is ready:
until ssh $SSHO $HOST "docker buildx version" 2>/dev/null; do echo waiting; sleep 10; done

# choose your debug key (pin a fixed one so it's predictable; else random):
DKEY=${DEBUG_API_KEY:-$(openssl rand -hex 16)}
echo "DEBUG_API_KEY = $DKEY"          # <-- copy this; guards /debug/*

# bundle committed code, upload, write .env, build + start:
git archive --format=tar.gz -o /tmp/app.tar.gz HEAD
scp $SSHO /tmp/app.tar.gz $HOST:/home/ec2-user/
ssh $SSHO $HOST "rm -rf ragproject && mkdir ragproject && tar xzf app.tar.gz -C ragproject && \
  printf 'AWS_REGION=us-east-1\nDEBUG_API_KEY=%s\n' '$DKEY' > ragproject/.env && \
  cd ragproject && docker compose -f docker-compose.yml -f compose.prod.yml up -d --build"
```

> The `-f docker-compose.yml -f compose.prod.yml` overlay sets `RAG_PROVIDER=bedrock`
> (committed in `compose.prod.yml`), so you no longer write it into `.env` on each
> deploy. Bedrock authenticates via the instance IAM role — no key on the box.
> Locally you instead set `RAG_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` in `.env`.

> Pin a fixed key with `export DEBUG_API_KEY=my-key` before running, so it's the
> same every deploy.

### Test (any deployment)

```bash
IP=<PUBLIC_IP>
curl http://$IP:8000/health
curl -X POST http://$IP:8000/ingest -H "Content-Type: application/json" \
  -d '{"text":"Paris is the capital of France.","source":"f.txt"}'
curl -X POST http://$IP:8000/query  -H "Content-Type: application/json" \
  -d '{"question":"What is the capital of France?"}'
```
Or open `http://<PUBLIC_IP>:8000/docs` in a browser.

---

## Reference

**Endpoints:** `/health`, `/ingest`, `/query`, `/ingest/file`, `/docs`,
`/debug/chunks` + `/debug-ui` (need `X-Debug-Key`).

**Cost:** the `t3.small` bills ~$0.50/day while running. `cdk destroy` stops it.

**Security:** `/query` and `/ingest` are public over plain HTTP (fine for a demo).
`/debug/*` requires the key. For production: add HTTPS + auth on public routes and
scope IAM down.
