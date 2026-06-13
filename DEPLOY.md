# Deploying ragproject to AWS

This is the runbook for deploying ragproject to a single EC2 instance running
the Docker Compose stack (FastAPI app + Postgres/pgvector), with the
infrastructure managed by **AWS CDK** (`infra/`).

Two lifecycles, kept separate:

- **Infrastructure** (the VM, network, IAM) — managed by CDK (`cdk deploy` / `cdk destroy`).
- **Application code** — shipped with a bundle-and-compose step (Step 5).

All commands are Windows PowerShell unless marked **(Git Bash)**.

---

## Prerequisites (install once)

| Tool | Check | Install |
| --- | --- | --- |
| Node.js | `node --version` | <https://nodejs.org> |
| CDK CLI | `cdk --version` | `npm install -g aws-cdk` |
| AWS CLI | `aws --version` | <https://aws.amazon.com/cli/> |
| Python 3.11+ | `python --version` | <https://python.org> |
| Docker Desktop | `docker --version` | for local testing |

> After `npm install -g aws-cdk` (or installing Docker), the command won't be
> found in the *same* terminal — open a **fresh terminal** and it works.

---

## Step 0 — AWS account setup (once)

1. Configure credentials and confirm they work:
   ```powershell
   aws configure                 # access key, secret, region us-east-1
   aws sts get-caller-identity   # prints your account ARN if OK
   ```
2. Attach **`AdministratorAccess`** to your IAM user (CDK uses CloudFormation,
   S3, SSM, IAM, EC2). Fine for a personal account; scope down for shared ones.
3. Enable **Bedrock model access**: Console -> Bedrock -> Model access -> grant
   Anthropic Claude + Amazon Titan Text Embeddings, and submit the Anthropic
   use-case form.

---

## Step 1 — Bootstrap CDK (once per account + region)

```powershell
cd infra
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cdk bootstrap aws://<ACCOUNT_ID>/us-east-1
```

Creates the `CDKToolkit` support stack. Done once; reused forever.

---

## Step 2 — Deploy the infrastructure

```powershell
# in infra/, with .venv active
$myip = (Invoke-WebRequest https://checkip.amazonaws.com -UseBasicParsing).Content.Trim()

cdk diff   -c my_ip=$myip                      # preview (optional)
cdk deploy -c my_ip=$myip --require-approval never
```

Note the **outputs** it prints: `PublicIp`, `InstanceId`, and `GetSshKeyCommand`.

> **Always pass `-c my_ip=$myip`.** It locks SSH (port 22) to your IP. If you run
> a bare `cdk deploy`, `my_ip` defaults to `0.0.0.0` and SSH is locked to nobody
> (`0.0.0.0/32`) — you'll get `Connection timed out` in Step 4/5.

### Fix: SSH locked out (or your IP changed)

If SSH times out, open port 22 to your current IP on the instance's security group:

```powershell
$myip = (Invoke-WebRequest https://checkip.amazonaws.com -UseBasicParsing).Content.Trim()
$id  = aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" --query "Reservations[0].Instances[0].InstanceId" --output text
$sg  = aws ec2 describe-instances --instance-ids $id --query "Reservations[0].Instances[0].SecurityGroups[0].GroupId" --output text
aws ec2 authorize-security-group-ingress --group-id $sg --protocol tcp --port 22 --cidr "$myip/32"
```

This is a manual quick-fix; the proper fix is to redeploy with `-c my_ip=$myip`
(CDK reverts manual security-group edits on the next deploy).

---

## Step 3 — Fetch the SSH key

CDK stored the private key in SSM Parameter Store. Run the printed
`GetSshKeyCommand`, saving to a file:

```powershell
$keyPath = "$HOME\.ssh\ragproject-cdk-key.pem"
aws ssm get-parameter --name /ec2/keypair/<KEY_PAIR_ID> --with-decryption `
  --query Parameter.Value --output text | Out-File -FilePath $keyPath -Encoding ascii
icacls $keyPath /inheritance:r /grant:r "$($env:USERNAME):(R)"
```

> `-Encoding ascii` is required — PowerShell's default UTF-16 corrupts the key.

---

## Step 4 — Wait for Docker (boot script)

The instance installs Docker via user-data on first boot (~1-2 min). Verify
**(Git Bash)**, substituting the IP:

```bash
ssh -o StrictHostKeyChecking=no -i ~/.ssh/ragproject-cdk-key.pem \
  ec2-user@<PUBLIC_IP> "docker --version && docker compose version && docker buildx version"
```

When all three print versions, it's ready. (`ec2-user` is the Amazon Linux login.)

---

## Step 5 — Deploy the application code

From the **repo root** **(Git Bash)**:

```bash
KEY=~/.ssh/ragproject-cdk-key.pem
HOST=ec2-user@<PUBLIC_IP>
SSHO="-o StrictHostKeyChecking=no -i $KEY"

# 1. bundle committed files only (no .venv/.git):
git archive --format=tar.gz -o /tmp/app.tar.gz HEAD

# 2. upload:
scp $SSHO /tmp/app.tar.gz $HOST:/home/ec2-user/

# Pick your own debug key, or let it generate a random one:
DKEY=${DEBUG_API_KEY:-$(openssl rand -hex 16)}
echo "DEBUG_API_KEY = $DKEY"     # <-- copy this; it guards the /debug/* endpoints

# 3. unpack, write .env (Bedrock on), start the stack:
ssh $SSHO $HOST "rm -rf ragproject && mkdir ragproject && tar xzf app.tar.gz -C ragproject && \
  printf 'RAG_PROVIDER=bedrock\nAWS_REGION=us-east-1\nDEBUG_API_KEY=%s\n' '$DKEY' > ragproject/.env && \
  cd ragproject && docker compose up -d --build"
```

To use a **fixed** key (so it's the same every deploy), set it first:
`export DEBUG_API_KEY=my-fixed-key-123`. Otherwise a random one is generated
and printed by the `echo` above.

### Retrieving the debug key from a running deployment

The server's `.env` is the source of truth. Retrieve it anytime in three steps:

```powershell
# 1. find the running instance's public IP
aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" `
  --query "Reservations[].Instances[].PublicIpAddress" --output text
```

```bash
# 2. ensure you have the SSH key (re-fetch from SSM if you deleted it; see Step 3)
# 3. SSH in and read the key (substitute the IP):
ssh -o StrictHostKeyChecking=no -i ~/.ssh/ragproject-cdk-key.pem \
  ec2-user@<PUBLIC_IP> "grep DEBUG_API_KEY ragproject/.env"
```

Prints `DEBUG_API_KEY=...`. If SSH times out, you're locked out — see
"Fix: SSH locked out" under Step 2. If there is **no** `.env`, the app wasn't
deployed yet (you ran only `cdk deploy`, not Step 5).

---

## Step 6 — Test

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

## Updating later

- **Infra change** (edit `infra/ragproject_infra/ragproject_stack.py`):
  `cdk deploy -c my_ip=$myip` — applies only the diff.
- **App change**: re-run **Step 5** (re-bundle + scp + `docker compose up -d --build`).
  No CDK needed.

---

## Step 7 — Tear it all down (stop billing)

```powershell
cd infra
.\.venv\Scripts\Activate.ps1
cdk destroy        # deletes the instance, security group, IAM role, key pair
```

Optionally also:

```powershell
aws iam detach-user-policy --user-name <you> `
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
Remove-Item $HOME\.ssh\ragproject-cdk-key.pem -Force
```

Leave the `CDKToolkit` bootstrap stack — it's free and reused next time.

---

## Mental model

```
ONE-TIME:  install tools -> aws configure -> cdk bootstrap
INFRA:     cdk deploy          (creates the server; repeatable, reviewable)
GET IN:    fetch SSH key from SSM
APP:       git archive -> scp -> docker compose up   (ships the code)
TEST:      curl / browser
TEARDOWN:  cdk destroy         (removes everything in one command)
```

The win over hand-typed CLI: Steps 2 and 7 each replace ~15 manual commands and
are fully reproducible — anyone with the repo gets an identical environment.

## Notes

- **Cost:** the `t3.small` bills ~$0.50/day while running. `cdk destroy` stops it.
- **Security:** `/query` and `/ingest` are public over plain HTTP (fine for a
  demo). `/debug/*` requires the `X-Debug-Key` header. For production, add HTTPS
  and auth on the public routes, and scope the IAM permissions down.
