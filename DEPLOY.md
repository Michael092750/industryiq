# Deploying IndustryIQ to AWS

The deployment has **two independent parts**:

- **Infrastructure** — the EC2 server, network, IAM. Managed by **AWS CDK** (`infra/`).
- **Application** — your code + Docker containers. Shipped by a **deploy script** (bundle → scp → `docker compose up`).

You almost always need **both**: `cdk deploy` creates an *empty server*; the app
script puts your code on it.

> **What you end up with:** one `t3.small` running the FastAPI backend + Postgres
> (pgvector) in Docker, answering on port `8000`. There is **no web "front page"** —
> only the JSON API plus `/docs` (interactive API explorer) and `/health`. The React
> UI in `frontend/` is a *separate* app and is **not** deployed here; see
> [Test](#test-any-deployment) for how to reach the app.

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

> Commands are written for **PowerShell** (infra) and **Git Bash** (the app deploy,
> which needs `ssh`/`scp`). Each code block says which shell it expects.

---

## 1. First-time deployment

Do the one-time setup (1a–1c), then deploy (1d).

### 1a. Install the tools (once)

Install all of these, then **verify each one prints a version** before continuing —
a missing or not-on-`PATH` tool is the #1 cause of deploy failures.

| Tool | Verify with | Install |
| --- | --- | --- |
| Python 3.11+ | `python --version` | <https://python.org> |
| Node.js (only needed for the CDK CLI) | `node --version` | <https://nodejs.org> |
| **AWS CLI** | `aws --version` | <https://aws.amazon.com/cli/> |
| **CDK CLI** | `cdk --version` | `npm install -g aws-cdk` |
| Git (provides Git Bash) | `git --version` | <https://git-scm.com> |
| Docker Desktop | `docker --version` | <https://docker.com> (local testing only) |

> **Two different "CDK" things — do not confuse them:**
> - **`cdk`** — the **CLI** that runs `cdk bootstrap` / `cdk deploy`. Comes *only*
>   from **`npm install -g aws-cdk`**.
> - **`aws-cdk-lib`** — the **Python library** the stack code imports. Comes from
>   `pip install -r requirements.txt` in step 1c.
>
> Installing the Python library does **not** give you the `cdk` command. "cdk not
> recognized" almost always means you skipped `npm install -g aws-cdk` (or it's a
> PATH issue — see [Troubleshooting](#6-troubleshooting)).

> The **AWS CLI is required** for the `aws ...` commands throughout this doc. If you
> deliberately skip it, every step has a **"No AWS CLI?"** Python fallback you can use
> instead (they need `pip install boto3`).

> After installing any CLI, open a **fresh terminal** — a tool installed into a
> running shell won't be on its `PATH`.

### 1b. AWS account setup (once)

1. **Create an access key:** Console → **IAM** → Users → *your user* → **Security
   credentials** tab → **Create access key** → use case *Command Line Interface*.
   Copy the **Access key ID** and **Secret access key** — the secret is shown **only
   once** (use *Download .csv*). Use an IAM user, not the root account.
2. **Grant permissions:** attach **`AdministratorAccess`** to that IAM user (CDK
   needs CloudFormation, S3, SSM, IAM, EC2).
3. **Configure the credentials** — either of:
   ```powershell
   aws configure                 # with the AWS CLI: enter key, secret, region us-east-1
   ```
   or, **without the AWS CLI**, set them for the current terminal session:
   ```powershell
   $env:AWS_ACCESS_KEY_ID     = "AKIA..."
   $env:AWS_SECRET_ACCESS_KEY = "..."
   $env:AWS_DEFAULT_REGION    = "us-east-1"
   ```
   (Env vars last only for that terminal; `aws configure` persists them in `~/.aws`.)
4. **Enable Bedrock model access** (**required**, or chat returns errors): Console →
   **Bedrock** → *Model access* → grant **Anthropic Claude** + **Amazon Titan**, and
   submit the Anthropic use-case form. Approval can take a little while.

Verify: `aws sts get-caller-identity` should print your account ARN.

### 1c. Bootstrap CDK (once per account + region)

```powershell
cd infra
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt               # installs the CDK *library* (aws-cdk-lib)
cdk bootstrap aws://<ACCOUNT_ID>/us-east-1     # needs the CDK *CLI* from step 1a
```

> `cdk bootstrap` failing with **"cdk not recognized"**? You haven't run
> `npm install -g aws-cdk` (step 1a). `pip install` does not provide the `cdk` command.

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

A fresh deploy creates a **new** public IP **and a new SSH key** — don't reuse the
old ones. (The stack creates the key pair, so `destroy` deletes it and the next
`deploy` regenerates it.)

---

## 3. Update the app only

The server is already running and you changed app code. **No CDK needed** — just
re-run **[Procedure B — Deploy the app](#procedure-b--deploy-the-app)**. It rebuilds
the image and restarts the containers in place. The same SSH key and IP keep working.

---

## 4. Update the infrastructure

You edited `infra/industryiq_infra/industryiq_stack.py` (e.g. instance size).
Re-run **[Procedure A](#procedure-a--deploy-the-infrastructure)** — CDK computes the
diff and applies only what changed. Run `cdk diff -c my_ip=$myip` first to preview.

---

## 5. Common tasks

> First set these in a **Git Bash** shell (used by several tasks):
> ```bash
> KEY=~/.ssh/industryiq-cdk-key.pem
> HOST=ec2-user@<PUBLIC_IP>
> SSHO="-o StrictHostKeyChecking=no -i $KEY"
> ```

**Find the running instance's public IP** (PowerShell):
```powershell
aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" `
  --query "Reservations[].Instances[].PublicIpAddress" --output text
```
> **No AWS CLI?**
> ```powershell
> python -c "import boto3; print([i.get('PublicIpAddress') for r in boto3.client('ec2','us-east-1').describe_instances(Filters=[{'Name':'instance-state-name','Values':['running']}])['Reservations'] for i in r['Instances']])"
> ```

**Re-fetch the SSH key** (if you deleted the `.pem`) — PowerShell, with the AWS CLI:
```powershell
$cmd = aws cloudformation describe-stacks --stack-name IndustryIqStack `
  --query "Stacks[0].Outputs[?OutputKey=='GetSshKeyCommand'].OutputValue" --output text
$keyPath = "$HOME\.ssh\industryiq-cdk-key.pem"
Invoke-Expression $cmd | Out-File -FilePath $keyPath -Encoding ascii
icacls $keyPath /inheritance:r /grant:r "$($env:USERNAME):(R)"
```
> **No AWS CLI?** Run the helper script (reads your credentials, writes the `.pem`,
> sets permissions):
> ```powershell
> pip install boto3
> python scripts/fetch_ssh_key.py
> ```

**Retrieve a server secret** (e.g. the `DEBUG_API_KEY` for `/debug/*`) — Git Bash:
```bash
ssh $SSHO $HOST "grep -E 'DEBUG_API_KEY|ADMIN_API_KEY' industryiq/.env"
```
No output / no file? The app isn't deployed — run Procedure B.

**SSH into the box** — Git Bash:
```bash
ssh $SSHO $HOST
```

---

## 6. Troubleshooting

**`cdk : not recognized`** — either you didn't `npm install -g aws-cdk` (step 1a), or
npm's global folder isn't on `PATH`. Quick PATH fix (this terminal only):
```powershell
$env:Path += ";$env:APPDATA\npm"
```
Permanent: add `%APPDATA%\npm` to your user PATH, then **fully restart VS Code** (a
new terminal tab inherits VS Code's old PATH).

**`ModuleNotFoundError: No module named 'aws_cdk'`** — you ran `cdk` without the
**infra** venv active. `cd infra; .\.venv\Scripts\Activate.ps1` first. (This is the
*infra* venv, not the app's root `.venv`.)

**`Unable to locate credentials` / `NoCredentialsError`** — your AWS credentials
aren't set in this terminal. Re-do step 1b (`aws configure`, or set the `AWS_*` env
vars). Env vars only live for the terminal session that set them.

**`401 {"detail":"Not authenticated"}` from `/conversations/*`** — these routes need
a login token. Register or log in at `/auth/*` and send `Authorization: Bearer
<token>`. See [Test](#test-any-deployment).

**`http://<IP>:8000/` shows `404 Not Found`** — that's expected: there is no front
page, only the API. Use `http://<IP>:8000/docs`. The React UI isn't deployed here.

**App container restarts / instance unresponsive after deploy** — you started the
Milvus stack. Deploy **only `db app`** (Procedure B); a bare `docker compose up` also
starts etcd + minio + milvus and OOMs the 2 GB `t3.small`.

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

> **Just pausing, not done?** Instead of destroying, **stop** the instance — compute
> charges stop, while the disk (~$1.60/mo), your data, and the key survive:
> ```powershell
> aws ec2 stop-instances --instance-ids <InstanceId>     # start-instances to resume
> ```
> (Or Console → EC2 → Instances → *Instance state* → Stop.) On restart the **public
> IP changes**, and the app **won't auto-start** — re-run the final `docker compose
> ... up -d db app` from Procedure B. Use `cdk destroy` to end all charges.

Optional extra cleanup (only if fully done with AWS):
```powershell
aws iam detach-user-policy --user-name <you> --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
Remove-Item $HOME\.ssh\industryiq-cdk-key.pem -Force
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

Then fetch the SSH key. **With the AWS CLI**, run the printed `GetSshKeyCommand`,
saving its output:
```powershell
$keyPath = "$HOME\.ssh\industryiq-cdk-key.pem"
aws ssm get-parameter --name /ec2/keypair/<KEY_PAIR_ID> --with-decryption `
  --query Parameter.Value --output text | Out-File -FilePath $keyPath -Encoding ascii
icacls $keyPath /inheritance:r /grant:r "$($env:USERNAME):(R)"
```
> `-Encoding ascii` is required — PowerShell's default UTF-16 corrupts the key.

> **No AWS CLI?** Do the same with one command:
> ```powershell
> pip install boto3
> python scripts/fetch_ssh_key.py        # writes ~/.ssh/industryiq-cdk-key.pem
> ```

### Procedure B — Deploy the app

The instance installs Docker on first boot (~1–2 min). From the **repo root** in
**Git Bash**, with `<PUBLIC_IP>` filled in:

```bash
KEY=~/.ssh/industryiq-cdk-key.pem
HOST=ec2-user@<PUBLIC_IP>
SSHO="-o StrictHostKeyChecking=no -i $KEY"

# wait until Docker is ready on the box:
until ssh $SSHO $HOST "docker buildx version" 2>/dev/null; do echo waiting; sleep 10; done

# pick the server secrets (random if unset). SAVE THESE — you need them later:
DKEY=${DEBUG_API_KEY:-$(openssl rand -hex 16)}     # X-Debug-Key, guards /debug/*
AKEY=${ADMIN_API_KEY:-$(openssl rand -hex 16)}     # X-Admin-Key, guards /admin/ingest
JWT=${JWT_SECRET:-$(openssl rand -hex 32)}         # signs login tokens
echo "DEBUG_API_KEY=$DKEY"; echo "ADMIN_API_KEY=$AKEY"; echo "JWT_SECRET=$JWT"

# bundle committed code, upload, write .env, build + start ONLY db + app:
git archive --format=tar.gz -o /tmp/app.tar.gz HEAD
scp $SSHO /tmp/app.tar.gz $HOST:/home/ec2-user/
ssh $SSHO $HOST "rm -rf industryiq && mkdir industryiq && tar xzf app.tar.gz -C industryiq && \
  printf 'AWS_REGION=us-east-1\nDEBUG_API_KEY=%s\nADMIN_API_KEY=%s\nJWT_SECRET=%s\n' '$DKEY' '$AKEY' '$JWT' > industryiq/.env && \
  cd industryiq && docker compose -f docker-compose.yml -f compose.prod.yml up -d --build db app"
```

> **Start only `db app`** — not a bare `docker compose up`. The base compose also
> defines a Milvus stack (etcd + minio + milvus) used for *local* benchmarking;
> starting it would exhaust the 2 GB `t3.small`. Production uses **pgvector** (the
> `db` service), which is the default backend.

> The `-f docker-compose.yml -f compose.prod.yml` overlay sets `RAG_PROVIDER=bedrock`
> (committed in `compose.prod.yml`), so you don't write it into `.env`. Bedrock
> authenticates via the instance IAM role — no AWS key on the box.

> Passing explicit `-f` files also **bypasses `docker-compose.override.yml`** — a
> local-dev-only file Compose auto-loads for a *bare* `docker compose up`. It bakes the
> `fastembed` embedder into the image and loads `.env` into the container (for
> `RAG_PROVIDER=anthropic` locally). Production skips it, so the image stays lean and
> needs no `ANTHROPIC_API_KEY`.

> **Pin the secrets** so they're identical every deploy: `export DEBUG_API_KEY=...`,
> `export ADMIN_API_KEY=...`, `export JWT_SECRET=...` before running. Changing
> `JWT_SECRET` invalidates everyone's existing login tokens.

> Only committed code ships (`git archive HEAD`). Commit your changes first, or the
> deploy won't include them.

### Test (any deployment)

The chat routes require a login token. Full round-trip in **Git Bash**:

```bash
IP=<PUBLIC_IP>
BASE=http://$IP:8000

curl $BASE/health                                  # -> {"status":"ok"}

# 1) register a user (use /auth/login instead if it already exists) -> returns a token:
curl -s -X POST $BASE/auth/register -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"password123"}'

# 2) copy the access_token from step 1 into TOKEN, then create a conversation:
TOKEN=<access_token-from-step-1>
curl -s -X POST $BASE/conversations -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"title":"smoke"}'

# 3) post a question to that conversation id (this calls Bedrock):
curl -s -X POST $BASE/conversations/<ID-FROM-STEP-2>/messages \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"question":"hello, what can you do?"}'
```

A `200` with an `answer` confirms the whole pipeline (auth → retrieve → Bedrock →
persist) works. `sources` stays empty until you ingest documents (see below). If
step 3 errors, Bedrock model access probably isn't granted yet (step 1b).

> **Prefer clicking?** Open `http://<PUBLIC_IP>:8000/docs` — register/login there,
> click **Authorize** (lock icon), paste the token, then call the conversation
> endpoints from the browser.

> **There is no front page at `http://<IP>:8000/` (it's a 404).** Only the backend
> API is deployed. To use the styled React UI, run it locally pointed at this server:
> ```powershell
> cd frontend
> "VITE_API_URL=http://<PUBLIC_IP>:8000" | Out-File -Encoding ascii .env.local
> npm install
> npm run dev        # opens http://localhost:5173
> ```

> **Knowledge base starts empty**, so answers cite nothing until you load documents.
> Ingest with `POST /admin/ingest` (multipart `file` + header `X-Admin-Key: <your
> ADMIN_API_KEY>`), or run `scripts/ingest_bulk.py` on the box against the compose
> Postgres.

---

## Reference

**Endpoints:**
- `/health`, `/docs` — open.
- **Auth:** `POST /auth/register`, `POST /auth/login` (return a bearer token),
  `GET /auth/me`.
- **Chat** (all require `Authorization: Bearer <token>`): `/conversations` (create,
  list), `/conversations/{id}` (rename, delete), `/conversations/{id}/messages`
  (`+/stream`), `/conversations/{id}/documents`.
- **Admin:** `/admin/ingest`, `/admin/ui` (need `X-Admin-Key`).
- **Debug:** `/debug/retrieve`, `/debug/chunks`, `/debug-ui` (need `X-Debug-Key`).
- There is **no** route at `/`.

**Cost:** the `t3.small` bills ~$0.50/day while running. `cdk destroy` stops it;
stopping the instance pauses compute while keeping data (see [Tear down](#7-tear-down)).

**Security:** chat is behind email login but served over plain HTTP (fine for a
demo). `/admin/*` and `/debug/*` require their keys and are disabled (404) unless the
key is configured. For production: add HTTPS, scope IAM down, and set a strong,
pinned `JWT_SECRET`.
