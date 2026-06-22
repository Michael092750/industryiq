# infra — IndustryIQ infrastructure (AWS CDK)

Infrastructure as code for the IndustryIQ deployment. Self-contained: its own
virtualenv and dependencies, separate from the application package.

## Setup (once)

```powershell
cd infra
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cdk bootstrap          # one-time per account/region
```

## Deploy / inspect / destroy

```powershell
# Lock SSH to your current IP:
$myip = (Invoke-WebRequest https://checkip.amazonaws.com -UseBasicParsing).Content.Trim()

cdk diff   -c my_ip=$myip      # preview changes
cdk deploy -c my_ip=$myip      # create/update the stack
cdk destroy                    # tear everything down
```

`cdk deploy` outputs the instance's public IP and the command to fetch the SSH
private key from SSM Parameter Store. Application code is deployed separately
(see the repo's deploy step), keeping infra and app lifecycles independent.
