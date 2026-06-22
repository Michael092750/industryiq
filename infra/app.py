#!/usr/bin/env python3
"""CDK app entry point for IndustryIQ's infrastructure."""

import aws_cdk as cdk
from industryiq_infra.industryiq_stack import IndustryIqStack

app = cdk.App()

# SSH is locked to this IP. Pass at deploy time: cdk deploy -c my_ip=1.2.3.4
my_ip = app.node.try_get_context("my_ip") or "0.0.0.0"

IndustryIqStack(
    app,
    # Keep the original CloudFormation stack name so an already-deployed stack
    # updates in place instead of CDK creating a new one.
    "RagprojectStack",
    my_ip=my_ip,
    env=cdk.Environment(account="646167486245", region="us-east-1"),
)

app.synth()
