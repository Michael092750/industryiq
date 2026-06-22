"""The IndustryIQ deployment, described as code.

Mirrors the manual EC2 + Docker Compose deployment: a single VM in the default
VPC, with a Bedrock IAM role, a firewall, and Docker pre-installed via user-data.
"""

from aws_cdk import CfnOutput, Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from constructs import Construct


class IndustryIqStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, *, my_ip: str, **kwargs: object
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Use the account's default VPC (public subnets, no NAT cost).
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        # Firewall: SSH only from your IP, the app port open to the world.
        sg = ec2.SecurityGroup(
            self, "AppSg", vpc=vpc, description="IndustryIQ app", allow_all_outbound=True
        )
        sg.add_ingress_rule(ec2.Peer.ipv4(f"{my_ip}/32"), ec2.Port.tcp(22), "SSH from my IP")
        sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(8000), "App from anywhere")

        # The VM's identity: may call Bedrock; SSM core enables keyless console access.
        role = iam.Role(self, "AppRole", assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"))
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonBedrockFullAccess")
        )
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")
        )

        # CDK creates the SSH key pair; private key lands in SSM Parameter Store.
        key_pair = ec2.KeyPair(self, "KeyPair", key_pair_name="ragproject-cdk-key")

        # Boot script: install Docker + the compose and buildx plugins.
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "dnf install -y docker git",
            "systemctl enable --now docker",
            "usermod -aG docker ec2-user",
            "mkdir -p /usr/local/lib/docker/cli-plugins",
            "curl -SL https://github.com/docker/compose/releases/latest/download/"
            "docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose",
            "chmod +x /usr/local/lib/docker/cli-plugins/docker-compose",
            "BX=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest "
            "| grep '\"tag_name\"' | head -1 | cut -d '\"' -f4)",
            "curl -sSL https://github.com/docker/buildx/releases/download/$BX/"
            "buildx-$BX.linux-amd64 -o /usr/local/lib/docker/cli-plugins/docker-buildx",
            "chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx",
        )

        instance = ec2.Instance(
            self,
            "AppInstance",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_type=ec2.InstanceType("t3.small"),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(),
            security_group=sg,
            role=role,
            key_pair=key_pair,
            user_data=user_data,
            block_devices=[
                ec2.BlockDevice(device_name="/dev/xvda", volume=ec2.BlockDeviceVolume.ebs(20))
            ],
        )

        # IMDS hop limit 2 so the Docker container can read the instance-role creds.
        cfn_instance = instance.node.default_child
        cfn_instance.add_property_override(  # type: ignore[union-attr]
            "MetadataOptions",
            {"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        )

        CfnOutput(self, "PublicIp", value=instance.instance_public_ip)
        CfnOutput(self, "InstanceId", value=instance.instance_id)
        CfnOutput(
            self,
            "GetSshKeyCommand",
            value=(
                f"aws ssm get-parameter --name /ec2/keypair/{key_pair.key_pair_id} "
                "--with-decryption --query Parameter.Value --output text"
            ),
        )
