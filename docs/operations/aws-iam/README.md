# AWS IAM least-privilege policy for `prsm-deployer`

Companion to sprint 403's IAM-cleanup runbook. This directory holds the
canonical least-privilege policy that replaces `AmazonEC2FullAccess` on the
`prsm-deployer` IAM user.

## When to apply

Sprint 403 created `prsm-deployer` with the broad managed policy
`AmazonEC2FullAccess` so the IAM cleanup could be unblocked quickly. The
broad scope was deliberately temporary — this directory closes that loop with
a tight custom policy.

## What's covered

`prsm-deployer-policy.json` grants exactly the EC2 actions used across the
sprint 388-411 multi-cloud bootstrap-fleet workflows:

- **DescribeAllReadOnly**: `ec2:Describe*` + `ec2:Get*` for inventory
  (instances, regions, AMIs, security groups, key pairs, VPCs)
- **InstanceLifecycle**: launch, start, stop, terminate, reboot
- **Networking**: create + delete security groups; authorize + revoke ingress
  and egress rules
- **Keypairs**: import (used by the bootstrap-eu/apac deploys), create, delete
- **Tags**: create + delete EC2 resource tags
- **STSReadOnly**: `sts:GetCallerIdentity` — needed for the `aws sts
  get-caller-identity` sanity-check in deploy scripts

## What's deliberately omitted

- **IAM actions**: prsm-deployer cannot modify IAM users / roles / policies.
  Root-level IAM management requires a root-console session.
- **S3, RDS, Lambda, etc.**: not used by the bootstrap-operator workflow.
- **EC2 image creation / modification**: not used.
- **VPC create/delete**: today we use the default VPC. If multi-VPC topology
  becomes operator-relevant later, add `ec2:CreateVpc` etc. then.

## How to apply (one-time, AWS console)

1. Sign in to the AWS console **as root** (or as an existing IAM user with
   `iam:CreatePolicy` + `iam:AttachUserPolicy` + `iam:DetachUserPolicy` — root
   is simplest).
2. **IAM** → **Policies** → **Create policy** → **JSON** tab
3. Paste the contents of `prsm-deployer-policy.json` → **Next**
4. Name: `PRSMDeployerLeastPrivilege` → **Create policy**
5. **IAM** → **Users** → click `prsm-deployer`
6. **Permissions** tab → find the `AmazonEC2FullAccess` row → **Remove**
7. **Add permissions** → **Attach policies directly** → search
   `PRSMDeployerLeastPrivilege` → check the box → **Next** → **Add permissions**

## Verification after apply

From your laptop:

```bash
aws sts get-caller-identity --output text
# Should still return: prsm-deployer ARN

# Sanity-test EC2 read access (should succeed):
aws ec2 describe-instances --region eu-central-1 \
  --filters Name=tag:Name,Values=bootstrap-eu --output table

# Sanity-test that IAM is now denied (should error):
aws iam list-users 2>&1 | head -2
# Expected: AccessDenied
```

If the EC2 describe works AND IAM is denied, the migration succeeded.

## Rolling back if needed

If a future operator workflow needs an EC2 action this policy doesn't cover,
either:
- Add the specific action to `prsm-deployer-policy.json` + update the live
  policy via **IAM** → **Policies** → `PRSMDeployerLeastPrivilege` → **Edit**,
- Or temporarily re-attach `AmazonEC2FullAccess` for the duration of the new
  workflow, then revert.

## Companion: MFA on root

Distinct from the policy hardening, the root account should have MFA enabled
since it's the only fallback to the IAM-managed credentials. Setup:

1. **AWS console as root** → top-right account name → **Security credentials**
2. Scroll to **Multi-factor authentication (MFA)** → **Assign MFA device**
3. Pick a device type:
   - **Authenticator app** (Google Authenticator, 1Password, Authy) — easiest,
     no hardware
   - **Hardware key** (YubiKey, Titan) — most secure, requires physical device
   - **Passkey** — modern, browser-bound
4. Follow the pairing flow.

After MFA is enabled, root password alone is no longer sufficient — any future
root sign-in (e.g., to delete prsm-deployer if rotating credentials) requires
the second factor.

## Cross-references

- `project_aws_iam_cleanup_pending.md` (now removed) — sprint 403 ran the
  initial root-key-deletion phase of this hardening
- Sprint 412 (this commit) closes sprint 403's remaining follow-on
