# PRSM Enterprise Operator Runbook

Reference for deploying the §7 Enterprise Confidentiality
Mode + federated inference stack. Covers initial bringup,
post-deploy verification, common operational tasks, and
troubleshooting.

Audience: an operator standing up PRSM for the first time
at a new enterprise customer.

The CLI tool referenced throughout is invoked via:

```bash
python -m prsm.enterprise.bringup_cli <subcommand>
```

inside the deployed image, or directly from a checkout
of the repo for local development.

---

## 1. First-time bringup

### 1a — Generate a starter config

```bash
python -m prsm.enterprise.bringup_cli generate \
    --base-dir /var/lib/prsm \
    > /etc/prsm/enterprise.env
```

`generate` produces a complete `.env` file template with:
- Fresh keypairs for every required `PRSM_*_PRIVKEY` env
  var (X25519 for transport vars; Ed25519 for signing
  vars)
- Persistence-dir env vars pointed at sensible defaults
  under `--base-dir`
- Commented-out lines for every optional var so you can
  see the inventory without them being active

The output validates out of the box — copy-paste it as
your starting `.env`, then fill in vendor-specific values
(RPC URL, network identifier, treasury addresses) as
needed.

### 1b — Confirm config is well-formed

```bash
export $(cat /etc/prsm/enterprise.env | xargs)
python -m prsm.enterprise.bringup_cli status
```

`status` prints a per-var inventory and returns rc=1 if
any required var is missing or malformed. Use this in CI
to fail-fast before a broken deploy reaches production.

### 1c — Boot the container

```bash
docker compose -f deploy/enterprise-demo/docker-compose.yml \
    --env-file /etc/prsm/enterprise.env \
    up -d
```

The compose file mounts `${PRSM_DEMO_DATA_DIR:-./prsm-demo-data}`
to `/var/lib/prsm` inside the container. Override
`PRSM_DEMO_DATA_DIR` to point at your durable storage
volume.

### 1d — Verify the boot

Two complementary checks:

```bash
# Static: are env vars valid?
docker compose exec prsm-enterprise-demo \
    python -m prsm.enterprise.bringup_cli status

# Dynamic: do the configured subsystems actually wire?
docker compose exec prsm-enterprise-demo \
    python -m prsm.enterprise.bringup_cli health
```

`health` confirms:
- Persistence dirs exist + are writable
- Each `PRSM_*_PRIVKEY` env var decodes to a valid 32-byte
  Ed25519/X25519 keypair
- Each subsystem the env vars enable can instantiate its
  orchestrator without raising

If both return rc=0, the deployment is ready.

### 1e — Run the end-to-end demo

```bash
docker compose exec prsm-enterprise-demo \
    python -m prsm.enterprise.bringup_cli demo
```

The `demo` subcommand exercises both the §7 enterprise
FL flow and the federated pipeline inference flow against
an in-process deployment, printing each step. Successful
output ends with:

```
✓ All demos passed — PRSM enterprise stack verified
  end-to-end.
```

This is the post-deploy proof-of-life. Any production
node should pass `demo` before being placed in front of a
real customer.

---

## 2. Persistence + state

The deployed image mounts a single volume:

| Container path     | Purpose                              |
|--------------------|--------------------------------------|
| `/var/lib/prsm`    | All persistence (FL jobs, pipeline   |
|                    | receipts, $CORP capability ledger,   |
|                    | disclosure intake, incident          |
|                    | response, upgrade orchestrator)      |

Subdirs of this volume are controlled by env vars:

- `PRSM_FEDERATED_LEARNING_DIR` → FL orchestrator state
- `PRSM_PIPELINE_ORCHESTRATOR_DIR` → pipeline orchestrator
  state
- `PRSM_CORP_CAPABILITY_DIR` → $CORP issuer + redemption
  ledger
- `PRSM_DISCLOSURE_INTAKE_DIR` → responsible-disclosure
  records
- `PRSM_INCIDENT_RESPONSE_DIR` → §14 incident lifecycle
- `PRSM_UPGRADE_ORCHESTRATOR_DIR` → UUPS upgrade proposals

**Always mount this volume from durable storage** — fresh
ephemeral state on every restart means lost capability
ledgers, lost incident records, lost upgrade history.

---

## 3. Key rotation

Keypair env vars hold raw 32-byte material; rotating a key
means generating a fresh value and restarting the affected
process. The container does not currently support
zero-downtime rotation.

```bash
# Generate a fresh Ed25519 privkey for the FL worker:
python -c "
from prsm.enterprise.federated_learning import generate_worker_keypair
priv, pub = generate_worker_keypair()
print(f'privkey: {priv}')
print(f'pubkey:  {pub}')
"

# Update /etc/prsm/enterprise.env with the new privkey
# Restart the container
docker compose -f deploy/enterprise-demo/docker-compose.yml \
    --env-file /etc/prsm/enterprise.env \
    up -d --force-recreate prsm-enterprise-demo

# Distribute the new pubkey to anyone verifying signatures
# from this node (orchestrators, downstream verifiers)
```

For the pipeline orchestrator key, the new pubkey must
also be propagated to anyone holding existing pipeline
receipts who wants to verify new ones. Receipts signed
under the old key remain verifiable with the old pubkey;
the rotation only affects new receipts.

---

## 4. Common operational tasks

### Inspect FL job state

```bash
ls -la /var/lib/prsm/fl/
cat /var/lib/prsm/fl/job-*.json | python -m json.tool
```

Job state is human-readable JSON. Aggregated gradients
are base64-encoded inside `round-*.json`.

### Inspect pipeline receipts

```bash
ls /var/lib/prsm/pipeline/round-*.json
```

Each round file contains the verifiable
`PipelineInferenceReceipt`. To verify externally:

```python
from prsm.compute.inference.pipeline_receipt import (
    PipelineInferenceReceipt, verify_pipeline_receipt,
)
import json

with open("/var/lib/prsm/pipeline/round-<id>.json") as f:
    round_data = json.load(f)
receipt = PipelineInferenceReceipt.from_dict(
    round_data["receipt"],
)
result = verify_pipeline_receipt(
    receipt,
    orchestrator_pubkey_b64="<the orchestrator's pubkey>",
)
print(result.ok, result.diagnostic)
```

### Check $CORP capability ledger

```bash
cat /var/lib/prsm/corp/issuers.json | python -m json.tool
cat /var/lib/prsm/corp/consumed.json | python -m json.tool
ls /var/lib/prsm/corp/ledger/
```

`issuers.json` lists every registered enterprise issuer;
`consumed.json` tracks per-capability redemption counts;
`ledger/<capability_id>.json` is the audit trail per
capability.

---

## 5. Troubleshooting

### `status` shows MISSING for a required var

Run `generate` to see what a complete config looks like.
Required vars (sprint 318 EnvVarSpec catalog):

- `PRSM_FEDERATED_WORKER_PRIVKEY`
- `PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY`
- `PRSM_FEDERATED_LEARNING_DIR`
- `PRSM_PIPELINE_ORCHESTRATOR_DIR`

Optional vars cover follow-on subsystems (disclosure,
incident, upgrade, transport encryption, etc.).

### `health` reports a directory not writable

Most common cause: the volume mount has wrong perms.
Inside the container the process runs as a non-root user
(image default) — make sure the host volume has 0o755
(directory) and is owned by a UID/GID the container's
user can write.

```bash
# Fix typical "permission denied" on a fresh mount:
chmod -R 0755 ./prsm-demo-data
chown -R 1000:1000 ./prsm-demo-data  # container UID
```

### `health` reports a keypair malformed

Most likely the env var was set with a quoting issue
(e.g., the base64 padding `=` got escaped or stripped).
Generate a fresh one:

```bash
python -c "
from prsm.enterprise.federated_learning import generate_worker_keypair
print(generate_worker_keypair()[0])
"
```

Confirm the value is exactly 44 chars long (32 raw bytes
base64-encoded with `==` padding). Update the env file
and restart.

### `demo` fails inside the container

The demo creates everything fresh per run — if it fails,
the issue is in the container itself, not the persisted
state. Common causes:

- Out of memory: torch CPU wheels need ~512 MB RSS at
  minimum
- Read-only filesystem: the demo writes to `/tmp/` inside
  the container; if your runtime is `--read-only`, mount
  a writable `/tmp` tmpfs

Run `docker compose logs prsm-enterprise-demo` for the
full traceback.

---

## 6. Metrics + structured logging

### Prometheus scrape

The deployed image exposes
`GET /admin/enterprise/metrics` in Prometheus text
exposition format. Scrape with a standard Prometheus
config:

```yaml
scrape_configs:
  - job_name: 'prsm-enterprise'
    metrics_path: /admin/enterprise/metrics
    static_configs:
      - targets: ['prsm-enterprise-demo:8000']
```

Exposed metrics (sprint 318d catalog):

- `fl_jobs_proposed_total` — counter
- `fl_rounds_aggregated_total` — counter
- `fl_worker_updates_accepted_total` — counter
- `fl_worker_updates_rejected_total` — counter (bad sig
  / unregistered worker / wrong round / duplicate)
- `pipeline_inference_jobs_proposed_total` — counter
- `pipeline_inference_completed_total` — counter
- `pipeline_inference_failed_total` — counter
- `corp_capabilities_redeemed_total` — counter
- `corp_capabilities_rejected_total` — counter
- `content_uploads_encrypted_total` — counter
- `incident_opened_total` — counter
- `fl_jobs_pending` — gauge (live snapshot)
- `pipeline_jobs_pending` — gauge (live snapshot)

For one-shot debug without setting up Prometheus:

```bash
python -m prsm.enterprise.bringup_cli metrics-snapshot
```

### Structured JSON logging

For production deploys piping logs to an aggregator
(Loki / Datadog / ELK / OTEL), enable JSON logging at
process startup:

```python
from prsm.enterprise.structured_logging import (
    configure_json_logging,
)
configure_json_logging()
```

This installs a JSON formatter on the root logger that
emits one parseable line per record with standard ops
fields (timestamp, level, logger, msg) plus any `extra=`
kwargs the calling code passed. Loggers calling
`logger.info("event", extra={"job_id": "..."})` get the
job_id as a top-level structured field.

## 7. CLI reference

| Subcommand              | Purpose                            |
|-------------------------|------------------------------------|
| `generate`              | Emit a complete `.env` template    |
|                         | with fresh keypairs.               |
| `status`                | Print env-var inventory; rc=1 if   |
|                         | required vars missing/malformed.   |
| `health`                | Run dynamic runtime checks; rc=1   |
|                         | if any subsystem won't wire.       |
| `metrics-snapshot`      | Print current metrics in           |
|                         | Prometheus text format             |
|                         | (one-shot debug).                  |
| `demo`                  | Run the end-to-end §7 FL +         |
|                         | pipeline inference demo;           |
|                         | post-deploy proof-of-life.         |

`status` + `health` are CI-friendly (rc=0 = healthy,
rc=1 = something needs fixing). Both are idempotent and
side-effect-free apart from creating missing persistence
directories under env-configured paths.

---

## 8. Related sprints

The deployment stack was assembled in this order:

- **Sprint 318** — `EnterpriseConfig`, env-var catalog,
  CLI status + generate
- **Sprint 318b** — end-to-end demo runnable
- **Sprint 318a** — Docker artifacts + live boot
  validation
- **Sprint 318c** — health subcommand + this runbook
- **Sprint 318d** — metrics registry + Prometheus
  endpoint + structured JSON logging + metrics-snapshot
  CLI

See `prsm/enterprise/bringup.py` for the canonical env-var
catalog; each spec carries the sprint of origin so you can
trace what each variable controls.
