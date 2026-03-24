# Phase 3 — Web-Based Node Onboarding UI

## Overview

Phase 1 fixed the P0 security/database bugs. Phase 2 cleared all test errors and added
IPFS auto-start. This phase delivers the remaining "Infrastructure Usability" item from
the original 5-phase plan: **a browser-accessible onboarding UI** that lets non-technical
users set up and configure a PRSM node without touching the CLI or editing config files.

The onboarding flow covers:
1. **Welcome** — confirm prerequisites (Python, dependencies, port availability)
2. **API Keys** — collect Anthropic / OpenAI keys, validate them with a live test call
3. **Backend Selection** — set primary + fallback chain based on what keys were provided
4. **Network Config** — choose P2P port, bootstrap node, confirm IPFS status
5. **Node Identity** — generate or import Ed25519 key pair
6. **Review & Launch** — write final config to `config/node_config.json`, start the node

**Delivery vehicle:** FastAPI router mounted at `/onboarding/` in the existing PRSM API
app. Responses are JSON for API clients AND fully-functional HTML (rendered via Jinja2
templates) for browser users. No new package dependencies required — Jinja2 ships with
FastAPI.

**Files touched:**
- `prsm/interface/api/onboarding_router.py` (new)
- `prsm/interface/api/templates/` (new directory — 6 HTML templates)
- `prsm/interface/api/router_registry.py` (add router registration)
- `prsm/interface/api/app_factory.py` (mount static files for template assets)
- `config/node_config.json.template` (new — canonical config template)

**Exit criterion:** `prsm node start` prints a URL like
`http://localhost:8080/onboarding/` that the user can open in a browser; navigating
that URL guides them through all 6 steps and writes a valid config; running
`pytest tests/test_onboarding_api.py -v` reports 0 failed.

---

## Step 1 — Read Existing Router Registration

Before writing any new code, read:
1. `prsm/interface/api/router_registry.py` — see exactly how routers are registered
2. `prsm/interface/api/app_factory.py` — see how the app is constructed
3. `prsm/interface/api/health_api.py` — use as a clean pattern for a simple router

Then determine:
- Does `router_registry.py` have a list or function to register routers? Use that.
- Is there already a `/static/` mount? If not, add one.

---

## Step 2 — Create `config/node_config.json.template`

This is the source of truth for what config the onboarding flow produces:

```json
{
  "primary_backend": "mock",
  "fallback_chain": ["mock"],
  "anthropic_api_key": null,
  "openai_api_key": null,
  "timeout_seconds": 120,
  "max_retries": 3,
  "p2p_port": 8765,
  "bootstrap_nodes": ["wss://bootstrap1.prsm-network.com:8765"],
  "node_identity_path": "config/node_identity.json",
  "ipfs_auto_start": true,
  "api_port": 8080,
  "log_level": "INFO"
}
```

---

## Step 3 — Create `prsm/interface/api/onboarding_router.py`

This is the main deliverable. Structure:

```python
"""
Node Onboarding API
===================

Browser-accessible wizard for first-time node setup.

Endpoints:
  GET  /onboarding/          → step 1: welcome + prerequisites check
  GET  /onboarding/api-keys  → step 2: API key form
  POST /onboarding/api-keys  → validate keys, advance to step 3
  GET  /onboarding/backend   → step 3: backend selection
  POST /onboarding/backend   → save backend config, advance to step 4
  GET  /onboarding/network   → step 4: network config
  POST /onboarding/network   → save network config, advance to step 5
  GET  /onboarding/identity  → step 5: node identity
  POST /onboarding/identity  → generate/save identity, advance to step 6
  GET  /onboarding/launch    → step 6: review config + launch
  POST /onboarding/launch    → write config to disk, return success

All POST endpoints accept both JSON (for API clients) and HTML form data
(for browser users). Responses are:
  - JSON when Accept: application/json or Content-Type: application/json
  - HTML redirect/render otherwise
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

router = APIRouter(prefix="/onboarding", tags=["onboarding"])

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

CONFIG_TEMPLATE_PATH = Path("config/node_config.json.template")
CONFIG_OUTPUT_PATH   = Path("config/node_config.json")
```

### Prerequisite Check (Step 1)

`GET /onboarding/` runs these checks and returns results as JSON + renders HTML:

```python
@router.get("/", response_class=HTMLResponse)
async def onboarding_welcome(request: Request):
    checks = await _run_prerequisite_checks()
    if _wants_json(request):
        return JSONResponse(checks)
    return templates.TemplateResponse("onboarding/welcome.html",
                                      {"request": request, "checks": checks})

async def _run_prerequisite_checks() -> Dict[str, Any]:
    import shutil, sys, importlib
    checks = {
        "python_version": {"ok": sys.version_info >= (3, 10), "value": sys.version},
        "ipfs_installed": {"ok": bool(shutil.which("ipfs")), "value": shutil.which("ipfs") or "not found"},
        "anthropic_sdk":  {"ok": importlib.util.find_spec("anthropic") is not None},
        "openai_sdk":     {"ok": importlib.util.find_spec("openai")    is not None},
        "port_8765_free": {"ok": _port_is_free(8765)},
        "port_8080_free": {"ok": _port_is_free(8080)},
    }
    checks["all_ok"] = all(v["ok"] for v in checks.values() if isinstance(v, dict))
    return checks

def _port_is_free(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0
```

### API Key Validation (Step 2)

`POST /onboarding/api-keys` performs a **live validation call** against Anthropic / OpenAI:

```python
class ApiKeyPayload(BaseModel):
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

@router.post("/api-keys")
async def submit_api_keys(request: Request,
                          anthropic_api_key: str = Form(default=""),
                          openai_api_key: str = Form(default="")):
    results = {}

    if anthropic_api_key:
        results["anthropic"] = await _validate_anthropic_key(anthropic_api_key)
    if openai_api_key:
        results["openai"] = await _validate_openai_key(openai_api_key)

    # Store validated keys in session (or just pass forward as hidden form fields)
    if _wants_json(request):
        return JSONResponse({"validation": results, "next": "/onboarding/backend"})
    return RedirectResponse(url="/onboarding/backend", status_code=303)


async def _validate_anthropic_key(key: str) -> Dict[str, Any]:
    """Make a minimal /v1/messages call to confirm the key is valid."""
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=key)
        await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}]
        )
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": str(e)[:120]}


async def _validate_openai_key(key: str) -> Dict[str, Any]:
    """Make a minimal chat completion call to confirm the key is valid."""
    try:
        import openai
        client = openai.AsyncOpenAI(api_key=key)
        await client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=1,
            messages=[{"role": "user", "content": "hi"}]
        )
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": str(e)[:120]}
```

### Backend Selection (Step 3)

`GET /onboarding/backend` returns the available backends based on which keys were
validated in step 2 (passed via query params or session). The form lets users set
`primary_backend` and drag-reorder the `fallback_chain`.

`POST /onboarding/backend` validates the selection and saves to an in-memory
`_pending_config` dict (a module-level dict is fine for single-process use).

### Network Config (Step 4)

`GET /onboarding/network` shows:
- P2P port (default 8765)
- Bootstrap node URL(s)
- IPFS status (from `_ensure_ipfs_available()`)
- API port (default 8080)

`POST /onboarding/network` validates ports are free and saves to `_pending_config`.

### Node Identity (Step 5)

`GET /onboarding/identity` shows existing identity file if present, or a
"Generate new identity" button.

`POST /onboarding/identity` with `action=generate` calls:
```python
from prsm.networking.transport import generate_node_identity
identity = generate_node_identity("user-node")
identity_path = Path("config/node_identity.json")
identity_path.write_text(json.dumps({
    "node_id": identity.node_id,
    "private_key_b64": identity.private_key_b64,
    "public_key_b64": identity.public_key_b64,
}, indent=2))
```

`POST /onboarding/identity` with `action=import` accepts an uploaded JSON file.

### Review & Launch (Step 6)

`GET /onboarding/launch` renders the assembled config (from `_pending_config`)
with a "Confirm & Write Config" button.

`POST /onboarding/launch` writes `config/node_config.json` from `_pending_config`
and returns:
```json
{
  "success": true,
  "config_path": "config/node_config.json",
  "next_step": "Run: prsm node start"
}
```

---

## Step 4 — HTML Templates

Create `prsm/interface/api/templates/onboarding/` with these files:

### `base.html`
Minimal Bootstrap 5.3 CDN layout (no npm required). Just `<link>` and `<script>`
tags pointing to `https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/`.

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PRSM Node Setup — {% block title %}{% endblock %}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        rel="stylesheet">
</head>
<body class="bg-light">
<nav class="navbar navbar-dark bg-dark mb-4">
  <div class="container"><span class="navbar-brand">PRSM Node Setup</span></div>
</nav>
<div class="container pb-5">
  <div class="row justify-content-center">
    <div class="col-lg-8">
      {% block content %}{% endblock %}
    </div>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

### `welcome.html`

Extends `base.html`. Shows a progress bar (step 1/6) and a checklist of
prerequisite check results with ✅ / ⚠️ icons. "Continue" button navigates to
`/onboarding/api-keys`.

### `api_keys.html`

Step 2/6. Two password inputs for Anthropic and OpenAI keys. Both optional. Note:
"Leave blank to use local/mock backend only." "Continue" button submits the form.

### `backend.html`

Step 3/6. Shows detected backends (Anthropic if key provided, OpenAI if key
provided, Local if Ollama found, Mock always available). Radio buttons for primary,
checkbox list for fallback chain.

### `network.html`

Step 4/6. Number inputs for P2P port and API port. Text input for bootstrap URL.
IPFS status card (green if running, yellow if installed-not-running, red if not found).

### `identity.html`

Step 5/6. Shows existing identity file content (node_id truncated) if present.
Two radio buttons: "Generate new" and "Use existing". Upload input appears for
"Use existing".

### `launch.html`

Step 6/6. Read-only JSON code block showing the assembled config. "Write Config
& Finish" button triggers POST. On success: green success alert with
`prsm node start` command displayed.

---

## Step 5 — Register the Router

In `prsm/interface/api/router_registry.py` (or wherever routers are registered),
add:

```python
from prsm.interface.api.onboarding_router import router as onboarding_router
# ... in the registration list:
app.include_router(onboarding_router)
```

Read `router_registry.py` first to find the exact registration pattern.

---

## Step 6 — Wire `prsm node start` to Print the URL

In `prsm/node/node.py` or `prsm/cli.py`, find where the node prints its startup
message and add:

```python
logger.info(
    "Node onboarding UI available",
    url=f"http://localhost:{api_port}/onboarding/"
)
```

---

## Step 7 — Write Tests

Create `tests/test_onboarding_api.py`:

```python
"""Tests for the node onboarding wizard API."""
import pytest
from fastapi.testclient import TestClient
from prsm.interface.api.main import app

client = TestClient(app)


def test_welcome_returns_200():
    resp = client.get("/onboarding/")
    assert resp.status_code == 200


def test_welcome_json_returns_checks():
    resp = client.get("/onboarding/", headers={"Accept": "application/json"})
    assert resp.status_code == 200
    data = resp.json()
    assert "python_version" in data
    assert "all_ok" in data


def test_api_keys_get_returns_200():
    resp = client.get("/onboarding/api-keys")
    assert resp.status_code == 200


def test_api_keys_post_without_keys_redirects():
    resp = client.post("/onboarding/api-keys",
                       data={"anthropic_api_key": "", "openai_api_key": ""},
                       follow_redirects=False)
    assert resp.status_code in (200, 302, 303)


def test_backend_get_returns_200():
    resp = client.get("/onboarding/backend")
    assert resp.status_code == 200


def test_network_get_returns_200():
    resp = client.get("/onboarding/network")
    assert resp.status_code == 200


def test_identity_get_returns_200():
    resp = client.get("/onboarding/identity")
    assert resp.status_code == 200


def test_launch_get_returns_200():
    resp = client.get("/onboarding/launch")
    assert resp.status_code == 200


def test_launch_post_writes_config(tmp_path, monkeypatch):
    """Test that POST /onboarding/launch writes config/node_config.json."""
    import prsm.interface.api.onboarding_router as m
    monkeypatch.setattr(m, "CONFIG_OUTPUT_PATH", tmp_path / "node_config.json")

    resp = client.post("/onboarding/launch",
                       json={"confirm": True},
                       headers={"Accept": "application/json"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert (tmp_path / "node_config.json").exists()
```

---

## Implementation Notes

- **No new package dependencies.** Jinja2 is already installed with FastAPI.
  Bootstrap via CDN — no npm/webpack.
- **In-memory `_pending_config`:** A module-level dict is fine since onboarding
  is a single-session, single-user flow. No need for Redis/cookies.
- **Hidden form fields for state.** Pass API key validity forward as hidden
  `<input type="hidden">` fields in the HTML form. Don't store raw keys beyond
  the active session (they should be written to `node_config.json` at launch step).
- **API key masking in config.** When writing `node_config.json`, write only the
  first 8 characters + `***` for any key with > 8 chars (same pattern as
  `BackendConfig.to_dict()`). The actual key is passed via environment variable
  at runtime, not stored in the config file.
  Actually — **store the full key** in `node_config.json` since this file is
  local-only (not committed). The user generated it, they own it.
- **Template auto-reload.** Jinja2 `auto_reload=True` is fine for development.
  In production, templates are baked in — set `auto_reload=False` if desired.
- **`_wants_json()` helper:**
  ```python
  def _wants_json(request: Request) -> bool:
      return "application/json" in request.headers.get("accept", "")
  ```

---

## Verification

After implementing:

```bash
# Run onboarding-specific tests
pytest tests/test_onboarding_api.py -v

# Full suite still passes
pytest --ignore=tests/benchmarks -q --timeout=60

# Manual smoke test
prsm node start --no-dashboard &
curl http://localhost:8080/onboarding/ -H "Accept: application/json" | python -m json.tool
```

Expected: 8+ new tests pass; full suite remains ≥3438 passing; JSON endpoint
returns a valid `checks` dict with `all_ok` key.

---

## Implementation Completed — 2026-03-24

### Summary

Phase 3 has been fully implemented. The web-based node onboarding UI is now available
at `/onboarding/` and provides a complete 6-step wizard for configuring a PRSM node.

### Files Created

1. **`config/node_config.json.template`** — Canonical config template with default values
2. **`prsm/interface/api/onboarding_router.py`** — Main onboarding router with all 6 steps:
   - Step 1: Welcome/prerequisites check
   - Step 2: API key collection and validation
   - Step 3: Backend selection (Anthropic/OpenAI/Ollama/Mock)
   - Step 4: Network configuration (ports, bootstrap nodes, IPFS)
   - Step 5: Node identity generation/import
   - Step 6: Review and launch (writes config file)
3. **`prsm/interface/api/templates/onboarding/`** — HTML templates:
   - `base.html` — Bootstrap 5.3 layout with progress bar
   - `welcome.html` — Prerequisites checklist
   - `api_keys.html` — API key input forms
   - `backend.html` — Backend selection radio buttons
   - `network.html` — Port and IPFS configuration
   - `identity.html` — Identity generation/import
   - `launch.html` — Config review and submit
   - `success.html` — Completion confirmation

### Files Modified

1. **`prsm/interface/api/router_registry.py`** — Added onboarding router registration
2. **`prsm/node/node.py`** — Added onboarding URL to startup log message

### Test Results

- **24 new onboarding tests** in `tests/test_onboarding_api.py`
- All 24 tests pass
- Full test suite: **3482 passed**, 81 skipped, 4 xfailed
- No regressions introduced

### Key Features

- **Dual-mode responses**: JSON for API clients, HTML for browsers
- **Live API key validation**: Makes test calls to Anthropic/OpenAI to verify keys
- **Automatic IPFS detection**: Checks if IPFS daemon is running
- **Ed25519 identity generation**: Uses existing `prsm.node.identity` module
- **Bootstrap CDN**: No npm/webpack required, Bootstrap 5.3 via CDN
- **In-memory session**: `_pending_config` dict for single-session onboarding

### Exit Criteria Met

✅ `prsm node start` prints onboarding URL: `http://localhost:8080/onboarding/`
✅ Navigating to `/onboarding/` guides users through all 6 steps
✅ Config written to `config/node_config.json`
✅ All tests pass: `pytest tests/test_onboarding_api.py -v` → 24 passed
