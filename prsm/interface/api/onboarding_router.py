"""
Node Onboarding API
===================

Browser-accessible wizard for first-time node setup.

Endpoints:
  GET  /onboarding/          -> step 1: welcome + prerequisites check
  GET  /onboarding/api-keys  -> step 2: API key form
  POST /onboarding/api-keys  -> validate keys, advance to step 3
  GET  /onboarding/backend   -> step 3: backend selection
  POST /onboarding/backend   -> save backend config, advance to step 4
  GET  /onboarding/network   -> step 4: network config
  POST /onboarding/network   -> save network config, advance to step 5
  GET  /onboarding/identity  -> step 5: node identity
  POST /onboarding/identity  -> generate/save identity, advance to step 6
  GET  /onboarding/launch    -> step 6: review config + launch
  POST /onboarding/launch    -> write config to disk, return success

All POST endpoints accept both JSON (for API clients) and HTML form data
(for browser users). Responses are:
  - JSON when Accept: application/json or Content-Type: application/json
  - HTML redirect/render otherwise
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json
import shutil
import sys
import importlib
import socket

from fastapi import APIRouter, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/onboarding", tags=["onboarding"])

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

CONFIG_TEMPLATE_PATH = Path("config/node_config.json.template")
CONFIG_OUTPUT_PATH = Path("config/node_config.json")
IDENTITY_OUTPUT_PATH = Path("config/node_identity.json")

# In-memory pending config for the onboarding session
_pending_config: Dict[str, Any] = {}


def _wants_json(request: Request) -> bool:
    """Check if the client wants a JSON response."""
    accept = request.headers.get("accept", "")
    content_type = request.headers.get("content-type", "")
    return "application/json" in accept or "application/json" in content_type


def _port_is_free(port: int) -> bool:
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


async def _run_prerequisite_checks() -> Dict[str, Any]:
    """Run prerequisite checks for the onboarding wizard."""
    checks = {
        "python_version": {
            "ok": sys.version_info >= (3, 10),
            "value": sys.version.split()[0],
            "label": f"Python {sys.version.split()[0]}",
        },
        "ipfs_installed": {
            "ok": bool(shutil.which("ipfs")),
            "value": shutil.which("ipfs") or "not found",
            "label": "IPFS CLI",
        },
        "anthropic_sdk": {
            "ok": importlib.util.find_spec("anthropic") is not None,
            "value": "installed" if importlib.util.find_spec("anthropic") else "not installed",
            "label": "Anthropic SDK",
        },
        "openai_sdk": {
            "ok": importlib.util.find_spec("openai") is not None,
            "value": "installed" if importlib.util.find_spec("openai") else "not installed",
            "label": "OpenAI SDK",
        },
        "port_8765_free": {
            "ok": _port_is_free(8765),
            "value": "free" if _port_is_free(8765) else "in use",
            "label": "P2P Port 8765",
        },
        "port_8080_free": {
            "ok": _port_is_free(8080),
            "value": "free" if _port_is_free(8080) else "in use",
            "label": "API Port 8080",
        },
    }
    checks["all_ok"] = all(v["ok"] for v in checks.values() if isinstance(v, dict))
    return checks


# =============================================================================
# Step 1: Welcome / Prerequisites
# =============================================================================

@router.get("/", response_class=HTMLResponse)
async def onboarding_welcome(request: Request):
    """Step 1: Welcome page with prerequisite checks."""
    checks = await _run_prerequisite_checks()
    if _wants_json(request):
        return JSONResponse(checks)
    return templates.TemplateResponse(request, "onboarding/welcome.html", {"request": request, "checks": checks, "step": 1})
    )


# =============================================================================
# Step 2: API Keys
# =============================================================================

class ApiKeyPayload(BaseModel):
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None


@router.get("/api-keys", response_class=HTMLResponse)
async def api_keys_form(request: Request):
    """Step 2: API key input form."""
    if _wants_json(request):
        return JSONResponse({"step": 2, "message": "Submit API keys via POST"})
    return templates.TemplateResponse(request, "onboarding/api_keys.html", {"request": request, "step": 2})
    )


@router.post("/api-keys")
async def submit_api_keys(
    request: Request,
    anthropic_api_key: str = Form(default=""),
    openai_api_key: str = Form(default=""),
):
    """Validate API keys and store them in pending config."""
    global _pending_config
    results = {}

    # Validate Anthropic key if provided
    if anthropic_api_key and anthropic_api_key.strip():
        results["anthropic"] = await _validate_anthropic_key(anthropic_api_key.strip())
        if results["anthropic"]["valid"]:
            _pending_config["anthropic_api_key"] = anthropic_api_key.strip()

    # Validate OpenAI key if provided
    if openai_api_key and openai_api_key.strip():
        results["openai"] = await _validate_openai_key(openai_api_key.strip())
        if results["openai"]["valid"]:
            _pending_config["openai_api_key"] = openai_api_key.strip()

    if _wants_json(request):
        return JSONResponse({
            "validation": results,
            "next": "/onboarding/backend",
            "pending_config": {k: "***" for k in _pending_config if "key" in k}
        })

    # Store validation results in session for next step
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
            model="gpt-4o-mini",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}]
        )
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": str(e)[:120]}


# =============================================================================
# Step 3: Backend Selection
# =============================================================================

@router.get("/backend", response_class=HTMLResponse)
async def backend_selection_form(request: Request):
    """Step 3: Backend selection form."""
    global _pending_config

    # Determine available backends based on what was validated
    available_backends = ["mock"]  # Always available

    if _pending_config.get("anthropic_api_key"):
        available_backends.append("anthropic")
    if _pending_config.get("openai_api_key"):
        available_backends.append("openai")

    # Check for local Ollama
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get("http://127.0.0.1:11434/api/tags")
            if resp.status_code == 200:
                available_backends.append("ollama")
    except Exception:
        pass

    if _wants_json(request):
        return JSONResponse({
            "step": 3,
            "available_backends": available_backends,
            "current_selection": _pending_config.get("primary_backend", "mock")
        })

    return templates.TemplateResponse(request, "onboarding/backend.html", {
            "request": request,
            "step": 3,
            "available_backends": available_backends,
            "current_selection": _pending_config.get("primary_backend", "mock")
        })
    )


@router.post("/backend")
async def submit_backend_selection(
    request: Request,
    primary_backend: str = Form(default="mock"),
    fallback_chain: str = Form(default="mock"),
):
    """Save backend selection."""
    global _pending_config

    _pending_config["primary_backend"] = primary_backend
    # Parse fallback chain (comma-separated)
    _pending_config["fallback_chain"] = [
        b.strip() for b in fallback_chain.split(",") if b.strip()
    ] or ["mock"]

    if _wants_json(request):
        return JSONResponse({
            "success": True,
            "primary_backend": primary_backend,
            "fallback_chain": _pending_config["fallback_chain"],
            "next": "/onboarding/network"
        })

    return RedirectResponse(url="/onboarding/network", status_code=303)


# =============================================================================
# Step 4: Network Config
# =============================================================================

@router.get("/network", response_class=HTMLResponse)
async def network_config_form(request: Request):
    """Step 4: Network configuration form."""
    global _pending_config

    # Check IPFS status
    ipfs_status = await _check_ipfs_status()

    if _wants_json(request):
        return JSONResponse({
            "step": 4,
            "ipfs_status": ipfs_status,
            "current_config": {
                "p2p_port": _pending_config.get("p2p_port", 8765),
                "api_port": _pending_config.get("api_port", 8080),
                "bootstrap_nodes": _pending_config.get("bootstrap_nodes", ["wss://bootstrap1.prsm-network.com:8765"])
            }
        })

    return templates.TemplateResponse(request, "onboarding/network.html", {
            "request": request,
            "step": 4,
            "ipfs_status": ipfs_status,
            "p2p_port": _pending_config.get("p2p_port", 8765),
            "api_port": _pending_config.get("api_port", 8080),
            "bootstrap_nodes": ",".join(_pending_config.get("bootstrap_nodes", ["wss://bootstrap1.prsm-network.com:8765"]))
        })
    )


async def _check_ipfs_status() -> Dict[str, Any]:
    """Check IPFS daemon status."""
    # Check if installed
    ipfs_path = shutil.which("ipfs")
    if not ipfs_path:
        return {"status": "not_installed", "message": "IPFS not found on PATH"}

    # Check if running
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post("http://127.0.0.1:5001/api/v0/id")
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "status": "running",
                    "message": f"IPFS node {data.get('ID', 'unknown')[:12]}...",
                    "node_id": data.get("ID")
                }
    except Exception:
        pass

    return {"status": "installed_not_running", "message": "IPFS installed but not running"}


@router.post("/network")
async def submit_network_config(
    request: Request,
    p2p_port: int = Form(default=8765),
    api_port: int = Form(default=8080),
    bootstrap_nodes: str = Form(default="wss://bootstrap1.prsm-network.com:8765"),
    ipfs_auto_start: str = Form(default="true"),
):
    """Save network configuration."""
    global _pending_config

    # Validate ports
    if not (1 <= p2p_port <= 65535):
        raise HTTPException(status_code=400, detail="P2P port must be between 1 and 65535")
    if not (1 <= api_port <= 65535):
        raise HTTPException(status_code=400, detail="API port must be between 1 and 65535")

    _pending_config["p2p_port"] = p2p_port
    _pending_config["api_port"] = api_port
    _pending_config["bootstrap_nodes"] = [
        n.strip() for n in bootstrap_nodes.split(",") if n.strip()
    ]
    _pending_config["ipfs_auto_start"] = ipfs_auto_start.lower() == "true"

    if _wants_json(request):
        return JSONResponse({
            "success": True,
            "config": {
                "p2p_port": p2p_port,
                "api_port": api_port,
                "bootstrap_nodes": _pending_config["bootstrap_nodes"],
                "ipfs_auto_start": _pending_config["ipfs_auto_start"]
            },
            "next": "/onboarding/identity"
        })

    return RedirectResponse(url="/onboarding/identity", status_code=303)


# =============================================================================
# Step 5: Node Identity
# =============================================================================

@router.get("/identity", response_class=HTMLResponse)
async def identity_form(request: Request):
    """Step 5: Node identity configuration."""
    global _pending_config

    # Check for existing identity
    existing_identity = None
    if IDENTITY_OUTPUT_PATH.exists():
        try:
            data = json.loads(IDENTITY_OUTPUT_PATH.read_text())
            existing_identity = {
                "node_id": data.get("node_id", "unknown"),
                "display_name": data.get("display_name", "prsm-node"),
                "created_at": data.get("created_at")
            }
        except Exception:
            pass

    if _wants_json(request):
        return JSONResponse({
            "step": 5,
            "existing_identity": existing_identity,
            "identity_path": str(IDENTITY_OUTPUT_PATH)
        })

    return templates.TemplateResponse(request, "onboarding/identity.html", {
            "request": request,
            "step": 5,
            "existing_identity": existing_identity,
            "identity_path": str(IDENTITY_OUTPUT_PATH)
        })
    )


@router.post("/identity")
async def submit_identity(
    request: Request,
    action: str = Form(default="generate"),
    display_name: str = Form(default="prsm-node"),
    identity_file: Optional[UploadFile] = File(default=None),
):
    """Generate or import node identity."""
    global _pending_config

    if action == "import" and identity_file:
        # Import identity from uploaded file
        try:
            content = await identity_file.read()
            identity_data = json.loads(content)

            # Validate required fields
            required = ["node_id", "public_key", "private_key"]
            if not all(k in identity_data for k in required):
                raise HTTPException(status_code=400, detail="Invalid identity file: missing required fields")

            # Save to identity path
            IDENTITY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            IDENTITY_OUTPUT_PATH.write_text(json.dumps(identity_data, indent=2))

            _pending_config["node_identity_path"] = str(IDENTITY_OUTPUT_PATH)

            if _wants_json(request):
                return JSONResponse({
                    "success": True,
                    "action": "import",
                    "node_id": identity_data["node_id"],
                    "next": "/onboarding/launch"
                })

            return RedirectResponse(url="/onboarding/launch", status_code=303)

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in identity file")

    else:
        # Generate new identity
        try:
            from prsm.node.identity import generate_node_identity, save_node_identity

            identity = generate_node_identity(display_name or "prsm-node")
            save_node_identity(identity, IDENTITY_OUTPUT_PATH)

            _pending_config["node_identity_path"] = str(IDENTITY_OUTPUT_PATH)

            if _wants_json(request):
                return JSONResponse({
                    "success": True,
                    "action": "generate",
                    "node_id": identity.node_id,
                    "public_key_b64": identity.public_key_b64,
                    "next": "/onboarding/launch"
                })

            return RedirectResponse(url="/onboarding/launch", status_code=303)

        except Exception as e:
            logger.error("Failed to generate identity", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to generate identity: {str(e)}")


# =============================================================================
# Step 6: Review & Launch
# =============================================================================

@router.get("/launch", response_class=HTMLResponse)
async def launch_review(request: Request):
    """Step 6: Review configuration and launch."""
    global _pending_config

    # Load template defaults for any missing fields
    final_config = _get_default_config()
    final_config.update(_pending_config)

    if _wants_json(request):
        return JSONResponse({
            "step": 6,
            "config": _mask_sensitive(final_config),
            "config_path": str(CONFIG_OUTPUT_PATH)
        })

    return templates.TemplateResponse(request, "onboarding/launch.html", {
            "request": request,
            "step": 6,
            "config": final_config,
            "config_path": str(CONFIG_OUTPUT_PATH)
        })
    )


@router.post("/launch")
async def execute_launch(request: Request):
    """Write configuration to disk and complete onboarding."""
    global _pending_config

    try:
        # Load template defaults and merge with pending config
        final_config = _get_default_config()
        final_config.update(_pending_config)

        # Ensure output directory exists
        CONFIG_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Write config file
        CONFIG_OUTPUT_PATH.write_text(json.dumps(final_config, indent=2))

        logger.info(
            "Node configuration written",
            config_path=str(CONFIG_OUTPUT_PATH),
            primary_backend=final_config.get("primary_backend")
        )

        # Clear pending config
        _pending_config = {}

        if _wants_json(request):
            return JSONResponse({
                "success": True,
                "config_path": str(CONFIG_OUTPUT_PATH),
                "next_step": "Run: prsm node start"
            })

        return templates.TemplateResponse(request, "onboarding/success.html", {
                "request": request,
                "config_path": str(CONFIG_OUTPUT_PATH),
                "step": 6
            })
        )

    except Exception as e:
        logger.error("Failed to write configuration", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to write configuration: {str(e)}")


def _get_default_config() -> Dict[str, Any]:
    """Load default configuration from template file."""
    if CONFIG_TEMPLATE_PATH.exists():
        try:
            return json.loads(CONFIG_TEMPLATE_PATH.read_text())
        except Exception:
            pass

    # Fallback defaults if template not found
    return {
        "primary_backend": "mock",
        "fallback_chain": ["mock"],
        "anthropic_api_key": None,
        "openai_api_key": None,
        "timeout_seconds": 120,
        "max_retries": 3,
        "p2p_port": 8765,
        "bootstrap_nodes": ["wss://bootstrap1.prsm-network.com:8765"],
        "node_identity_path": "config/node_identity.json",
        "ipfs_auto_start": True,
        "api_port": 8080,
        "log_level": "INFO"
    }


def _mask_sensitive(config: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive values like API keys for display."""
    masked = config.copy()
    for key in ["anthropic_api_key", "openai_api_key"]:
        if masked.get(key):
            val = masked[key]
            if len(val) > 8:
                masked[key] = val[:8] + "***"
    return masked


# =============================================================================
# Utility Endpoints
# =============================================================================

@router.get("/status")
async def get_onboarding_status(request: Request):
    """Get current onboarding status."""
    global _pending_config
    return JSONResponse({
        "pending_config": _mask_sensitive(_pending_config),
        "config_exists": CONFIG_OUTPUT_PATH.exists(),
        "identity_exists": IDENTITY_OUTPUT_PATH.exists()
    })


@router.delete("/reset")
async def reset_onboarding(request: Request):
    """Reset the onboarding session."""
    global _pending_config
    _pending_config = {}
    return JSONResponse({"success": True, "message": "Onboarding session reset"})
