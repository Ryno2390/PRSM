"""Migration bridge: old NodeConfig (node_config.json) -> new PRSMConfig (config.yaml).

Detects ~/.prsm/node_config.json from the legacy wizard and converts it to
the new unified PRSMConfig format at ~/.prsm/config.yaml.  After a successful
migration the old file is renamed to node_config.json.bak.
"""

import json
from pathlib import Path
from typing import Dict, Any


# Paths
_OLD_CONFIG = Path.home() / ".prsm" / "node_config.json"
_OLD_CONFIG_BAK = Path.home() / ".prsm" / "node_config.json.bak"


# Old NodeRole enum values -> new NodeRole values
_ROLE_MAP: Dict[str, str] = {
    "full": "full",
    "compute": "contributor",
    "storage": "consumer",
}


def _old_config_path() -> Path:
    return _OLD_CONFIG


def can_migrate() -> bool:
    """Check if an old-format node_config.json exists and no new config.yaml yet."""
    # Lazy import to avoid pydantic dependency at module level
    new_config = Path.home() / ".prsm" / "config.yaml"
    return _old_config_path().exists() and not new_config.exists()


def _load_old_config() -> Dict[str, Any]:
    """Load and return the old node_config.json as a dict."""
    return json.loads(_old_config_path().read_text())


def _convert(old: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old NodeConfig dict to PRSMConfig-compatible dict."""
    new: Dict[str, Any] = {}

    # Display name
    if "display_name" in old:
        new["display_name"] = old["display_name"]

    # Role mapping: old format stores roles as a list of strings
    old_roles = old.get("roles", ["full"])
    if isinstance(old_roles, list) and old_roles:
        primary_role = old_roles[0]  # take first role
    elif isinstance(old_roles, str):
        primary_role = old_roles
    else:
        primary_role = "full"
    new["node_role"] = _ROLE_MAP.get(primary_role, "full")

    # Resource allocation
    if "cpu_allocation_pct" in old:
        new["cpu_pct"] = max(10, min(90, old["cpu_allocation_pct"]))
    if "memory_allocation_pct" in old:
        new["memory_pct"] = max(10, min(90, old["memory_allocation_pct"]))
    if "gpu_allocation_pct" in old:
        new["gpu_pct"] = max(0, min(100, old["gpu_allocation_pct"]))
    if "storage_gb" in old:
        new["storage_gb"] = max(1.0, float(old["storage_gb"]))
    if "max_concurrent_jobs" in old:
        new["max_concurrent_jobs"] = max(1, min(20, old["max_concurrent_jobs"]))
    if "upload_mbps_limit" in old:
        new["upload_mbps_limit"] = max(0.0, float(old["upload_mbps_limit"]))

    # Scheduling
    if "active_hours_start" in old:
        new["active_hours_start"] = old["active_hours_start"]
    if "active_hours_end" in old:
        new["active_hours_end"] = old["active_hours_end"]
    if "active_days" in old:
        new["active_days"] = old["active_days"]

    # Network
    if "p2p_port" in old:
        new["p2p_port"] = old["p2p_port"]
    if "api_port" in old:
        new["api_port"] = old["api_port"]
    if "bootstrap_nodes" in old:
        new["bootstrap_nodes"] = old["bootstrap_nodes"]

    # Mark as migrated / setup complete
    new["setup_completed"] = True
    new["setup_version"] = "1.0.0"

    return new


def migrate_if_needed() -> bool:
    """Auto-migrate old node_config.json to config.yaml if applicable.

    Returns True if a migration was performed, False otherwise.
    """
    if not can_migrate():
        return False

    try:
        old_data = _load_old_config()
        new_data = _convert(old_data)

        # Lazy import to avoid top-level pydantic dependency
        from prsm.cli_modules.config_schema import PRSMConfig

        config = PRSMConfig(**new_data)
        config.save()

        # Rename old file to .bak
        _old_config_path().rename(_OLD_CONFIG_BAK)

        # Inform the user
        import click
        click.echo("  ✅ Migrated old node_config.json → config.yaml")
        click.echo(f"     Old config backed up to {_OLD_CONFIG_BAK.name}")
        click.echo()

        return True
    except Exception as exc:
        # Don't crash the CLI on migration failure — just warn
        import click
        click.echo(f"  ⚠  Config migration failed: {exc}")
        click.echo("     Old config preserved. Run `prsm setup` to reconfigure.")
        click.echo()
        return False
