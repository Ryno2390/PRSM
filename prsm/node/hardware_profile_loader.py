"""Sprint 681 — load the local hardware_profile for DISCOVERY_ANNOUNCE.

Resolution order:
  1. ``PRSM_HARDWARE_PROFILE_FILE`` env → operator-pinned JSON path.
     If set but file missing/invalid → return None (operator was
     being explicit; don't silently fall through).
  2. ``<cache_dir>/hardware_profile.json`` — cache from prior runs.
  3. Fresh compute via the supplied (or default) profiler factory.
     Result is cached to <cache_dir>/hardware_profile.json for
     next start (write failure is non-fatal).

Any unexpected error returns None. Pre-680 wire format is preserved
when None is propagated through sprint 680's announce_self() path.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".prsm"
_CACHE_FILENAME = "hardware_profile.json"


def _default_profiler_factory():
    from prsm.compute.wasm.profiler import HardwareProfiler
    return HardwareProfiler()


def load_local_hardware_profile(
    cache_dir: Optional[Path] = None,
    profiler_factory: Callable[[], Any] = _default_profiler_factory,
) -> Optional[Dict[str, Any]]:
    """Load or compute the local hardware profile as a dict."""
    cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR

    # 1) explicit env override
    env_path = os.environ.get("PRSM_HARDWARE_PROFILE_FILE", "").strip()
    if env_path:
        env_file = Path(env_path)
        if not env_file.exists():
            logger.warning(
                "PRSM_HARDWARE_PROFILE_FILE=%s does not exist; peer "
                "will not advertise hardware_profile.", env_path,
            )
            return None
        try:
            data = json.loads(env_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "PRSM_HARDWARE_PROFILE_FILE=%s failed to parse: %s; "
                "peer will not advertise hardware_profile.",
                env_path, exc,
            )
            return None
        if not isinstance(data, dict):
            logger.warning(
                "PRSM_HARDWARE_PROFILE_FILE=%s did not contain a JSON "
                "object; peer will not advertise hardware_profile.",
                env_path,
            )
            return None
        return data

    # 2) on-disk cache
    cache_file = cache_dir / _CACHE_FILENAME
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if isinstance(data, dict):
                return data
            logger.warning(
                "hardware_profile cache %s top-level is not a dict; "
                "recomputing.", cache_file,
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "hardware_profile cache %s unreadable (%s); recomputing.",
                cache_file, exc,
            )

    # 3) fresh compute
    try:
        profiler = profiler_factory()
        profile = profiler.detect()
        data = profile.to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Hardware profiler raised: %s; peer will not advertise "
            "hardware_profile.", exc,
        )
        return None

    # Cache for next start (non-fatal on failure)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))
    except (OSError, PermissionError) as exc:
        logger.debug(
            "Could not write hardware_profile cache to %s: %s "
            "(non-fatal).", cache_file, exc,
        )

    return data
