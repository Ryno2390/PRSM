"""Sprint 566 / F25 — compute_provider.detect_resources subprocess scoping.

Surfaced during sprint-566 droplet redeploy: the Linux operator
daemon crashed at startup with::

    UnboundLocalError: cannot access local variable 'subprocess'
    where it is not associated with a value
    File ".../compute_provider.py", line 151, in detect_resources
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):

Root cause — Python scoping bug. ``import subprocess`` at the top
of the module makes `subprocess` a module-level name. But a SECOND
``import subprocess`` inside the Darwin-only ``elif`` branch makes
Python treat `subprocess` as a function-local variable for the
WHOLE function (any assignment statement, including imports,
forces local scope). On non-Darwin systems the Darwin branch is
skipped, the local `subprocess` is never assigned, and the
subsequent reference at line 151 (`subprocess.TimeoutExpired`)
raises UnboundLocalError.

The bug hid on Mac dev because the Darwin branch always runs
there + the local `import subprocess` always succeeds. Linux
droplet (where the operator daemon actually lives) crashed every
startup since the sprint that introduced the conditional re-import.

Fix: remove the redundant `import subprocess` inside the elif.
The top-level import on line 17 covers all references.
"""
from __future__ import annotations

from unittest.mock import patch


def test_detect_resources_runs_clean_on_linux():
    """The function must execute end-to-end on a Linux-shaped
    platform without raising UnboundLocalError. Pre-fix would
    crash; post-fix must return a Resources object."""
    from prsm.node.compute_provider import detect_resources

    with patch("platform.system", return_value="Linux"):
        # detect_resources must NOT raise UnboundLocalError on
        # any platform — Darwin-only re-import was the bug.
        resources = detect_resources()
    # Returns a Resources-shaped object (truthy attrs probed
    # below — the EXACT shape varies per platform; pin only the
    # invariant fields that compute_provider always populates).
    assert hasattr(resources, "cpu_count")
    assert isinstance(resources.cpu_count, int)
    assert resources.cpu_count >= 1


def test_detect_resources_runs_clean_on_darwin():
    """Darwin path must still work post-fix (Mac dev regression
    guard). The Apple-M-series nominal-clock lookup uses
    subprocess to call sysctl."""
    from prsm.node.compute_provider import detect_resources

    with patch("platform.system", return_value="Darwin"):
        resources = detect_resources()
    assert hasattr(resources, "cpu_count")


def test_subprocess_imported_at_module_level_only():
    """Defensive: there must be exactly one ``import subprocess``
    in compute_provider.py — the top-level one. A second import
    inside a function body would re-trigger the scoping bug.

    This test reads the source as text so a future regressor that
    re-adds the inner import gets caught at CI time, not at
    next Linux daemon start.
    """
    import re
    import inspect
    import prsm.node.compute_provider as mod

    src = inspect.getsource(mod)
    import_lines = re.findall(r"^[ \t]*import\s+subprocess\b", src, re.M)
    assert len(import_lines) == 1, (
        f"Sprint 566 F25 invariant: `import subprocess` must appear "
        f"exactly once (at module top-level). Found {len(import_lines)} "
        f"occurrence(s). A second import inside a function body would "
        f"re-trigger UnboundLocalError on non-Darwin platforms."
    )


def test_detect_resources_handles_no_nvidia_smi():
    """The GPU block (lines 141-152) does a subprocess.run that
    fails on hosts without nvidia-smi. The except clause must
    catch FileNotFoundError without raising UnboundLocalError.
    Pre-fix: this except would re-raise UnboundLocalError on
    Linux + no-Darwin path. Post-fix: clean fall-through."""
    from prsm.node.compute_provider import detect_resources

    # Linux + no nvidia-smi is the normal droplet posture.
    with patch("platform.system", return_value="Linux"):
        resources = detect_resources()
    # GPU block should NOT have raised; either gpu_available is
    # False OR (if nvidia-smi exists, which it doesn't on droplet)
    # populated normally.
    assert hasattr(resources, "gpu_available")
