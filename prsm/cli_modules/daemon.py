"""PRSM daemon management — start, stop, restart, status, logs, and service installation.

Extracted from cli.py (Phase 3 of the Setup Experience plan). Uses Rich for
display, subprocess.Popen for backgrounding, and psutil for process/resource info.
Supports macOS (launchd) and Linux (systemd) service installation.
"""

import datetime
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import click

from .theme import THEME, ICONS
from .ui import console, resource_bar

# ---- Daemon paths ----------------------------------------------------------

_DAEMON_PID_FILE = Path.home() / ".prsm" / "daemon.pid"
_DAEMON_LOG_FILE = Path.home() / ".prsm" / "logs" / "daemon.log"

# ---- PID helpers -----------------------------------------------------------


def _get_daemon_pid() -> Optional[int]:
    """Read daemon PID from file, or None if not found."""
    if _DAEMON_PID_FILE.exists():
        try:
            return int(_DAEMON_PID_FILE.read_text().strip())
        except (ValueError, OSError):
            return None
    return None


def _is_daemon_running(pid: Optional[int] = None) -> bool:
    """Check if daemon process is running."""
    if pid is None:
        pid = _get_daemon_pid()
    if pid is None:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        try:
            import os
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _get_daemon_uptime(pid: int) -> str:
    """Get daemon uptime as human-readable string."""
    try:
        import psutil
        proc = psutil.Process(pid)
        start_time = datetime.datetime.fromtimestamp(proc.create_time())
        delta = datetime.datetime.now() - start_time
        hours, rem = divmod(int(delta.total_seconds()), 3600)
        mins, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours}h {mins}m {secs}s"
        elif mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"
    except Exception:
        return "unknown"


def _get_system_cpu_usage() -> Optional[int]:
    """Return current CPU usage percentage."""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.1)
    except Exception:
        return None


def _get_system_memory_usage() -> Optional[dict]:
    """Return memory usage info: pct, used_gb, total_gb."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "pct": int(mem.percent),
            "used_gb": round(mem.used / (1024 ** 3), 1),
            "total_gb": round(mem.total / (1024 ** 3), 1),
        }
    except Exception:
        return None


def _get_system_disk_usage() -> Optional[dict]:
    """Return disk usage info: used_gb, total_gb, free_gb."""
    try:
        import psutil
        disk = psutil.disk_usage(str(Path.home()))
        return {
            "used_gb": round(disk.used / (1024 ** 3), 1),
            "total_gb": round(disk.total / (1024 ** 3), 1),
            "free_gb": round(disk.free / (1024 ** 3), 1),
        }
    except Exception:
        return None


def _get_log_size() -> Optional[str]:
    """Return daemon log file size as human-readable string."""
    try:
        if _DAEMON_LOG_FILE.exists():
            b = _DAEMON_LOG_FILE.stat().st_size
            if b > 1024 ** 2:
                return f"{b / (1024 ** 2):.1f} MB"
            elif b > 1024:
                return f"{b / 1024:.0f} KB"
            return f"{b} B"
    except Exception:
        pass
    return None


# ---- Node command builder --------------------------------------------------


def _load_env_file():
    """Load ~/.prsm/.env into os.environ so the daemon sees API keys."""
    env_path = Path.home() / ".prsm" / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val:
                os.environ[key] = val


def _get_daemon_node_cmd(host: str, port: int) -> list:
    """Build command for 'prsm node start --no-dashboard' as a subprocess."""
    # Load .env before the daemon inherits env vars
    _load_env_file()

    cmd = [sys.executable, "-m", "prsm.cli", "node", "start",
           "--no-dashboard", "--api-port", str(port)]
    try:
        from prsm.cli_modules.config_schema import PRSMConfig
        cfg = PRSMConfig.load()
        if cfg.bootstrap_nodes:
            cmd.extend(["--bootstrap", ",".join(cfg.bootstrap_nodes)])
    except Exception:
        pass
    return cmd


# ---- Start -----------------------------------------------------------------


def daemon_start(host: str = "127.0.0.1", port: int = 8000):
    """Start PRSM node in the background with full P2P connectivity."""
    pid = _get_daemon_pid()
    if pid and _is_daemon_running(pid):
        console.print(f"[red]{ICONS['error']} Daemon already running (PID {pid})[/red]")
        console.print("Run 'prsm daemon stop' first, or 'prsm daemon status' for details.")
        raise SystemExit(1)

    if _DAEMON_PID_FILE.exists():
        _DAEMON_PID_FILE.unlink()

    # Load .env so the daemon inherits API keys
    _load_env_file()

    _DAEMON_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    cmd = _get_daemon_node_cmd(host, port)
    log_fh = open(_DAEMON_LOG_FILE, "a")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=dict(os.environ),  # pass loaded env vars
        )
        _DAEMON_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DAEMON_PID_FILE.write_text(str(proc.pid))

        console.print(f"[green]{ICONS['success']} PRSM daemon started (full node)[/green]")
        console.print(f"  PID:      {proc.pid}")
        console.print(f"  Endpoint: http://{host}:{port}")
        console.print(f"  Log:      {_DAEMON_LOG_FILE}")
        console.print(f"  Mode:     P2P node + API server")
        console.print(f"\n[dim]Run 'prsm daemon logs --follow' to watch logs.[/dim]")
    except Exception as e:
        console.print(f"[red]{ICONS['error']} Error starting daemon: {e}[/red]")
        raise SystemExit(1)
    finally:
        log_fh.close()


# ---- Stop ------------------------------------------------------------------


def daemon_stop(timeout: int = 10):
    """Stop the PRSM daemon. Returns True if stopped successfully."""
    pid = _get_daemon_pid()
    if not pid:
        console.print("[dim]Daemon is not running (no PID file found).[/dim]")
        return False

    if not _is_daemon_running(pid):
        _DAEMON_PID_FILE.unlink()
        console.print("[dim]Daemon was not running (cleaned up stale PID file).[/dim]")
        return False

    console.print(f"Stopping daemon (PID {pid})...", style=THEME.warning)

    try:
        import os
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _DAEMON_PID_FILE.unlink()
        console.print("[dim]Daemon was not running (cleaned up PID file).[/dim]")
        return False
    except PermissionError:
        console.print(f"[red]{ICONS['error']} Permission denied. Try as the daemon owner.[/red]")
        raise SystemExit(1)

    start = time.time()
    while time.time() - start < timeout:
        if not _is_daemon_running(pid):
            _DAEMON_PID_FILE.unlink()
            console.print(f"  [green]{ICONS['success']} Daemon stopped gracefully.[/green]")
            return True
        time.sleep(0.5)

    console.print("[yellow]Graceful shutdown timed out. Sending SIGKILL...[/yellow]")
    try:
        import os
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
    except ProcessLookupError:
        pass

    if _DAEMON_PID_FILE.exists():
        _DAEMON_PID_FILE.unlink()
    console.print(f"  [green]{ICONS['success']} Daemon killed.[/green]")
    return True


# ---- Restart ---------------------------------------------------------------


def daemon_restart(host: str = "127.0.0.1", port: int = 8000, timeout: int = 10):
    """Stop then start the daemon. If not running, just starts it."""
    pid = _get_daemon_pid()
    was_running = pid is not None and _is_daemon_running(pid)

    if was_running:
        daemon_stop(timeout=timeout)

    daemon_start(host=host, port=port)


# ---- Status ----------------------------------------------------------------


def daemon_status(output_format: str = "text"):
    """Show daemon status with resource bars and config info."""
    pid = _get_daemon_pid()
    running = pid is not None and _is_daemon_running(pid)

    data = {
        "ok": True,
        "running": running,
        "pid": pid if running else None,
        "uptime": _get_daemon_uptime(pid) if running else None,
        "pid_file": str(_DAEMON_PID_FILE),
        "log_file": str(_DAEMON_LOG_FILE),
        "log_size": _get_log_size(),
    }

    if output_format == "json":
        from prsm.cli import _agent_output
        _agent_output(data)
        return

    # Rich output
    console.print(f"\n  {ICONS['prism']} PRSM Daemon Status", style=THEME.heading)
    console.print(f"    {'─' * 22}", style=THEME.rule)

    if running:
        console.print(f"    Status:     {ICONS['success']} Running (PID {pid})")
        console.print(f"    Uptime:     {_get_daemon_uptime(pid)}")
    else:
        if pid and not running:
            console.print(f"    Status:     {ICONS['warning']} Stopped (stale PID file)")
        else:
            console.print(f"    Status:     {ICONS['info']} Stopped")

    # Resource usage bars
    cpu = _get_system_cpu_usage()
    mem = _get_system_memory_usage()
    disk = _get_system_disk_usage()
    log_size = _get_log_size()

    if cpu is not None:
        console.print()
        resource_bar("CPU", cpu, 100, "%")
    if mem:
        resource_bar("Memory", mem["pct"], 100, "%")
    if disk:
        console.print(
            f"      {'Storage':>12}  {disk['used_gb']} GB used / {disk['total_gb']} GB total"
        )
    if log_size:
        console.print(
            f"      {'Log size':>12}  {log_size}", style=THEME.dim
        )

    # Config summary
    try:
        from prsm.cli_modules.config_schema import PRSMConfig
        cfg = PRSMConfig.load()
        if cfg.setup_completed or cfg.node_role:
            console.print()
            console.print(f"    {ICONS['bullet']} Config:", style=THEME.heading)
            role_labels = {"contributor": "Contributor", "consumer": "Consumer", "full": "Full Node"}
            role_label = role_labels.get(cfg.node_role, cfg.node_role)
            console.print(f"      Role:       {role_label}")
            console.print(f"      P2P Port:   {cfg.p2p_port}")
            console.print(f"      API Port:   {cfg.api_port}")
            mcp_status = f"enabled (:{cfg.mcp_server_port})" if cfg.mcp_server_enabled else "disabled"
            console.print(f"      MCP Server: {mcp_status}")
            if cfg.bootstrap_nodes:
                console.print(f"      Bootstrap:  {len(cfg.bootstrap_nodes)} node(s)")
            if cfg.wallet_address:
                console.print(f"      Wallet:     {cfg.wallet_address[:12]}...")
    except Exception:
        pass

    console.print()
    console.print(f"      PID file: {_DAEMON_PID_FILE}", style=THEME.dim)
    console.print(f"      Log file: {_DAEMON_LOG_FILE}", style=THEME.dim)


# ---- Logs ------------------------------------------------------------------


def daemon_logs(lines: int = 50, follow: bool = False):
    """Show daemon logs. --follow for tail -f style."""
    if not _DAEMON_LOG_FILE.exists():
        console.print(f"[dim]No log file found at {_DAEMON_LOG_FILE}[/dim]")
        console.print("Start the daemon first with 'prsm daemon start'.")
        return

    if follow:
        console.print(f"Following {_DAEMON_LOG_FILE} (Ctrl+C to stop)...\n")
        with open(_DAEMON_LOG_FILE, "r") as f:
            f.seek(0, 2)
            try:
                while True:
                    line = f.readline()
                    if line:
                        print(line, end="")
                    else:
                        time.sleep(0.1)
            except KeyboardInterrupt:
                console.print("\n[dim]Stopped following logs.[/dim]")
    else:
        result = subprocess.run(
            ["tail", "-n", str(lines), str(_DAEMON_LOG_FILE)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            console.print(f"[red]{ICONS['error']} Error reading log file: {result.stderr}[/red]")


# ---- Service install / uninstall -------------------------------------------


def daemon_service_install(dry_run: bool = False, host: str = "127.0.0.1", port: int = 8000):
    """Install PRSM daemon as a system service (launchd or systemd)."""
    system = sys.platform
    if system == "darwin":
        _install_launchd_service(dry_run, host, port)
    elif system == "linux":
        _install_systemd_service(dry_run, host, port)
    else:
        console.print(f"[red]{ICONS['error']} Platform '{system}' not supported[/red]")
        console.print("Service installation only on macOS and Linux.")
        raise SystemExit(1)


def daemon_service_uninstall(yes: bool = False):
    """Remove the PRSM daemon system service."""
    system = sys.platform
    if system == "darwin":
        _uninstall_launchd_service(yes)
    elif system == "linux":
        _uninstall_systemd_service(yes)
    else:
        console.print(f"[red]{ICONS['error']} Platform '{system}' not supported[/red]")
        raise SystemExit(1)


def _install_launchd_service(dry_run: bool, host: str, port: int):
    """Install launchd agent on macOS."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "ai.prsm.node.plist"
    cmd_args = _get_daemon_node_cmd(host, port)
    program_args = "\n".join(f"        <string>{arg}</string>" for arg in cmd_args)

    plist = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.prsm.node</string>
    <key>ProgramArguments</key>
    <array>
{program_args}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>30</integer>
    <key>StandardOutPath</key>
    <string>{_DAEMON_LOG_FILE}</string>
    <key>StandardErrorPath</key>
    <string>{_DAEMON_LOG_FILE}</string>
    <key>WorkingDirectory</key>
    <string>{Path.home()}</string>
</dict>
</plist>
'''
    if dry_run:
        console.print(f"\n[bold]Generated launchd plist:[/bold]\n")
        console.print(plist)
        console.print(f"\n[dim]Would write to: {plist_path}[/dim]")
        console.print(f"[dim]Would run: launchctl load {plist_path}[/dim]")
        return

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist)
    console.print(f"[green]{ICONS['success']} Created launchd plist[/green]")
    console.print(f"  Path: {plist_path}")

    result = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, text=True)
    if result.returncode == 0:
        console.print("  Loaded: launchctl load")
        console.print("\n[dim]The daemon will start automatically and restart on login.[/dim]")
    else:
        console.print(f"[yellow]{ICONS['warning']} launchctl load failed: {result.stderr}[/yellow]")
        console.print(f"Try: launchctl load {plist_path}")


def _install_systemd_service(dry_run: bool, host: str, port: int):
    """Install systemd user unit on Linux."""
    service_path = Path.home() / ".config" / "systemd" / "user" / "prsm-node.service"
    cmd_args = _get_daemon_node_cmd(host, port)
    exec_start = " ".join(cmd_args)

    unit = f'''[Unit]
Description=PRSM Node Daemon
After=network.target

[Service]
Type=simple
ExecStart={exec_start}
Restart=always
RestartSec=10
StandardOutput=append:{_DAEMON_LOG_FILE}
StandardError=append:{_DAEMON_LOG_FILE}
WorkingDirectory={Path.home()}

[Install]
WantedBy=default.target
'''
    if dry_run:
        console.print(f"\n[bold]Generated systemd unit:[/bold]\n")
        console.print(unit)
        console.print(f"\n[dim]Would write to: {service_path}[/dim]")
        console.print("[dim]Would run: systemctl --user daemon-reload && systemctl --user enable --now prsm-node[/dim]")
        return

    service_path.parent.mkdir(parents=True, exist_ok=True)
    service_path.write_text(unit)
    console.print(f"[green]{ICONS['success']} Created systemd unit[/green]")
    console.print(f"  Path: {service_path}")

    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    result = subprocess.run(
        ["systemctl", "--user", "enable", "--now", "prsm-node.service"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        console.print("  Enabled: systemctl --user enable --now")
        console.print("\n[dim]The daemon will start automatically and restart on boot.[/dim]")
    else:
        console.print(f"[yellow]{ICONS['warning']} systemctl enable failed: {result.stderr}[/yellow]")


def _uninstall_launchd_service(yes: bool):
    """Uninstall launchd agent on macOS."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "ai.prsm.node.plist"
    if not plist_path.exists():
        console.print("[dim]No launchd service found.[/dim]")
        return
    if not yes and not click.confirm(f"Unload and remove {plist_path}?"):
        console.print("[dim]Cancelled.[/dim]")
        return
    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True, text=True)
    console.print("[dim]Unloaded launchd service.[/dim]")
    plist_path.unlink()
    console.print(f"[green]{ICONS['success']} Removed {plist_path}[/green]")


def _uninstall_systemd_service(yes: bool):
    """Uninstall systemd user unit on Linux."""
    service_path = Path.home() / ".config" / "systemd" / "user" / "prsm-node.service"
    if not service_path.exists():
        console.print("[dim]No systemd service found.[/dim]")
        return
    if not yes and not click.confirm(f"Disable and remove {service_path}?"):
        console.print("[dim]Cancelled.[/dim]")
        return
    subprocess.run(["systemctl", "--user", "stop", "prsm-node.service"], capture_output=True)
    subprocess.run(["systemctl", "--user", "disable", "prsm-node.service"], capture_output=True)
    service_path.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    console.print(f"[green]{ICONS['success']} Removed {service_path}[/green]")
