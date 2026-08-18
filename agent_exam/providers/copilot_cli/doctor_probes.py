from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ...schemas import CheckResult


def check_binary() -> CheckResult:
    """Verify the copilot binary is on PATH and returns a version string."""
    try:
        out = subprocess.run(
            ["copilot", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return CheckResult(
            name="copilot binary",
            status="FAIL",
            hint="copilot not on PATH",
        )
    if out.returncode != 0:
        return CheckResult(
            name="copilot binary",
            status="FAIL",
            hint=f"`copilot --version` exited {out.returncode}",
        )
    return CheckResult(
        name="copilot binary",
        status="OK",
        hint=out.stdout.strip(),
    )


def check_probe_model(probe_result) -> CheckResult:
    """Verify the probe attempt recorded a non-empty model name."""
    model = (probe_result.model or "").strip() if probe_result is not None else ""
    if not model:
        return CheckResult(
            name="copilot probe model",
            status="WARN",
            hint=(
                "probe completed but no model name was recorded — "
                "check authentication and copilot version"
            ),
        )
    return CheckResult(
        name="copilot probe model",
        status="OK",
        hint=model,
    )


def personal_mcp_servers(config_path: Path | None = None) -> list[str]:
    """Names of the MCP servers in the developer's own Copilot CLI config.

    ``--additional-mcp-config`` augments that file rather than replacing it,
    so each of these is disabled by name to keep a trial hermetic.
    """
    path = config_path or Path.home() / ".copilot" / "mcp-config.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    return sorted(servers) if isinstance(servers, dict) else []
