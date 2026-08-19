from __future__ import annotations

import json
import subprocess
from functools import cache
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


# Where a plugin declares its MCP servers: a config file of its own, or an
# `mcpServers` key in its manifest holding either the same mapping or a path
# to a file with one.
_PLUGIN_CONFIGS = (".mcp.json", ".github/mcp.json")
_PLUGIN_MANIFESTS = (
    ".plugin/plugin.json",
    ".github/plugin/plugin.json",
    ".claude-plugin/plugin.json",
)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _server_names(data) -> list[str]:
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    return sorted(servers) if isinstance(servers, dict) else []


def _plugin_mcp_servers(plugin_dir: Path) -> list[str]:
    for rel in _PLUGIN_CONFIGS:
        data = _read_json(plugin_dir / rel)
        if data is not None:
            return _server_names(data)
    for rel in _PLUGIN_MANIFESTS:
        data = _read_json(plugin_dir / rel)
        servers = data.get("mcpServers") if isinstance(data, dict) else None
        if isinstance(servers, str):
            return _server_names(_read_json(plugin_dir / servers))
        if servers is not None:
            return _server_names(data)
    return []


@cache
def _scan_personal_mcp_servers(
    user_config: Path, plugins_root: Path
) -> tuple[str, ...]:
    names = set(_server_names(_read_json(user_config)))
    for plugin_dir in sorted(plugins_root.glob("*/*")):
        names.update(_plugin_mcp_servers(plugin_dir))
    return tuple(sorted(names))


def personal_mcp_servers(
    config_path: Path | None = None, plugins_dir: Path | None = None
) -> list[str]:
    """Names of the MCP servers the developer's own Copilot CLI setup loads,
    from its user config file and from every installed plugin.

    ``--additional-mcp-config`` augments those rather than replacing them, so
    each of these is disabled by name to keep a trial hermetic. Whether a
    plugin is currently enabled is not consulted: disabling a server that
    would not have loaded anyway costs nothing, while missing one that does
    breaks the trial.

    Scanned once per pair of paths, since every attempt of a run asks the
    same question of the same home directory.
    """
    copilot_dir = Path.home() / ".copilot"
    return list(
        _scan_personal_mcp_servers(
            config_path or copilot_dir / "mcp-config.json",
            plugins_dir or copilot_dir / "installed-plugins",
        )
    )


def check_personal_mcp_servers(cfg=None) -> CheckResult:
    """Report the developer's own MCP servers, and any name they share with a
    configured one.

    Copilot CLI merges ``--additional-mcp-config`` last, so a shared name
    resolves to the configured definition, and the developer's server of that
    name cannot be disabled without taking the configured one with it.
    """
    personal = personal_mcp_servers()
    if not personal:
        return CheckResult(
            name="personal mcp servers",
            status="OK",
            hint="none set up",
        )
    shared = sorted(set(personal) & set(cfg.mcp_servers if cfg else ()))
    if shared:
        return CheckResult(
            name="personal mcp servers",
            status="WARN",
            hint=(
                f"{', '.join(shared)} named both under mcp_servers: and in your own "
                "Copilot CLI setup, which stays enabled behind the configured one"
            ),
        )
    return CheckResult(
        name="personal mcp servers",
        status="OK",
        hint=f"{', '.join(personal)} disabled per trial",
    )
