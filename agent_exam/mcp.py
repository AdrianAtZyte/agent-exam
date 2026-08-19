from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from typing import TYPE_CHECKING

from .config import McpStdioServer
from .errors import UsageError
from .schemas import CheckResult
from .trajectory_walk import iter_tool_calls

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from .config import Config
    from .providers.base import Provider
    from .schemas import Turn

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_refs(value: str, where: str) -> str:
    """Substitute ``${VAR}`` references from the parent environment."""

    def replace(match: re.Match) -> str:
        name = match.group(1)
        try:
            return os.environ[name]
        except KeyError:
            raise UsageError(
                f"{where}: ${{{name}}} is not set in the environment"
            ) from None

    return _ENV_REF.sub(replace, value)


def _env_refs(value: str) -> list[str]:
    return _ENV_REF.findall(value)


def _selected(cfg: Config, names: list[str] | None) -> dict:
    if names is None:
        return dict(cfg.mcp_servers)
    return {name: cfg.mcp_servers[name] for name in names if name in cfg.mcp_servers}


def resolve_servers(cfg: Config, names: list[str] | None = None) -> dict[str, dict]:
    """Return the selected servers as MCP JSON, with ``${VAR}`` expanded.

    *names* selects a subset of ``cfg.mcp_servers``; ``None`` selects all of
    them. Raises :py:class:`UsageError` when a referenced environment
    variable is unset, so a missing credential surfaces before the agent
    runs rather than as a tool failure mid-task.
    """
    out: dict[str, dict] = {}
    for name, server in _selected(cfg, names).items():
        data = server.model_dump()
        if isinstance(server, McpStdioServer):
            # Optional in the MCP JSON everyone copy-pastes, and rejected
            # by some harnesses' own config schemas.
            data.pop("type")
        for key in ("env", "headers"):
            if key in data:
                data[key] = {
                    k: _expand_env_refs(v, f"mcp_servers.{name}.{key}.{k}")
                    for k, v in data[key].items()
                }
        out[name] = data
    return out


def render_mcp_json(run_tmp_root: Path, servers: dict[str, dict]) -> Path:
    """Write *servers* as an MCP config file and return its path.

    The file lands directly under *run_tmp_root*, next to the attempt cwd
    rather than in it, so a rendered credential is not archived with the
    run's artifacts. The name is random because trigger attempts share a
    cwd and would otherwise overwrite each other's file mid-run.
    """
    path = run_tmp_root / f"{uuid.uuid4().hex[:12]}.mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}))
    return path


_CANONICAL_PREFIX = "mcp__"
_SEPARATORS = ("__", "_", "-")


def canonical_tool_name(name: str, servers: Iterable[str]) -> str:
    """Rewrite an MCP tool name into Claude Code's ``mcp__<server>__<tool>``.

    For harnesses that report an MCP call as one joined string and nothing
    else, so the only way back to the server is to match the configured
    names against the spellings in use. A name that belongs to no
    configured server is returned unchanged.

    Harnesses that name the server in a field of their own resolve the call
    from that field instead; this guesses, and a native tool whose name
    happens to start with a server name would be guessed wrong.
    """
    if name.startswith(_CANONICAL_PREFIX):
        return name
    bare = name[4:] if name.startswith(("mcp_", "mcp-")) else name
    # Longest first, so a server named `github` doesn't claim a tool of
    # `github-actions`.
    for server in sorted(servers, key=len, reverse=True):
        for separator in _SEPARATORS:
            prefix = f"{server}{separator}"
            if bare.startswith(prefix):
                return f"{_CANONICAL_PREFIX}{server}__{bare[len(prefix) :]}"
    return name


def is_mcp_tool(name: str) -> bool:
    """Whether *name* is an MCP tool in its canonical spelling."""
    return name.startswith(_CANONICAL_PREFIX)


def settles_tool_trigger(name: str, target: str, negative: bool) -> bool:
    """Whether a call to *name* settles a trigger aimed at tool *target*.

    The target itself always does. A positive case is settled by a call to
    any MCP tool: the case grades on the first one, so reaching for another
    server's tool answers it just as decisively. A negative case has to run
    the turn out, since the agent can call one MCP tool and still reach for
    the target afterwards.
    """
    return name == target or (not negative and is_mcp_tool(name))


def canonicalize_tool_names(trajectory: list[Turn], servers: Iterable[str]) -> None:
    """Rename every MCP tool call in *trajectory* to its canonical spelling,
    in place, so one ``tool_called:`` line grades on any harness.
    """
    servers = list(servers)
    for call in iter_tool_calls(trajectory):
        call.name = canonical_tool_name(call.name, servers)


def preflight(cfg: Config, provider: Provider) -> list[CheckResult]:
    """Static checks for the configured MCP servers: stdio commands resolve
    on ``PATH``, referenced environment variables are set, and the selected
    harness can actually attach servers.
    """
    # Imported here because the provider registry imports every provider, and
    # the providers import this module.
    from .providers.base import Provider as _Base

    if not cfg.mcp_servers:
        return []

    results: list[CheckResult] = []

    if type(provider).stage_mcp_config is _Base.stage_mcp_config:
        results.append(
            CheckResult(
                name="mcp servers supported",
                status="WARN",
                hint=(
                    f"{provider.name} attaches no MCP servers, so the "
                    f"{len(cfg.mcp_servers)} configured under mcp_servers: "
                    "do nothing in this run"
                ),
            )
        )

    missing_cmd = sorted(
        name
        for name, server in cfg.mcp_servers.items()
        if isinstance(server, McpStdioServer) and not shutil.which(server.command)
    )
    if missing_cmd:
        results.append(
            CheckResult(
                name="mcp server commands",
                status="FAIL",
                hint=f"not on PATH: {', '.join(missing_cmd)}",
            )
        )

    missing_vars = sorted(
        {
            var
            for server in cfg.mcp_servers.values()
            for value in (
                *getattr(server, "env", {}).values(),
                *getattr(server, "headers", {}).values(),
            )
            for var in _env_refs(value)
            if var not in os.environ
        }
    )
    if missing_vars:
        results.append(
            CheckResult(
                name="mcp server environment",
                status="FAIL",
                hint=f"referenced but unset: {', '.join(missing_vars)}",
            )
        )

    if not missing_cmd and not missing_vars:
        results.append(
            CheckResult(
                name="mcp servers",
                status="OK",
                hint=f"{len(cfg.mcp_servers)} configured",
            )
        )
    return results


def connection_check(statuses: dict[str, str] | None) -> CheckResult:
    """Report the MCP connection statuses a harness announced at session
    start. A server that failed to connect leaves the agent silently
    without its tools, which reads as a skill failure.
    """
    if not statuses:
        return CheckResult(
            name="mcp servers connected",
            status="OK",
            hint="no MCP servers attached",
        )
    failed = sorted(
        f"{name} ({status})"
        for name, status in statuses.items()
        if status != "connected"
    )
    if failed:
        return CheckResult(
            name="mcp servers connected",
            status="FAIL",
            hint=f"did not connect: {', '.join(failed)}",
        )
    return CheckResult(
        name="mcp servers connected",
        status="OK",
        hint=f"{len(statuses)} connected",
    )
