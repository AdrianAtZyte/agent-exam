from __future__ import annotations

from typing import TYPE_CHECKING

from ..schemas import AssertionResult, RunResult
from ._shared import McpToolConfig

if TYPE_CHECKING:
    from pathlib import Path


class McpToolNotCalledConfig(McpToolConfig):
    """`mcp_tool_not_called: search` or
    `mcp_tool_not_called: {server: files, tool: search, arguments: {...}}`."""


def check(
    config: McpToolNotCalledConfig, result: RunResult, cwd: Path
) -> AssertionResult:
    """Pass when the tool was never called, or never with arguments matching
    `arguments` when given."""
    label = config.label
    if missing := config.missing_file(cwd):
        return AssertionResult(pass_=False, reason=f"missing {missing}", details={})
    calls = config.calls(result.trajectory)
    matching = sum(1 for c in calls if not config.mismatches(c, cwd))
    details = {"tool": label, "count": len(calls), "matching": matching}
    if not calls:
        return AssertionResult(
            pass_=True, reason=f"{label} not called", details=details
        )
    reason = f"{label} called {len(calls)} time{'s' if len(calls) != 1 else ''}"
    if config.arguments:
        reason += f", {matching} with matching arguments"
    return AssertionResult(pass_=matching == 0, reason=reason, details=details)
