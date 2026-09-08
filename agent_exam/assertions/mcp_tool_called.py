from __future__ import annotations

from typing import TYPE_CHECKING

from ..schemas import AssertionResult, RunResult
from ._shared import McpToolConfig

if TYPE_CHECKING:
    from pathlib import Path


class McpToolCalledConfig(McpToolConfig):
    """`mcp_tool_called: search` or
    `mcp_tool_called: {server: files, tool: search, arguments: {...}}`."""


def check(config: McpToolCalledConfig, result: RunResult, cwd: Path) -> AssertionResult:
    """Pass when the tool was called at least once, with arguments matching
    `arguments` when given."""
    label = config.label
    if missing := config.missing_file(cwd):
        return AssertionResult(pass_=False, reason=f"missing {missing}", details={})
    calls = config.calls(result.trajectory)
    mismatches = [config.mismatches(c, cwd) for c in calls]
    matching = sum(1 for m in mismatches if not m)
    details = {"tool": label, "count": len(calls), "matching": matching}
    reason = f"{label} called {len(calls)} time{'s' if len(calls) != 1 else ''}"
    if config.arguments and calls:
        reason += f", {matching} with matching arguments"
        if not matching:
            reason += ": " + "; ".join(
                f"call {i} of {len(calls)}: " + ", ".join(m)
                for i, m in enumerate(mismatches, 1)
            )
        details["mismatches"] = mismatches
    return AssertionResult(pass_=matching > 0, reason=reason, details=details)
