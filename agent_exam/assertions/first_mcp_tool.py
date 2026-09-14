from __future__ import annotations

from typing import TYPE_CHECKING

from ..mcp import is_mcp_tool, is_mcp_tool_target
from ..schemas import AssertionResult, RunResult
from ..trajectory_walk import iter_tool_calls
from ._shared import McpToolConfig

if TYPE_CHECKING:
    from pathlib import Path


class FirstMcpToolConfig(McpToolConfig):
    """`first_mcp_tool: search` or
    `first_mcp_tool: {server: files, tool: search}`."""


def check(config: FirstMcpToolConfig, result: RunResult, cwd: Path) -> AssertionResult:
    """Pass when the tool is the first MCP tool the agent reached for.

    Native tools are ignored: an agent greps and reads before it decides
    which tool the request calls for, so only MCP calls carry the routing
    decision.
    """
    label = config.label
    details: dict = {"expected": label, "actual": None}
    if missing := config.missing_file(cwd):
        return AssertionResult(
            pass_=False, reason=f"missing {missing}", details=details
        )
    for call in iter_tool_calls(result.trajectory):
        if not is_mcp_tool(call.name):
            continue
        details["actual"] = call.name
        if not is_mcp_tool_target(call.name, config.target):
            return AssertionResult(
                pass_=False,
                reason=f"{call.name} called before {label}",
                details=details,
            )
        if mismatches := config.mismatches(call, cwd):
            return AssertionResult(
                pass_=False,
                reason=f"{label} called first, but " + ", ".join(mismatches),
                details={**details, "mismatches": mismatches},
            )
        return AssertionResult(
            pass_=True, reason=f"{label} called first", details=details
        )
    return AssertionResult(pass_=False, reason=f"{label} never called", details=details)
