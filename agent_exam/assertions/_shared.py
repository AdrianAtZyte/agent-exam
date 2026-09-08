"""Config models shared across multiple assertions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field, StrictBool, field_validator, model_validator

from .._models import _ScalarShorthandModel, _StrictModel
from ..mcp import is_mcp_tool_target, mcp_tool_target
from ..trajectory_walk import iter_tool_calls

if TYPE_CHECKING:
    from pathlib import Path

    from ..schemas import ToolCallBlock, Turn


class JudgeConfigBase(_ScalarShorthandModel):
    """Shared config for `judge` and `judge_agent`: a criterion plus
    optional `include_trajectory` / `pass_on`. Accepts the scalar
    shorthand `judge: <criterion>` and the full mapping form."""

    _shorthand_key: ClassVar[str] = "criterion"
    criterion: str
    include_trajectory: StrictBool = True
    pass_on: list[str] | None = Field(default=None, min_length=1)

    @field_validator("criterion")
    @classmethod
    def _non_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty or whitespace")
        return v

    @field_validator("pass_on")
    @classmethod
    def _no_empty_entries(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and not all(s for s in v):
            raise ValueError("entries must be non-empty strings")
        return v


class ArgumentMatcher(_StrictModel):
    """The mapping form of one entry under `arguments:`, for an expected
    value that is not a literal: `{equals_file: <path>}` or
    `{matches: <regex>}`."""

    equals_file: str | None = None
    matches: str | None = None

    @model_validator(mode="after")
    def _validate_expected(self) -> ArgumentMatcher:
        if (self.equals_file is None) == (self.matches is None):
            raise ValueError("needs exactly one of 'equals_file' or 'matches'")
        if self.matches is not None:
            try:
                re.compile(self.matches)
            except re.error as exc:
                raise ValueError(
                    f"invalid regex pattern {self.matches!r}: {exc}"
                ) from exc
        return self


class McpToolConfig(_ScalarShorthandModel):
    """Shared config for `first_mcp_tool`, `mcp_tool_called` and
    `mcp_tool_not_called`: `<assertion>: <tool>` or
    `<assertion>: {server: <server>, tool: <tool>, arguments: {...}}`.

    Without `server`, the tool counts on whichever server serves one.
    Each `arguments` entry is an expected literal, or an
    `ArgumentMatcher` mapping.
    """

    _shorthand_key: ClassVar[str] = "tool"
    server: str | None = Field(default=None, min_length=1)
    tool: str = Field(min_length=1)
    arguments: dict[str, str | int | float | bool | ArgumentMatcher] | None = None

    @property
    def target(self) -> str:
        return mcp_tool_target(self.tool, self.server)

    @property
    def label(self) -> str:
        return f"{self.server}/{self.tool}" if self.server else self.tool

    def calls(self, trajectory: list[Turn]) -> list[ToolCallBlock]:
        """Every call of the tool, subagents included."""
        target = self.target
        return [
            c for c in iter_tool_calls(trajectory) if is_mcp_tool_target(c.name, target)
        ]

    def missing_file(self, cwd: Path) -> str | None:
        """The first `equals_file` under `arguments` that *cwd* lacks."""
        for expected in (self.arguments or {}).values():
            if (
                isinstance(expected, ArgumentMatcher)
                and expected.equals_file
                and not (cwd / expected.equals_file).is_file()
            ):
                return expected.equals_file
        return None

    def mismatches(self, call: ToolCallBlock, cwd: Path) -> list[str]:
        """How each of *call*'s arguments departs from `arguments`, one entry
        per argument that does; empty when the call matches."""
        out = []
        for key, expected in (self.arguments or {}).items():
            if key not in call.input:
                out.append(f"{key} missing")
                continue
            actual = call.input[key]
            if isinstance(expected, ArgumentMatcher):
                if not isinstance(actual, str):
                    out.append(f"{key} is {type(actual).__name__}, not a string")
                elif expected.matches is not None:
                    if re.search(expected.matches, actual.strip()) is None:
                        out.append(
                            f"{key} is {_shown(actual)!r}, "
                            f"expected match for {expected.matches}"
                        )
                else:
                    wanted = (cwd / expected.equals_file).read_text(errors="replace")
                    if actual.strip() != wanted.strip():
                        out.append(
                            f"{key} is {len(actual)} chars, expected "
                            f"{len(wanted.strip())} ({expected.equals_file})"
                        )
            elif isinstance(expected, str):
                if not isinstance(actual, str) or actual.strip() != expected.strip():
                    out.append(f"{key} is {_shown(actual)!r}, expected {expected!r}")
            elif actual != expected:
                out.append(f"{key} is {_shown(actual)!r}, expected {expected!r}")
        return out


def _shown(value: object) -> object:
    if isinstance(value, str) and len(value) > 200:
        return value[:200] + "…"
    return value
