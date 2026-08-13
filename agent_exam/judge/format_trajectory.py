from __future__ import annotations

import json

from ..schemas import TextBlock, ThinkingBlock, ToolCallBlock, Turn

DEFAULT_MAX_CHARS = 50_000

# Per-block truncation limits inside the judge-facing trajectory.
#
# Text, tool inputs and tool results all carry judge-critical information:
# text is what the agent shows the user (e.g. a 40-field schema review with
# descriptions + examples), tool inputs are the action being graded (a Bash
# command line, a written file body), and tool results carry command output /
# file contents / API responses. Only thinking is summary enough that a tight
# limit loses no signal.
#
# Tool results are the widest ones: a bounded test crawl or a fetched
# Markdown docs page runs 10-25k chars, and the values a criterion asks
# about (the logged item dicts, a figure quoted from the docs) sit in the
# middle, where head+tail truncation drops them. Whole trajectories run
# well under DEFAULT_MAX_CHARS, so there is room to keep more.
TEXT_BLOCK_MAX = 5_000
THINKING_BLOCK_MAX = 200
TOOL_INPUT_MAX = 4_000
TOOL_RESULT_MAX = 4_000


def format_trajectory(
    trajectory: list[Turn], max_chars: int = DEFAULT_MAX_CHARS
) -> str:
    """Render a trajectory as compact text for the judge prompt.

    Layout per turn:

        [turn 3, assistant]
          thinking: The user wants ...
          text: I'll read the spec file first.
          tool_use Read({"file_path": "/path/spec.json"}) → ok: <result...>

    Subagents render indented under their parent ToolCallBlock. Over
    *max_chars*, the subagent bodies go first, since the parent turns are the
    trajectory being graded; if the parent turns alone still do not fit, their
    middle is truncated with an explicit marker.
    """
    joined = "\n".join(_render(trajectory, depth=0))
    if len(joined) > max_chars:
        joined = "\n".join(_render(trajectory, depth=0, subagents=False))
    if len(joined) <= max_chars:
        return joined
    keep = max_chars // 2
    return f"{joined[:keep]}\n...[trajectory truncated]...\n{joined[-keep:]}"


def _render(trajectory: list[Turn], depth: int, subagents: bool = True) -> list[str]:
    indent = "  " * depth
    out: list[str] = []
    for i, turn in enumerate(trajectory):
        out.append(f"{indent}[turn {i}, {turn.role}]")
        for block in turn.content:
            if isinstance(block, TextBlock):
                if block.text.strip():
                    out.append(f"{indent}  text: {_trunc(block.text, TEXT_BLOCK_MAX)}")
            elif isinstance(block, ThinkingBlock):
                if block.text.strip():
                    out.append(
                        f"{indent}  thinking: {_trunc(block.text, THINKING_BLOCK_MAX)}"
                    )
            elif isinstance(block, ToolCallBlock):
                input_summary = _trunc_mid(
                    json.dumps(block.input, ensure_ascii=False, default=str),
                    TOOL_INPUT_MAX,
                )
                result_summary = _trunc_mid(block.result, TOOL_RESULT_MAX)
                out.append(
                    f"{indent}  tool_use {block.name}({input_summary})"
                    f" → {block.status}: {result_summary}"
                )
                if block.subagent:
                    if subagents:
                        out.extend(_render(block.subagent, depth + 1))
                    else:
                        out.append(
                            f"{indent}  ...[{len(block.subagent)}"
                            " subagent turns omitted]..."
                        )
    return out


def _trunc(s: str, n: int) -> str:
    if s is None:
        return ""
    s = s.replace("\n", " ")
    if len(s) <= n:
        return s
    return s[:n] + "..."


def _trunc_mid(s: str, n: int) -> str:
    """Truncate by keeping the head and tail, showing how many chars were removed."""
    if s is None:
        return ""
    s = s.replace("\n", " ")
    if len(s) <= n:
        return s
    keep = n // 2
    removed = len(s) - 2 * keep
    return s[:keep] + f"...[{removed} chars removed]..." + s[-keep:]


def final_output_text(trajectory: list[Turn]) -> str:
    """Return the text of the last assistant turn (concatenated TextBlocks)."""
    for turn in reversed(trajectory):
        if turn.role != "assistant":
            continue
        texts = [b.text for b in turn.content if isinstance(b, TextBlock)]
        return "\n".join(t for t in texts if t is not None)
    return ""
