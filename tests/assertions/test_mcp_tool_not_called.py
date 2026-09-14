from __future__ import annotations

from fixtures.canned_run_result import assistant_turn, run_result, tool_call

from agent_exam.assertions import mcp_tool_not_called
from agent_exam.assertions.mcp_tool_not_called import McpToolNotCalledConfig


def _run(*inputs):
    calls = [
        tool_call("mcp__web__fetch", tool_use_id=f"tu_{i}", input_=inp)
        for i, inp in enumerate(inputs)
    ]
    return run_result([assistant_turn(*calls)])


def test_not_called_passes(cwd):
    r = mcp_tool_not_called.check(
        McpToolNotCalledConfig.model_validate("fetch"), run_result([]), cwd
    )
    assert r.pass_
    assert r.reason == "fetch not called"


def test_called_fails(cwd):
    r = mcp_tool_not_called.check(
        McpToolNotCalledConfig.model_validate("fetch"), _run({}), cwd
    )
    assert not r.pass_
    assert r.reason == "fetch called 1 time"


def test_arguments_narrow_what_counts(cwd):
    cfg = McpToolNotCalledConfig.model_validate(
        {
            "server": "web",
            "tool": "fetch",
            "arguments": {"url": {"matches": "^http://"}},
        }
    )
    assert mcp_tool_not_called.check(cfg, _run({"url": "https://a"}), cwd).pass_
    r = mcp_tool_not_called.check(
        cfg, _run({"url": "https://a"}, {"url": "http://b"}), cwd
    )
    assert not r.pass_
    assert r.reason == "web/fetch called 2 times, 1 with matching arguments"
