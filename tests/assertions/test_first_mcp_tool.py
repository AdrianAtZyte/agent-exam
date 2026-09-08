from __future__ import annotations

from fixtures.canned_run_result import (
    assistant_turn,
    run_result,
    tool_call,
    user_turn,
)

from agent_exam.assertions import first_mcp_tool
from agent_exam.assertions.first_mcp_tool import FirstMcpToolConfig

_TARGET = "mcp__files__search"


def _trial_with_tools(*names):
    calls = [tool_call(n, tool_use_id=f"tu_{i}") for i, n in enumerate(names)]
    return run_result([user_turn("go"), assistant_turn(*calls)])


def _cfg(**kw) -> FirstMcpToolConfig:
    return FirstMcpToolConfig.model_validate(
        {"server": "files", "tool": "search", **kw}
    )


def test_native_calls_before_the_target_pass(cwd):
    r = first_mcp_tool.check(_cfg(), _trial_with_tools("Grep", "Read", _TARGET), cwd)
    assert r.pass_
    assert r.reason == "files/search called first"


def test_another_mcp_tool_first_fails(cwd):
    r = first_mcp_tool.check(
        _cfg(), _trial_with_tools("Read", "mcp__notes__search", _TARGET), cwd
    )
    assert not r.pass_
    assert "mcp__notes__search" in r.reason


def test_target_never_called_fails(cwd):
    r = first_mcp_tool.check(_cfg(), _trial_with_tools("Read"), cwd)
    assert not r.pass_
    assert "never called" in r.reason


def test_without_a_server_any_servers_tool_counts(cwd):
    cfg = FirstMcpToolConfig.model_validate("search")
    assert first_mcp_tool.check(cfg, _trial_with_tools("mcp__notes__search"), cwd).pass_
    assert not first_mcp_tool.check(
        cfg, _trial_with_tools("mcp__notes__list"), cwd
    ).pass_


def test_arguments_of_the_first_call_are_checked(cwd):
    cfg = _cfg(arguments={"query": "invoice"})
    ok = run_result([assistant_turn(tool_call(_TARGET, input_={"query": "invoice"}))])
    bad = run_result([assistant_turn(tool_call(_TARGET, input_={"query": "receipt"}))])
    assert first_mcp_tool.check(cfg, ok, cwd).pass_
    r = first_mcp_tool.check(cfg, bad, cwd)
    assert not r.pass_
    assert (
        r.reason
        == "files/search called first, but query is 'receipt', expected 'invoice'"
    )
