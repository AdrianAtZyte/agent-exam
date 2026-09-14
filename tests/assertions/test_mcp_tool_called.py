from __future__ import annotations

from fixtures.canned_run_result import assistant_turn, run_result, tool_call

from agent_exam.assertions import mcp_tool_called
from agent_exam.assertions.mcp_tool_called import McpToolCalledConfig

TOOL = "mcp__zyte__extract_from_user_html"


def _run(*inputs):
    calls = [
        tool_call(TOOL, tool_use_id=f"tu_{i}", input_=inp)
        for i, inp in enumerate(inputs)
    ]
    return run_result([assistant_turn(*calls)])


def _cfg(**kw) -> McpToolCalledConfig:
    return McpToolCalledConfig.model_validate({"tool": "extract_from_user_html", **kw})


def test_called_without_arguments(cwd):
    assert mcp_tool_called.check(_cfg(), _run({}), cwd).pass_
    r = mcp_tool_called.check(_cfg(), run_result([]), cwd)
    assert not r.pass_
    assert r.reason == "extract_from_user_html called 0 times"


def test_a_wrong_server_does_not_count(cwd):
    r = mcp_tool_called.check(_cfg(server="other"), _run({}), cwd)
    assert not r.pass_


def test_one_matching_call_is_enough(cwd):
    r = mcp_tool_called.check(
        _cfg(arguments={"type": "product"}),
        _run({"type": "productList"}, {"type": "product "}),
        cwd,
    )
    assert r.pass_
    assert (
        r.reason == "extract_from_user_html called 2 times, 1 with matching arguments"
    )


def test_no_matching_call_lists_every_mismatch(cwd):
    r = mcp_tool_called.check(
        _cfg(arguments={"type": "product", "n": 1}),
        _run({"type": "productList", "n": 1}, {"n": 2}),
        cwd,
    )
    assert not r.pass_
    assert r.reason == (
        "extract_from_user_html called 2 times, 0 with matching arguments: "
        "call 1 of 2: type is 'productList', expected 'product'; "
        "call 2 of 2: type missing, n is 2, expected 1"
    )


def test_non_string_where_a_string_is_expected(cwd):
    r = mcp_tool_called.check(
        _cfg(arguments={"type": "product"}), _run({"type": 3}), cwd
    )
    assert not r.pass_
    assert "type is 3, expected 'product'" in r.reason


def test_equals_file(cwd):
    html = "<html>" + "x" * 100 + "</html>\n"
    (cwd / "book.html").write_text(html)
    cfg = _cfg(arguments={"userHtml": {"equals_file": "book.html"}})
    assert mcp_tool_called.check(cfg, _run({"userHtml": html}), cwd).pass_
    r = mcp_tool_called.check(cfg, _run({"userHtml": "<html/>"}, {"userHtml": 1}), cwd)
    assert not r.pass_
    assert "call 1 of 2: userHtml is 7 chars, expected 113 (book.html)" in r.reason
    assert "call 2 of 2: userHtml is int, not a string" in r.reason


def test_equals_file_missing(cwd):
    cfg = _cfg(arguments={"userHtml": {"equals_file": "book.html"}})
    r = mcp_tool_called.check(cfg, _run({"userHtml": "x"}), cwd)
    assert not r.pass_
    assert r.reason == "missing book.html"


def test_matches(cwd):
    cfg = _cfg(arguments={"url": {"matches": r"^https://example\.com/"}})
    assert mcp_tool_called.check(
        cfg, _run({"url": " https://example.com/a "}), cwd
    ).pass_
    r = mcp_tool_called.check(cfg, _run({"url": "http://example.com/a"}), cwd)
    assert not r.pass_
    assert (
        "url is 'http://example.com/a', expected match for ^https://example\\.com/"
        in r.reason
    )


def test_subagent_calls_count(cwd):
    inner = assistant_turn(tool_call(TOOL, input_={"type": "product"}))
    outer = tool_call("Agent", tool_use_id="tu_agent", subagent=[inner])
    r = mcp_tool_called.check(
        _cfg(arguments={"type": "product"}), run_result([assistant_turn(outer)]), cwd
    )
    assert r.pass_
