"""Connection status of the attached MCP servers, read off the harness's
session-start event. A server that dies on startup leaves the agent without
its tools and says nothing else about it.
"""

from __future__ import annotations

import io
import json

from agent_exam.mcp import connection_check
from agent_exam.providers.claude_code.stream_parser import StreamState, drain_stream

_INIT = {
    "type": "system",
    "subtype": "init",
    "session_id": "abc",
    "mcp_servers": [
        {"name": "files", "status": "connected"},
        {"name": "remote", "status": "failed"},
    ],
}


def _drain(events: list[dict]) -> StreamState:
    state = StreamState()
    payload = "".join(json.dumps(e) + "\n" for e in events).encode()
    drain_stream(io.BytesIO(payload), state)
    return state


def test_stream_records_server_statuses():
    state = _drain([_INIT, {"type": "result", "total_cost_usd": 0.1}])

    assert state.mcp_server_status == {"files": "connected", "remote": "failed"}
    # The init event still seeds the session id, and the result event is
    # still parsed after it.
    assert state.session_id == "abc"
    assert state.total_cost_usd == 0.1


def test_stream_records_nothing_without_servers():
    state = _drain([{"type": "system", "subtype": "init", "mcp_servers": []}])

    assert state.mcp_server_status == {}


def test_check_fails_on_a_server_that_did_not_connect():
    result = connection_check({"files": "connected", "remote": "failed"})

    assert result.status == "FAIL"
    assert "remote (failed)" in result.hint
    assert "files" not in result.hint


def test_check_passes_when_every_server_connected():
    assert connection_check({"files": "connected"}).status == "OK"


def test_check_fails_when_an_expected_server_is_missing():
    """The config never reached the CLI, so the agent has no tools at all."""
    result = connection_check({}, ["files"])

    assert result.status == "FAIL"
    assert "files (not attached)" in result.hint


def test_check_passes_with_nothing_attached():
    assert connection_check(None, ["files"]).status == "OK"
    assert connection_check({}).status == "OK"


_CONFIG = """\
default_harness: dummy
mcp_servers:
  files:
    command: sh
providers:
  dummy:
    judge_model: haiku
"""


def _run_with_statuses(root, monkeypatch, statuses):
    """Run one dummy attempt whose harness announced *statuses*."""
    from agent_exam.config import load_config
    from agent_exam.providers.dummy import DummyProvider
    from agent_exam.runner import RunRequest, run

    (root / "evals" / "suites" / "s" / "tasks").mkdir(parents=True)
    (root / "skills" / "skill-a").mkdir(parents=True)
    (root / "skills" / "skill-a" / "SKILL.md").write_text("# skill-a")
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(_CONFIG)
    (root / "evals" / "suites" / "s" / "tasks" / "t.yaml").write_text(
        "kind: execute\nprompt: x\nassertions: []\n"
    )

    invoke = DummyProvider.invoke

    def with_statuses(self, *args, **kwargs):
        result = invoke(self, *args, **kwargs)
        result.mcp_server_status = statuses
        return result

    monkeypatch.setattr(DummyProvider, "invoke", with_statuses)

    exit_code = run(
        load_config(root),
        RunRequest(
            specs=[("s", None)],
            provider="dummy",
            model="",
            k=1,
            n_parallel=1,
            without_skill=False,
        ),
    )
    run_dir = next(iter((root / "evals" / "runs").iterdir()))
    report = json.loads(next(iter((run_dir / "reports").iterdir())).read_text())
    attempt = json.loads(
        (run_dir / "artifacts" / "s" / "t" / "attempt-1" / "attempt.json").read_text()
    )
    return exit_code, report, attempt


def test_attempt_errors_when_a_server_did_not_connect(tmp_path, monkeypatch):
    """Without this the agent just lacks its tools, and the task fails as if
    the skill had routed wrong."""
    exit_code, report, attempt = _run_with_statuses(
        tmp_path / "proj", monkeypatch, {"files": "failed"}
    )

    assert exit_code != 0
    assert [a["verdict"] for a in report["attempts"]] == ["error"]
    assert attempt["mcp_server_status"] == {"files": "failed"}


def test_attempt_is_graded_when_every_server_connected(tmp_path, monkeypatch):
    exit_code, report, attempt = _run_with_statuses(
        tmp_path / "proj", monkeypatch, {"files": "connected"}
    )

    assert exit_code == 0
    assert [a["verdict"] for a in report["attempts"]] == ["pass"]
    assert attempt["mcp_server_status"] == {"files": "connected"}
