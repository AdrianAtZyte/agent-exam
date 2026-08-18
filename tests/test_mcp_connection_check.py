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

    assert state.mcp_servers == {"files": "connected", "remote": "failed"}
    # The init event still seeds the session id, and the result event is
    # still parsed after it.
    assert state.session_id == "abc"
    assert state.total_cost_usd == 0.1


def test_stream_records_nothing_without_servers():
    state = _drain([{"type": "system", "subtype": "init", "mcp_servers": []}])

    assert state.mcp_servers == {}


def test_check_fails_on_a_server_that_did_not_connect():
    result = connection_check({"files": "connected", "remote": "failed"})

    assert result.status == "FAIL"
    assert "remote (failed)" in result.hint
    assert "files" not in result.hint


def test_check_passes_when_every_server_connected():
    assert connection_check({"files": "connected"}).status == "OK"


def test_check_passes_with_nothing_attached():
    assert connection_check(None).status == "OK"
    assert connection_check({}).status == "OK"
