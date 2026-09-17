"""`mcp_servers:` in config.yaml and in a task file: parsing, `${VAR}`
expansion, and the checks that refuse a run the servers cannot serve.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.parse
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from agent_exam.config import McpHttpServer, McpStdioServer, load_config
from agent_exam.errors import UsageError
from agent_exam.mcp import preflight, resolve_servers, token_lifetimes
from agent_exam.oauth import login
from agent_exam.providers import get_provider
from agent_exam.tasks import load_task
from agent_exam.validation import validate_suite

if TYPE_CHECKING:
    from pathlib import Path


def _project(tmp_path: Path, config: str) -> Path:
    root = tmp_path / "proj"
    (root / "evals" / "suites" / "s" / "tasks").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(dedent(config))
    return root


_CONFIG = """\
default_harness: dummy
mcp_servers:
  files:
    command: mcp-files
    args: ["--root", "."]
    env:
      TOKEN: "${MCP_TOKEN}"
  remote:
    type: http
    url: https://example.test/mcp
    headers:
      Authorization: "Bearer ${MCP_TOKEN}"
"""


def test_config_parses_both_server_shapes(tmp_path):
    cfg = load_config(_project(tmp_path, _CONFIG))

    assert isinstance(cfg.mcp_servers["files"], McpStdioServer)
    assert cfg.mcp_servers["files"].args == ["--root", "."]
    assert isinstance(cfg.mcp_servers["remote"], McpHttpServer)
    assert cfg.mcp_servers["remote"].url == "https://example.test/mcp"


def test_config_defaults_a_url_only_server_to_http(tmp_path):
    root = _project(
        tmp_path,
        """\
        mcp_servers:
          remote:
            url: https://example.test/mcp
        """,
    )
    cfg = load_config(root)

    assert cfg.mcp_servers["remote"] == McpHttpServer(url="https://example.test/mcp")
    assert resolve_servers(cfg)["remote"]["type"] == "http"


def test_config_rejects_unknown_server_key(tmp_path):
    root = _project(
        tmp_path,
        """\
        mcp_servers:
          files:
            command: mcp-files
            cwd: /tmp
        """,
    )
    with pytest.raises(UsageError):
        load_config(root)


def test_config_rejects_a_server_name_with_a_dot(tmp_path):
    """The name is half of an `mcp__<server>__<tool>` tool name and a TOML key
    path in the config codex_cli renders, where a dot would nest instead."""
    root = _project(
        tmp_path,
        """\
        mcp_servers:
          foo.bar:
            command: mcp-files
        """,
    )
    with pytest.raises(UsageError):
        load_config(root)


def test_config_rejects_a_server_name_with_a_doubled_underscore(tmp_path):
    """`mcp__a__b__search` would read as a call to `b` on server `a`."""
    root = _project(
        tmp_path,
        """\
        mcp_servers:
          a__b:
            command: mcp-files
        """,
    )
    with pytest.raises(UsageError):
        load_config(root)


def test_resolve_expands_env_refs(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    cfg = load_config(_project(tmp_path, _CONFIG))

    resolved = resolve_servers(cfg)

    assert resolved["files"]["env"] == {"TOKEN": "s3cret"}
    assert resolved["remote"]["headers"] == {"Authorization": "Bearer s3cret"}
    # `type` is dropped for stdio — harnesses that take the MCP JSON verbatim
    # accept it, but some of their own config schemas reject it.
    assert "type" not in resolved["files"]
    assert resolved["remote"]["type"] == "http"


def test_resolve_selects_a_subset(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    cfg = load_config(_project(tmp_path, _CONFIG))

    assert sorted(resolve_servers(cfg, ["files"])) == ["files"]
    assert resolve_servers(cfg, []) == {}


def test_resolve_reports_a_missing_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    cfg = load_config(_project(tmp_path, _CONFIG))

    with pytest.raises(UsageError, match=r"MCP_TOKEN"):
        resolve_servers(cfg)


def test_resolve_expands_project_root_in_stdio_command_and_args(tmp_path):
    root = _project(
        tmp_path,
        """\
        default_harness: dummy
        mcp_servers:
          files:
            command: "${PROJECT_ROOT}/bin/mcp-files"
            args: ["--root", "${PROJECT_ROOT}"]
        """,
    )
    cfg = load_config(root)

    resolved = resolve_servers(cfg)

    assert resolved["files"]["command"] == f"{root}/bin/mcp-files"
    assert resolved["files"]["args"] == ["--root", str(root)]


def test_resolve_reports_a_missing_variable_in_stdio_command(tmp_path):
    root = _project(
        tmp_path,
        """\
        default_harness: dummy
        mcp_servers:
          files:
            command: "${MCP_FILES_BIN}"
        """,
    )
    cfg = load_config(root)

    with pytest.raises(UsageError, match=r"MCP_FILES_BIN"):
        resolve_servers(cfg)


def test_resolve_reports_a_missing_variable_in_stdio_args(tmp_path):
    root = _project(
        tmp_path,
        """\
        default_harness: dummy
        mcp_servers:
          files:
            command: mcp-files
            args: ["--token", "${MCP_FILES_ARG}"]
        """,
    )
    cfg = load_config(root)

    with pytest.raises(UsageError, match=r"MCP_FILES_ARG"):
        resolve_servers(cfg)


def test_preflight_does_not_treat_project_root_as_a_missing_variable(tmp_path):
    root = _project(
        tmp_path,
        """\
        default_harness: dummy
        mcp_servers:
          files:
            command: sh
            args: ["-c", "echo ${PROJECT_ROOT}"]
        """,
    )
    cfg = load_config(root)

    results = preflight(cfg, get_provider("claude_code"))

    assert [r.status for r in results] == ["OK"]


def test_preflight_reports_a_missing_variable_in_stdio_args(tmp_path):
    root = _project(
        tmp_path,
        """\
        default_harness: dummy
        mcp_servers:
          files:
            command: sh
            args: ["-c", "echo ${MCP_FILES_ARG}"]
        """,
    )
    cfg = load_config(root)

    results = preflight(cfg, get_provider("claude_code"))

    by_name = {r.name: r for r in results}
    assert by_name["mcp server environment"].status == "FAIL"
    assert "MCP_FILES_ARG" in by_name["mcp server environment"].hint


def test_preflight_reports_missing_command_and_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    cfg = load_config(_project(tmp_path, _CONFIG))

    results = preflight(cfg, get_provider("claude_code"))

    by_name = {r.name: r for r in results}
    assert by_name["mcp server commands"].status == "FAIL"
    assert "files" in by_name["mcp server commands"].hint
    assert by_name["mcp server environment"].status == "FAIL"
    assert "MCP_TOKEN" in by_name["mcp server environment"].hint


def test_preflight_warns_when_the_harness_ignores_the_config(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    cfg = load_config(_project(tmp_path, _CONFIG))

    results = preflight(cfg, get_provider("dummy"))

    warn = next(r for r in results if r.name == "mcp servers supported")
    assert warn.status == "WARN"
    assert "dummy" in warn.hint
    # A harness that ignores mcp_servers entirely never reaches the
    # command/env checks below — they'd only ever fail over servers it
    # was never going to attach in the first place.
    assert [r.name for r in results] == ["mcp servers supported"]


def test_preflight_warns_when_the_harness_reports_no_connection_status(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    cfg = load_config(_project(tmp_path, _CONFIG))

    results = preflight(cfg, get_provider("codex_cli"))

    warn = next(r for r in results if r.name == "mcp connection status")
    assert warn.status == "WARN"
    assert "codex_cli" in warn.hint


def test_preflight_is_silent_without_servers(tmp_path):
    cfg = load_config(_project(tmp_path, "default_harness: dummy\n"))

    assert preflight(cfg, get_provider("dummy")) == []


def test_task_selects_servers(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            mcp_servers: [files]
            assertions: []
            """
        )
    )
    assert load_task(p, "s")[0].mcp_servers == ["files"]


def test_task_defaults_to_every_server(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions: []\n")

    assert load_task(p, "s")[0].mcp_servers is None


def test_validation_rejects_an_undeclared_server(tmp_path):
    root = _project(tmp_path, _CONFIG)
    (root / "evals" / "suites" / "s" / "tasks" / "t.yaml").write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            mcp_servers: [flies]
            assertions: []
            """
        )
    )
    cfg = load_config(root)

    fails = [c for c in validate_suite(cfg, "s") if c.status == "FAIL"]

    assert [c.name for c in fails] == ["s: mcp servers declared"]
    assert "flies" in fails[0].hint


_SCOPED_CONFIG = """\
default_harness: claude_code
mcp_servers:
  local:
    command: sh
  remote:
    type: http
    url: https://example.test/mcp
    headers:
      Authorization: "Bearer ${MCP_TOKEN}"
"""


def _task(tmp_path: Path, servers: str, name: str = "t"):
    p = tmp_path / f"{name}.yaml"
    p.write_text(f"kind: execute\nprompt: x\nmcp_servers: {servers}\nassertions: []\n")
    return load_task(p, "s")[0]


def test_preflight_skips_servers_no_task_selects(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    cfg = load_config(_project(tmp_path, _SCOPED_CONFIG))
    tasks = [_task(tmp_path, "[local]")]

    results = preflight(cfg, get_provider("claude_code"), tasks)

    assert [r.status for r in results] == ["OK"]


def test_preflight_covers_every_server_a_task_selects(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    cfg = load_config(_project(tmp_path, _SCOPED_CONFIG))
    tasks = [_task(tmp_path, "[local]"), _task(tmp_path, "[remote]", "b")]

    results = preflight(cfg, get_provider("claude_code"), tasks)

    by_name = {r.name: r for r in results}
    assert by_name["mcp server environment"].status == "FAIL"


def test_preflight_checks_everything_for_a_task_without_a_subset(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    cfg = load_config(_project(tmp_path, _SCOPED_CONFIG))
    p = tmp_path / "all.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions: []\n")
    tasks = [_task(tmp_path, "[local]"), load_task(p, "s")[0]]

    results = preflight(cfg, get_provider("claude_code"), tasks)

    assert any(r.status == "FAIL" for r in results)


def test_preflight_is_silent_when_no_task_selects_a_server(tmp_path):
    cfg = load_config(_project(tmp_path, _SCOPED_CONFIG))

    assert preflight(cfg, get_provider("claude_code"), [_task(tmp_path, "[]")]) == []


def test_preflight_rejects_a_codex_header_it_cannot_send_before_the_run(
    tmp_path, monkeypatch
):
    """codex_cli's own header-shape constraint surfaces at preflight, not
    only when the first attempt stages the config."""
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    root = _project(
        tmp_path,
        _SCOPED_CONFIG.replace('Authorization: "Bearer ${MCP_TOKEN}"', "X-Key: k"),
    )
    cfg = load_config(root)

    results = preflight(cfg, get_provider("codex_cli"))

    by_name = {r.name: r for r in results}
    assert by_name["mcp server configuration"].status == "FAIL"
    assert "remote" in by_name["mcp server configuration"].hint


def test_preflight_rejects_a_codex_sse_server_before_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    root = _project(tmp_path, _SCOPED_CONFIG.replace("type: http", "type: sse"))
    cfg = load_config(root)

    results = preflight(cfg, get_provider("codex_cli"))

    by_name = {r.name: r for r in results}
    assert by_name["mcp server configuration"].status == "FAIL"
    assert "sse" in by_name["mcp server configuration"].hint


_OAUTH_CONFIG = """\
default_harness: dummy
mcp_servers:
  reports:
    type: http
    url: https://reports.example.test/mcp
    oauth:
      token_url: https://auth.example.test/token
      client_id: "${REPORTS_CLIENT_ID}"
      client_secret: "${REPORTS_CLIENT_SECRET}"
      env_var: REPORTS_TOKEN
    headers:
      Authorization: "Bearer ${REPORTS_TOKEN}"
"""


class _FakeTokenResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _fake_urlopen(payload: dict):
    def urlopen(request, timeout=None):
        return _FakeTokenResponse(payload)

    return urlopen


def test_config_parses_oauth(tmp_path):
    cfg = load_config(_project(tmp_path, _OAUTH_CONFIG))

    oauth = cfg.mcp_servers["reports"].oauth
    assert oauth.token_url == "https://auth.example.test/token"
    assert oauth.env_var == "REPORTS_TOKEN"


def test_resolve_fetches_an_oauth_token_and_expands_it(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORTS_CLIENT_ID", "id")
    monkeypatch.setenv("REPORTS_CLIENT_SECRET", "secret")
    monkeypatch.delenv("REPORTS_TOKEN", raising=False)
    monkeypatch.setattr(
        "urllib.request.urlopen", _fake_urlopen({"access_token": "tok123"})
    )
    cfg = load_config(_project(tmp_path, _OAUTH_CONFIG))

    resolved = resolve_servers(cfg)

    assert resolved["reports"]["headers"] == {"Authorization": "Bearer tok123"}
    assert "oauth" not in resolved["reports"]
    assert os.environ["REPORTS_TOKEN"] == "tok123"


def _counting_urlopen(payload: dict, calls: list):
    def urlopen(request, timeout=None):
        calls.append(request.full_url)
        return _FakeTokenResponse({**payload, "access_token": f"tok{len(calls)}"})

    return urlopen


def test_resolve_records_how_long_the_oauth_token_lives(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORTS_CLIENT_ID", "id")
    monkeypatch.setenv("REPORTS_CLIENT_SECRET", "secret")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _fake_urlopen({"access_token": "tok123", "expires_in": 180}),
    )
    cfg = load_config(_project(tmp_path, _OAUTH_CONFIG))

    resolve_servers(cfg)

    assert token_lifetimes() == {"reports": 180}


def test_resolving_again_reuses_a_token_with_life_left(tmp_path, monkeypatch):
    """Every attempt resolves the servers it attaches, and a rotating grant
    revokes the whole family when two of them run it at once."""
    monkeypatch.setenv("REPORTS_CLIENT_ID", "id")
    monkeypatch.setenv("REPORTS_CLIENT_SECRET", "secret")
    calls: list[str] = []
    monkeypatch.setattr(
        "urllib.request.urlopen", _counting_urlopen({"expires_in": 180}, calls)
    )
    cfg = load_config(_project(tmp_path, _OAUTH_CONFIG))

    first = resolve_servers(cfg)["reports"]["headers"]
    again = resolve_servers(cfg)["reports"]["headers"]

    assert first == again == {"Authorization": "Bearer tok1"}
    assert len(calls) == 1


def test_a_spent_token_is_minted_again(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORTS_CLIENT_ID", "id")
    monkeypatch.setenv("REPORTS_CLIENT_SECRET", "secret")
    calls: list[str] = []
    monkeypatch.setattr(
        "urllib.request.urlopen", _counting_urlopen({"expires_in": 180}, calls)
    )
    cfg = load_config(_project(tmp_path, _OAUTH_CONFIG))

    resolve_servers(cfg)
    # Past the point the run renews at, short of the token's own expiry.
    later = time.time() + 150
    monkeypatch.setattr("agent_exam.mcp.time.time", lambda: later)
    resolved = resolve_servers(cfg)

    assert resolved["reports"]["headers"] == {"Authorization": "Bearer tok2"}
    assert len(calls) == 2


def test_resolve_reports_a_missing_oauth_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("REPORTS_CLIENT_ID", raising=False)
    monkeypatch.setenv("REPORTS_CLIENT_SECRET", "secret")
    cfg = load_config(_project(tmp_path, _OAUTH_CONFIG))

    with pytest.raises(UsageError, match=r"REPORTS_CLIENT_ID"):
        resolve_servers(cfg)


def test_resolve_reports_an_oauth_token_endpoint_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORTS_CLIENT_ID", "id")
    monkeypatch.setenv("REPORTS_CLIENT_SECRET", "secret")

    def urlopen(request, timeout=None):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    cfg = load_config(_project(tmp_path, _OAUTH_CONFIG))

    with pytest.raises(UsageError, match=r"request to"):
        resolve_servers(cfg)


def test_resolve_reports_an_oauth_response_without_a_token(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORTS_CLIENT_ID", "id")
    monkeypatch.setenv("REPORTS_CLIENT_SECRET", "secret")
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen({}))
    cfg = load_config(_project(tmp_path, _OAUTH_CONFIG))

    with pytest.raises(UsageError, match=r"no access_token"):
        resolve_servers(cfg)


def test_preflight_reports_a_missing_oauth_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("REPORTS_CLIENT_SECRET", raising=False)
    monkeypatch.setenv("REPORTS_CLIENT_ID", "id")
    cfg = load_config(_project(tmp_path, _OAUTH_CONFIG))

    results = preflight(cfg, get_provider("claude_code"))

    by_name = {r.name: r for r in results}
    assert by_name["mcp server environment"].status == "FAIL"
    assert "REPORTS_CLIENT_SECRET" in by_name["mcp server environment"].hint
    assert "REPORTS_TOKEN" not in by_name["mcp server environment"].hint


def test_preflight_passes_with_the_oauth_token_not_yet_exported(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORTS_CLIENT_ID", "id")
    monkeypatch.setenv("REPORTS_CLIENT_SECRET", "secret")
    monkeypatch.delenv("REPORTS_TOKEN", raising=False)
    cfg = load_config(_project(tmp_path, _OAUTH_CONFIG))

    results = preflight(cfg, get_provider("claude_code"))

    by_name = {r.name: r for r in results}
    assert "mcp server environment" not in by_name
    assert by_name["mcp servers"].status == "OK"


_LOGIN_CONFIG = """\
default_harness: dummy
mcp_servers:
  reports:
    url: https://reports.example.test/mcp
    oauth:
      env_var: REPORTS_TOKEN
    headers:
      Authorization: "Bearer ${REPORTS_TOKEN}"
"""


def test_config_defaults_oauth_to_the_authorization_header(tmp_path):
    config = _LOGIN_CONFIG.replace(
        "    oauth:\n      env_var: REPORTS_TOKEN\n    headers:\n"
        '      Authorization: "Bearer ${REPORTS_TOKEN}"\n',
        "    oauth: {}\n",
    )

    server = load_config(_project(tmp_path, config)).mcp_servers["reports"]

    assert server.oauth.env_var == "MCP_REPORTS_TOKEN"
    assert server.headers == {"Authorization": "Bearer ${MCP_REPORTS_TOKEN}"}


def test_config_requires_env_var_on_a_stdio_server(tmp_path):
    config = (
        _OAUTH_CONFIG.replace(
            "    type: http\n    url: https://reports.example.test/mcp",
            "    command: mcp-reports",
        )
        .replace("      env_var: REPORTS_TOKEN\n", "")
        .replace("headers:", "env:")
    )

    with pytest.raises(UsageError, match=r"set env_var"):
        load_config(_project(tmp_path, config))


def _store_login(monkeypatch, tmp_path, entry: dict | None) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    path = tmp_path / "xdg" / "agent-exam" / "mcp-oauth.json"
    if entry is not None:
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"https://reports.example.test/mcp": entry}))
    return path


def test_config_rejects_a_client_secret_without_its_client(tmp_path):
    config = _OAUTH_CONFIG.replace('      client_id: "${REPORTS_CLIENT_ID}"\n', "")

    with pytest.raises(
        UsageError, match=r"client_secret needs token_url and client_id"
    ):
        load_config(_project(tmp_path, config))


def test_config_rejects_a_token_url_without_a_client_secret(tmp_path):
    config = _LOGIN_CONFIG.replace(
        "      env_var:",
        "      token_url: https://auth.example.test/token\n      env_var:",
    )

    with pytest.raises(UsageError, match=r"token_url is discovered"):
        load_config(_project(tmp_path, config))


def test_config_rejects_a_login_on_a_stdio_server(tmp_path):
    config = _LOGIN_CONFIG.replace(
        "    url: https://reports.example.test/mcp", "    command: mcp-reports"
    ).replace("headers:", "env:")

    with pytest.raises(UsageError, match=r"no URL to log in to"):
        load_config(_project(tmp_path, config))


def test_resolve_refreshes_a_stored_login(tmp_path, monkeypatch):
    path = _store_login(
        monkeypatch,
        tmp_path,
        {
            "token_endpoint": "https://auth.example.test/token",
            "client_id": "dyn",
            "refresh_token": "ref1",
        },
    )
    monkeypatch.delenv("REPORTS_TOKEN", raising=False)
    seen = {}

    def urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["body"] = dict(urllib.parse.parse_qsl(request.data.decode()))
        return _FakeTokenResponse({"access_token": "tok", "refresh_token": "ref2"})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    cfg = load_config(_project(tmp_path, _LOGIN_CONFIG))

    resolved = resolve_servers(cfg)

    assert resolved["reports"]["headers"] == {"Authorization": "Bearer tok"}
    assert seen["url"] == "https://auth.example.test/token"
    assert seen["body"] == {
        "grant_type": "refresh_token",
        "refresh_token": "ref1",
        "client_id": "dyn",
        "resource": "https://reports.example.test/mcp",
    }
    stored = json.loads(path.read_text())["https://reports.example.test/mcp"]
    assert stored["refresh_token"] == "ref2"


def test_resolve_reports_a_missing_login(tmp_path, monkeypatch):
    _store_login(monkeypatch, tmp_path, None)
    cfg = load_config(_project(tmp_path, _LOGIN_CONFIG))

    with pytest.raises(UsageError, match=r"agent-exam mcp login reports"):
        resolve_servers(cfg)


def test_preflight_reports_a_missing_login(tmp_path, monkeypatch):
    _store_login(monkeypatch, tmp_path, None)
    cfg = load_config(_project(tmp_path, _LOGIN_CONFIG))

    results = preflight(cfg, get_provider("claude_code"))

    by_name = {r.name: r for r in results}
    assert by_name["mcp server logins"].status == "FAIL"
    assert "agent-exam mcp login reports" in by_name["mcp server logins"].hint
    assert "mcp servers" not in by_name


def test_preflight_passes_with_a_stored_login(tmp_path, monkeypatch):
    _store_login(monkeypatch, tmp_path, {"refresh_token": "ref"})
    cfg = load_config(_project(tmp_path, _LOGIN_CONFIG))

    results = preflight(cfg, get_provider("claude_code"))

    by_name = {r.name: r for r in results}
    assert "mcp server logins" not in by_name
    assert by_name["mcp servers"].status == "OK"


def test_login_registers_a_client_and_stores_the_refresh_token(tmp_path, monkeypatch):
    path = _store_login(monkeypatch, tmp_path, None)
    cfg = load_config(_project(tmp_path, _LOGIN_CONFIG))
    seen = {}

    def urlopen(request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        if (
            url
            == "https://reports.example.test/.well-known/oauth-protected-resource/mcp"
        ):
            return _FakeTokenResponse(
                {
                    "authorization_servers": ["https://auth.example.test/tenant"],
                    "scopes_supported": ["reports:read"],
                }
            )
        if (
            url
            == "https://auth.example.test/.well-known/oauth-authorization-server/tenant"
        ):
            return _FakeTokenResponse(
                {
                    "authorization_endpoint": "https://auth.example.test/authorize",
                    "token_endpoint": "https://auth.example.test/token",
                    "registration_endpoint": "https://auth.example.test/register",
                }
            )
        if url == "https://auth.example.test/register":
            seen["registration"] = json.loads(request.data)
            return _FakeTokenResponse({"client_id": "dyn"})
        if url == "https://auth.example.test/token":
            seen["token"] = dict(urllib.parse.parse_qsl(request.data.decode()))
            return _FakeTokenResponse({"access_token": "acc", "refresh_token": "ref"})
        raise urllib.error.URLError(f"unexpected {url}")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    def open_url(url):
        endpoint, _, query = url.partition("?")
        seen["authorize_endpoint"] = endpoint
        params = seen["authorize"] = dict(urllib.parse.parse_qsl(query))
        redirect = urllib.parse.urlsplit(params["redirect_uri"])

        def visit():
            conn = http.client.HTTPConnection(
                redirect.hostname, redirect.port, timeout=5
            )
            conn.request("GET", f"/callback?code=the-code&state={params['state']}")
            conn.getresponse().read()

        threading.Thread(target=visit).start()

    login(cfg, "reports", open_url=open_url, timeout=10)

    assert seen["authorize_endpoint"] == "https://auth.example.test/authorize"
    assert seen["registration"]["redirect_uris"] == [seen["authorize"]["redirect_uri"]]
    assert seen["registration"]["scope"] == "reports:read"
    assert seen["authorize"]["scope"] == "reports:read"
    assert seen["authorize"]["resource"] == "https://reports.example.test/mcp"
    digest = hashlib.sha256(seen["token"]["code_verifier"].encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    assert seen["authorize"]["code_challenge"] == challenge
    assert seen["token"]["grant_type"] == "authorization_code"
    assert seen["token"]["code"] == "the-code"
    assert seen["token"]["client_id"] == "dyn"
    stored = json.loads(path.read_text())["https://reports.example.test/mcp"]
    assert stored == {
        "token_endpoint": "https://auth.example.test/token",
        "client_id": "dyn",
        "refresh_token": "ref",
    }
