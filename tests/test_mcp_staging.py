"""Attaching MCP servers to an attempt: what gets rendered, where it lands,
and how each harness's argv or config carries it.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest

from agent_exam.config import load_config
from agent_exam.errors import UsageError
from agent_exam.providers.claude_code.provider import ClaudeCodeProvider
from agent_exam.providers.codex_cli.provider import CodexCliProvider
from agent_exam.providers.copilot_cli.doctor_probes import personal_mcp_servers
from agent_exam.providers.copilot_cli.provider import CopilotCliProvider
from agent_exam.providers.opencode.provider import OpenCodeProvider
from agent_exam.runner import RunRequest, run

_CONFIG = """\
default_harness: dummy
mcp_servers:
  files:
    command: sh
    args: ["--root", "."]
    env:
      TOKEN: "${MCP_TOKEN}"
  remote:
    type: http
    url: https://example.test/mcp
    headers:
      Authorization: "Bearer ${MCP_TOKEN}"
"""


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    root = tmp_path / "proj"
    (root / "evals").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(dedent(_CONFIG))
    return load_config(root)


def _rendered(options: dict) -> dict:
    return json.loads(Path(options["mcp_config_path"]).read_text())


def test_claude_renders_the_standard_mcp_json(cfg, tmp_path):
    options = ClaudeCodeProvider().stage_mcp_config(tmp_path, cfg)

    assert _rendered(options) == {
        "mcpServers": {
            "files": {
                "command": "sh",
                "args": ["--root", "."],
                "env": {"TOKEN": "s3cret"},
            },
            "remote": {
                "type": "http",
                "url": "https://example.test/mcp",
                "headers": {"Authorization": "Bearer s3cret"},
            },
        }
    }
    assert options["mcp_server_names"] == ["files", "remote"]


def test_claude_renders_only_the_selected_servers(cfg, tmp_path):
    options = ClaudeCodeProvider().stage_mcp_config(tmp_path, cfg, ["files"])

    assert list(_rendered(options)["mcpServers"]) == ["files"]
    assert ClaudeCodeProvider().stage_mcp_config(tmp_path, cfg, []) == {}


def test_rendered_config_lands_beside_the_attempt_cwd(cfg, tmp_path):
    """The cwd is archived into the run artifacts, so a rendered credential
    must not be written inside it."""
    attempt_cwd = tmp_path / "attempt"
    attempt_cwd.mkdir()

    options = ClaudeCodeProvider().stage_mcp_config(tmp_path, cfg)

    assert Path(options["mcp_config_path"]).parent == tmp_path
    assert list(attempt_cwd.iterdir()) == []


def test_run_keeps_the_rendered_config_out_of_the_archive(tmp_path, monkeypatch):
    """End to end: the pool hands staging the run tmp root, so nothing
    `.mcp.json` reaches the archived cwd."""
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    root = tmp_path / "proj"
    (root / "evals" / "suites" / "s" / "tasks").mkdir(parents=True)
    (root / "skills" / "skill-a").mkdir(parents=True)
    (root / "skills" / "skill-a" / "SKILL.md").write_text("# skill-a")
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(
        dedent(_CONFIG) + "providers:\n  dummy:\n    judge_model: haiku\n"
    )
    (root / "evals" / "suites" / "s" / "tasks" / "t.yaml").write_text(
        "kind: execute\nprompt: x\nassertions: []\n"
    )
    monkeypatch.setattr(
        "agent_exam.providers.dummy.DummyProvider.stage_mcp_config",
        ClaudeCodeProvider.stage_mcp_config,
        raising=False,
    )

    cfg = load_config(root)
    exit_code = run(
        cfg,
        RunRequest(
            specs=[("s", None)],
            provider="dummy",
            model="",
            k=1,
            n_parallel=1,
            without_skill=False,
            cleanup_tmp_root=False,
        ),
    )
    assert exit_code == 0

    run_dir = next(iter((root / "evals" / "runs").iterdir()))
    run_json = json.loads((run_dir / "run.json").read_text())
    tmp_root = Path(run_json["config"]["tmp_root"])
    assert list(tmp_root.glob("*.mcp.json"))
    assert not list(tmp_root.glob("*/*.mcp.json"))
    assert not list(run_dir.rglob("*.mcp.json"))


def _capture_cmd(provider, provider_options, cwd=Path("/tmp")):
    """Capture the argv a provider would spawn, without spawning it."""
    captured = {}

    class _FakePopenError(Exception):
        pass

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        raise _FakePopenError

    with (
        patch("subprocess.Popen", side_effect=fake_popen),
        pytest.raises(_FakePopenError),
    ):
        provider._invoke_once(
            prompt="x",
            model="",
            cwd=cwd,
            provider_options=provider_options,
            stop_on_first_skill=False,
            timeout_seconds=30,
        )
    return captured


def test_claude_argv_carries_the_config_path():
    cmd = _capture_cmd(
        ClaudeCodeProvider(), {"mcp_config_path": Path("/tmp/x.mcp.json")}
    )["cmd"]

    assert cmd[cmd.index("--mcp-config") + 1] == "/tmp/x.mcp.json"
    assert "--strict-mcp-config" in cmd


def test_claude_is_strict_even_without_servers():
    """Otherwise the developer's own `~/.claude.json` servers load in."""
    cmd = _capture_cmd(ClaudeCodeProvider(), {})["cmd"]

    assert "--strict-mcp-config" in cmd
    assert "--mcp-config" not in cmd


def test_claude_allowlist_pre_approves_each_server():
    cmd = _capture_cmd(
        ClaudeCodeProvider(),
        {"allowed_tools": ["Read"], "mcp_server_names": ["files", "remote"]},
    )["cmd"]

    assert cmd[cmd.index("--allowed-tools") + 1] == (
        "Read,Skill,mcp__files,mcp__remote"
    )


def test_claude_allowlist_is_not_created_by_servers_alone():
    """No allowlist means every tool is pre-approved already; inventing one
    here would restrict the run instead of widening it."""
    cmd = _capture_cmd(ClaudeCodeProvider(), {"mcp_server_names": ["files"]})["cmd"]

    assert "--allowed-tools" not in cmd


def test_opencode_ships_servers_through_its_config(cfg, tmp_path):
    options = OpenCodeProvider().stage_mcp_config(tmp_path, cfg)

    assert options["mcp_config"] == {
        "files": {
            "type": "local",
            "command": ["sh", "--root", "."],
            "enabled": True,
            "environment": {"TOKEN": "s3cret"},
        },
        "remote": {
            "type": "remote",
            "url": "https://example.test/mcp",
            "enabled": True,
            "headers": {"Authorization": "Bearer s3cret"},
        },
    }

    env = _capture_cmd(OpenCodeProvider(), options, cwd=tmp_path)["env"]
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["mcp"] == options["mcp_config"]


def test_codex_passes_servers_as_config_overrides(cfg, tmp_path):
    options = CodexCliProvider().stage_mcp_config(tmp_path, cfg, ["files"])

    cmd = CodexCliProvider()._build_cmd(
        prompt="x", model="", cwd=tmp_path, provider_options=options
    )

    assert 'mcp_servers.files.command="sh"' in cmd
    assert 'mcp_servers.files.args=["--root", "."]' in cmd
    assert 'mcp_servers.files.env.TOKEN="s3cret"' in cmd


def test_codex_refuses_an_http_server_with_headers(cfg, tmp_path):
    with pytest.raises(UsageError, match=r"remote.*headers"):
        CodexCliProvider().stage_mcp_config(tmp_path, cfg, ["remote"])


def test_copilot_argv_carries_the_config_path(cfg, tmp_path):
    options = CopilotCliProvider().stage_mcp_config(tmp_path, cfg)

    cmd = _capture_cmd(CopilotCliProvider(), options, cwd=tmp_path)["cmd"]

    assert "--disable-builtin-mcps" in cmd
    assert cmd[cmd.index("--additional-mcp-config") + 1] == (
        f"@{options['mcp_config_path']}"
    )


def test_copilot_disables_the_developers_own_servers(cfg, tmp_path, monkeypatch):
    """Copilot CLI augments its own config rather than replacing it, so the
    personal servers have to be turned off one by one."""
    personal = tmp_path / "mcp-config.json"
    personal.write_text(json.dumps({"mcpServers": {"personal": {}, "files": {}}}))
    monkeypatch.setattr(
        "agent_exam.providers.copilot_cli.provider.personal_mcp_servers",
        lambda: personal_mcp_servers(personal),
    )
    options = CopilotCliProvider().stage_mcp_config(tmp_path, cfg, ["files"])

    cmd = _capture_cmd(CopilotCliProvider(), options, cwd=tmp_path)["cmd"]

    assert cmd[cmd.index("--disable-mcp-server") + 1] == "personal"
    # `files` is ours this run, so disabling it would disable what the task
    # is about.
    assert "files" not in cmd
