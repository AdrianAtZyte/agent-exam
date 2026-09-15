import io
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_exam.errors import FrameworkError
from agent_exam.providers.claude_code.provider import ClaudeCodeProvider


class _FakeProcess:
    """A `claude -p` that exits at once, keeping what it was fed on stdin."""

    returncode = 0

    def __init__(self):
        self.stdin = io.BytesIO()
        self.stdin.close = lambda: None
        self.stdout = io.BytesIO()
        self.stderr = io.BytesIO()

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0


def test_prompt_is_fed_over_stdin_not_argv():
    prompt = "x" * 200_000
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["stdin"] = kwargs["stdin"]
        captured["process"] = _FakeProcess()
        return captured["process"]

    with (
        patch("subprocess.Popen", side_effect=fake_popen),
        pytest.raises(FrameworkError, match="session_id"),
    ):
        ClaudeCodeProvider()._invoke_once(
            prompt=prompt,
            model="",
            cwd=Path("/tmp"),
            provider_options={},
            stop_on_first_skill=False,
            timeout_seconds=30,
        )
    assert prompt not in captured["cmd"]
    assert captured["stdin"] is subprocess.PIPE
    assert captured["process"].stdin.getvalue() == prompt.encode()
