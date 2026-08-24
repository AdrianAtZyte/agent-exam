from __future__ import annotations

import subprocess

import pytest

from agent_exam.providers.process_utils import wait_or_terminate


class _InterruptOnce:
    """Wraps a real Popen, raising KeyboardInterrupt on its first `wait()`.

    Later calls (made by `terminate_tree` while cleaning up after the
    interrupt) delegate to the real process, so the kill it triggers
    actually reaches something.
    """

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process
        self._raised = False

    @property
    def pid(self) -> int:
        return self._process.pid

    def wait(self, timeout: float | None = None) -> int:
        if not self._raised:
            self._raised = True
            raise KeyboardInterrupt
        return self._process.wait(timeout=timeout)


def test_wait_or_terminate_returns_false_on_clean_exit():
    process = subprocess.Popen(["sleep", "0"], start_new_session=True)
    assert wait_or_terminate(process, 10) is False
    assert process.returncode == 0


def test_wait_or_terminate_kills_on_timeout():
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        assert wait_or_terminate(process, 0.05) is True
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_wait_or_terminate_kills_and_reraises_on_interrupt():
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        with pytest.raises(KeyboardInterrupt):
            wait_or_terminate(_InterruptOnce(process), 30)
        assert process.wait(timeout=10) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
