"""build_child_env keeps the framework's own Python environment out of the
harness subprocess, so a fixture project's `uv run` / bare `python` resolve
against the fixture rather than against `<repo>/.venv` or the host user's
site-packages."""

from __future__ import annotations

import os

from agent_exam.providers.child_env import apply_env_overrides, build_child_env


def _base(**extra: str) -> dict[str, str]:
    base = {
        "HOME": "/home/dev",
        "VIRTUAL_ENV": "/repo/.venv",
        "UV_RUN_RECURSION_DEPTH": "1",
        "PATH": os.pathsep.join(["/repo/.venv/bin", "/usr/local/bin", "/usr/bin"]),
    }
    base.update(extra)
    return base


def test_strips_framework_python_env():
    env = build_child_env(base=_base(PYTHONPATH="/repo/src", PYTHONHOME="/repo"))

    assert "VIRTUAL_ENV" not in env
    assert "UV_RUN_RECURSION_DEPTH" not in env
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert env["HOME"] == "/home/dev"


def test_removes_framework_venv_bin_from_path():
    env = build_child_env(base=_base())

    assert env["PATH"].split(os.pathsep) == ["/usr/local/bin", "/usr/bin"]


def test_keeps_path_intact_without_virtual_env():
    base = _base()
    del base["VIRTUAL_ENV"]

    env = build_child_env(base=base)

    assert "/repo/.venv/bin" in env["PATH"].split(os.pathsep)


def test_hides_host_user_site_packages():
    env = build_child_env(base=_base())

    assert env["PYTHONNOUSERSITE"] == "1"


def test_drop_removes_provider_specific_vars():
    env = build_child_env(base=_base(CLAUDECODE="1"), drop=("CLAUDECODE",))

    assert "CLAUDECODE" not in env


def test_task_overrides_win_over_stripping():
    """A task that deliberately sets one of the stripped vars still gets it —
    the strip is a default, not a policy."""
    env = build_child_env(
        {"VIRTUAL_ENV": "/fixture/.venv", "PYTHONNOUSERSITE": "0"},
        base=_base(),
    )

    assert env["VIRTUAL_ENV"] == "/fixture/.venv"
    assert env["PYTHONNOUSERSITE"] == "0"


def test_override_none_unsets():
    env = build_child_env({"HOME": None}, base=_base())

    assert "HOME" not in env


def test_apply_env_overrides_is_in_place_and_returns_env():
    env = {"A": "1", "B": "2"}

    result = apply_env_overrides(env, {"A": "9", "B": None, "C": "3"})

    assert result is env
    assert env == {"A": "9", "C": "3"}
