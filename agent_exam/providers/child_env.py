from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_LEAKED_VARS = (
    "VIRTUAL_ENV",
    "UV_PROJECT_ENVIRONMENT",
    "UV_RUN_RECURSION_DEPTH",
    "PYTHONPATH",
    "PYTHONHOME",
)
"""Variables removed from the harness subprocess environment.

``UV_RUN_RECURSION_DEPTH`` is uv's own nested-``uv run`` guard — inherited from
the ``uv run agent-exam`` that started the framework, it makes the agent's first
``uv run`` look like a nested one."""


def _strip_venv_from_path(path: str, venv: str | None) -> str:
    """Drop ``<venv>/bin`` (and ``/Scripts``) entries from a PATH string."""
    if not venv:
        return path
    normalized = {os.path.normpath(Path(venv) / sub) for sub in ("bin", "Scripts")}
    return os.pathsep.join(
        entry
        for entry in path.split(os.pathsep)
        if entry and os.path.normpath(entry) not in normalized
    )


def apply_env_overrides(
    env: dict[str, str], env_overrides: Mapping[str, str | None] | None
) -> dict[str, str]:
    """Apply a task's *env_overrides* to *env* in place; ``None`` means unset."""
    for key, value in (env_overrides or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def build_child_env(
    env_overrides: Mapping[str, str | None] | None = None,
    *,
    drop: Iterable[str] = (),
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment for a harness subprocess.

    Every provider spawns the host agent as a subprocess, which inherits this
    process's environment. `agent-exam` itself normally runs under ``uv run``
    from a repo checkout, which leaks the *framework's* Python environment into
    the agent's shell: ``VIRTUAL_ENV`` and ``PATH`` point at ``<repo>/.venv``,
    so a bare ``python``, ``pytest`` or ``scrapy`` inside a fixture project
    resolves to the framework's interpreter rather than the fixture's, and the
    host user's ``~/.local/lib/pythonX.Y/site-packages`` is visible to ``pip
    list``. Both make attempts non-reproducible across machines and waste agent
    turns on environment archaeology a real user wouldn't hit.

    Starts from *base* (default: ``os.environ``), removes those variables (see
    ``_LEAKED_VARS``), sets ``PYTHONNOUSERSITE``, drops any provider-specific
    *drop* keys, then applies *env_overrides* — where a ``None`` value means
    "unset this variable", and which comes last so a task can still deliberately
    set any of the above.
    """
    source = os.environ if base is None else base
    env = {k: v for k, v in source.items() if k not in _LEAKED_VARS}

    for key in drop:
        env.pop(key, None)

    venv = source.get("VIRTUAL_ENV")
    if venv and "PATH" in env:
        env["PATH"] = _strip_venv_from_path(env["PATH"], venv)

    # Keep the host user's ~/.local/lib/pythonX.Y/site-packages out of sys.path
    # (and out of `pip list`/`pip show`). Project work happens through `uv run`,
    # which uses the project venv and is unaffected.
    env["PYTHONNOUSERSITE"] = "1"

    return apply_env_overrides(env, env_overrides)
