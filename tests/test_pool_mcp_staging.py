"""Memoization of a run's MCP config: staged once per (provider, run tmp
root, server set) regardless of the order the servers were asked for, staged
in the parent and handed to the workers rather than once per process, and
forgotten once the run ends so a process that drives the pool more than once
doesn't accumulate one dead entry per run forever. A set holding an OAuth
server is the exception, staged afresh for every attempt so that none of them
runs on a token already spent.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from agent_exam.pool import (
    _MCP_STAGING,
    PoolPlan,
    _init_worker,
    _mcp_options,
    _warn_short_lived_tokens,
    forget_mcp_staging,
    run_plan,
)


@dataclass
class _FakeProvider:
    name: str = "fake"
    calls: list[list[str] | None] = field(default_factory=list)

    def stage_mcp_config(self, run_tmp_root, cfg, servers):
        self.calls.append(servers)
        return {"mcp_server_names": sorted(servers) if servers else []}


def _cfg(*names, oauth=()):
    """A config stub whose servers are *names*, those in *oauth* carrying an
    ``oauth`` block."""
    return SimpleNamespace(
        mcp_servers={
            name: SimpleNamespace(oauth=object() if name in oauth else None)
            for name in names
        }
    )


def test_the_same_server_set_in_a_different_order_hits_the_cache(tmp_path):
    provider = _FakeProvider()

    _mcp_options(provider, tmp_path, _cfg("a", "b"), ["a", "b"])
    _mcp_options(provider, tmp_path, _cfg("a", "b"), ["b", "a"])

    assert len(provider.calls) == 1
    forget_mcp_staging(tmp_path)


def test_forget_mcp_staging_drops_only_that_run(tmp_path):
    provider = _FakeProvider()
    other_root = tmp_path / "other"
    other_root.mkdir()

    _mcp_options(provider, tmp_path, _cfg("a"), ["a"])
    _mcp_options(provider, other_root, _cfg("a"), ["a"])

    forget_mcp_staging(tmp_path)

    assert not any(key[1] == tmp_path for key in _MCP_STAGING)
    assert any(key[1] == other_root for key in _MCP_STAGING)

    forget_mcp_staging(other_root)


class _CountingPool:
    """Stands in for ProcessPoolExecutor, recording what the initializer
    would have been given and resolving every submission to nothing."""

    initargs: tuple = ()

    def __init__(self, **kwargs):
        type(self).initargs = kwargs["initargs"]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, _fn, task, attempt_n, *args):
        fut: Future = Future()
        fut.set_result(None)
        return fut


def test_a_parallel_run_stages_once_for_every_worker(tmp_path, monkeypatch):
    """Rendering the config of a set without an OAuth server is the same work
    whichever attempt asks for it, so the run does it once and the workers
    read the result out of the memo."""
    provider = _FakeProvider(name="dummy")
    monkeypatch.setattr(
        "agent_exam.pool.get_provider", lambda name: provider, raising=True
    )
    monkeypatch.setattr(
        "agent_exam.pool.ProcessPoolExecutor", _CountingPool, raising=True
    )
    tasks = [
        SimpleNamespace(name=name, suite="s", concurrency_group=None, mcp_servers=["a"])
        for name in ("one", "two", "three")
    ]
    plan = PoolPlan(tasks=tasks, attempts_per_task=4, n_parallel=3)

    list(
        run_plan(
            _cfg("a"),
            plan,
            tmp_path,
            "dummy",
            "model",
            SimpleNamespace(attempt_cwd=lambda suite, task, n: Path("/archive")),
        )
    )

    assert provider.calls == [["a"]]

    # And a worker, starting out with the empty memo of a fresh process,
    # re-stages nothing once the initializer hands it the parent's.
    memo = _CountingPool.initargs[1]
    forget_mcp_staging(tmp_path)
    _init_worker({}, memo)
    _mcp_options(provider, tmp_path, _cfg("a"), ["a"])

    assert provider.calls == [["a"]]
    forget_mcp_staging(tmp_path)


def _timed_plan(*timeouts):
    tasks = [
        SimpleNamespace(kind="execute", timeout_seconds=t, mcp_servers=None)
        for t in timeouts
    ]
    return PoolPlan(tasks=tasks, attempts_per_task=1, n_parallel=1)


def _held_token(monkeypatch, lifetime):
    monkeypatch.setattr(
        "agent_exam.mcp._TOKENS",
        {"zyte": {"token": "t", "lifetime": lifetime, "renew_at": None}},
    )


def test_a_token_that_lapses_within_a_task_budget_is_called_out(monkeypatch, capsys):
    """An attempt starts on a fresh token but keeps it to the end, so a task
    with room to outlive one has its tool calls rejected part-way through."""
    _held_token(monkeypatch, 180)

    _warn_short_lived_tokens(
        SimpleNamespace(default_task_timeout_seconds=300),
        _timed_plan(60, 900),
        {"zyte"},
    )

    warning = capsys.readouterr().err
    assert "zyte (180s)" in warning
    assert "900s task budget" in warning


def test_a_token_outliving_every_task_says_nothing(monkeypatch, capsys):
    _held_token(monkeypatch, 3600)

    _warn_short_lived_tokens(
        SimpleNamespace(default_task_timeout_seconds=300), _timed_plan(900), {"zyte"}
    )

    assert capsys.readouterr().err == ""


def test_a_server_this_run_leaves_out_says_nothing(monkeypatch, capsys):
    """The cache outlives the run that filled it, so a second run in the same
    process must not be warned about a server it does not attach."""
    _held_token(monkeypatch, 180)

    _warn_short_lived_tokens(
        SimpleNamespace(default_task_timeout_seconds=300), _timed_plan(900), set()
    )

    assert capsys.readouterr().err == ""
