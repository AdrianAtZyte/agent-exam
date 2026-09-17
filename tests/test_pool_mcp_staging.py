"""Memoization of a run's MCP config: staged once per (provider, run tmp
root, server set) regardless of the order the servers were asked for, staged
in the parent and handed to the workers rather than once per process, and
forgotten once the run ends so a process that drives the pool more than once
doesn't accumulate one dead entry per run forever.
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


def test_the_same_server_set_in_a_different_order_hits_the_cache(tmp_path):
    provider = _FakeProvider()

    _mcp_options(provider, tmp_path, None, ["a", "b"])
    _mcp_options(provider, tmp_path, None, ["b", "a"])

    assert len(provider.calls) == 1
    forget_mcp_staging(tmp_path)


def test_forget_mcp_staging_drops_only_that_run(tmp_path):
    provider = _FakeProvider()
    other_root = tmp_path / "other"
    other_root.mkdir()

    _mcp_options(provider, tmp_path, None, ["a"])
    _mcp_options(provider, other_root, None, ["a"])

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
    """A server carrying `oauth` refreshes its stored login while staging,
    and an authorization server that rotates refresh tokens revokes the
    family when a second process presents the superseded one."""
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
            SimpleNamespace(),
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
    _mcp_options(provider, tmp_path, None, ["a"])

    assert provider.calls == [["a"]]
    forget_mcp_staging(tmp_path)
