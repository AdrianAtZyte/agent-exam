"""A worker that dies for an unexpected reason costs its own attempt and
nothing else: `run_plan` yields an `error` outcome for it and keeps draining
the rest of the batch, so the run still writes its report."""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_exam.errors import RateLimitExhausted
from agent_exam.pool import AttemptOutcome, PoolPlan, run_plan


class _FakePool:
    """Stands in for ProcessPoolExecutor: runs nothing, resolves each
    submission from `outcomes` keyed by task name."""

    def __init__(self, outcomes, **kwargs):
        self.outcomes = outcomes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, _fn, task, attempt_n, *args):
        fut: Future = Future()
        result = self.outcomes[task.name]
        if isinstance(result, Exception):
            fut.set_exception(result)
        else:
            fut.set_result(result)
        return fut


def _plan(*names):
    tasks = [
        SimpleNamespace(name=name, suite="s", concurrency_group=None) for name in names
    ]
    return PoolPlan(tasks=tasks, attempts_per_task=1, n_parallel=2)


def _run(monkeypatch, plan, outcomes):
    monkeypatch.setattr(
        "agent_exam.pool.ProcessPoolExecutor",
        lambda **kwargs: _FakePool(outcomes),
    )
    paths = SimpleNamespace(
        attempt_cwd=lambda suite, task, n: Path("/archive") / suite / task / str(n)
    )
    return run_plan(
        SimpleNamespace(),
        plan,
        Path("/tmp-root"),
        "dummy",
        "model",
        paths,
    )


def _ok(name):
    return AttemptOutcome(
        suite="s",
        task_name=name,
        attempt_n=1,
        attempt_cwd=Path("/archive") / name,
        run_result=None,
        error_verdict=None,
    )


def test_worker_crash_costs_only_its_own_attempt(monkeypatch):
    plan = _plan("boom", "fine")
    outcomes = {"boom": OSError("archiving blew up"), "fine": _ok("fine")}

    results = {o.task_name: o for o in _run(monkeypatch, plan, outcomes)}

    assert results["boom"].error_verdict == "error"
    assert results["boom"].run_result is None
    assert results["fine"].error_verdict is None


def test_framework_errors_still_abort_the_run(monkeypatch):
    plan = _plan("exhausted")
    outcomes = {"exhausted": RateLimitExhausted("no budget left")}

    with pytest.raises(RateLimitExhausted):
        list(_run(monkeypatch, plan, outcomes))
