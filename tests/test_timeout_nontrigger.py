"""A positive trigger can only report "skill never fired" as a timeout,
so `_settled_nontrigger` decides when that timeout is really a fail.
"""

from __future__ import annotations

from pathlib import Path

from agent_exam.pool import _settled_nontrigger
from agent_exam.schemas import Metrics, RunResult, SkillInvocation, Tokens, Turn
from agent_exam.tasks import Task


def _run(n_tool_calls: int, skills: tuple[str, ...] = ()) -> RunResult:
    turn = Turn(
        role="assistant",
        content=[],
        skill_invocations=[
            SkillInvocation(skill_name=s, trigger_kind="skill_tool") for s in skills
        ],
    )
    return RunResult(
        trajectory=[turn],
        metrics=Metrics(
            wall_time_seconds=60.0,
            tokens=Tokens(),
            cost_usd=None,
            peak_context=0,
            turn_count=1,
            n_tool_calls=n_tool_calls,
        ),
    )


def _task(should_trigger: bool | None) -> Task:
    return Task(
        suite="scrapy",
        name="t-0",
        kind="trigger" if should_trigger is not None else "execute",
        prompt="Debug this Scrapy pipeline.",
        description=None,
        assertions=[],
        fixture=None,
        env={},
        timeout_seconds=None,
        concurrency_group=None,
        raw={},
        source_path=Path("/tmp/t.yaml"),
        should_trigger=should_trigger,
    )


def test_positive_that_worked_without_a_skill_is_settled():
    assert _settled_nontrigger(_task(True), _run(n_tool_calls=7))


def test_cold_start_timeout_is_not_settled():
    """No tool ran — the agent never got to route, so the timeout stands."""
    assert not _settled_nontrigger(_task(True), _run(n_tool_calls=0))


def test_skill_fire_is_not_settled():
    assert not _settled_nontrigger(_task(True), _run(7, ("scrapy",)))


def test_negative_trigger_and_execute_tasks_keep_the_timeout():
    assert not _settled_nontrigger(_task(False), _run(n_tool_calls=7))
    assert not _settled_nontrigger(_task(None), _run(n_tool_calls=7))


def test_no_partial_trajectory_keeps_the_timeout():
    assert not _settled_nontrigger(_task(True), None)
