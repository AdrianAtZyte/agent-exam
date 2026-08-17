"""Tests for task tags: declaration, union, and run-time selection."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from agent_exam.config import load_config
from agent_exam.errors import UsageError
from agent_exam.tasks import Task, load_suite, select_by_tags
from agent_exam.validation import validate_suite


def _task(suite: str, name: str, tags=(), source: str | None = None) -> Task:
    return Task(
        suite=suite,
        name=name,
        kind="execute",
        prompt="do the thing",
        description=None,
        assertions=[],
        fixture=None,
        env={},
        timeout_seconds=None,
        concurrency_group=None,
        raw={},
        source_path=Path(f"{source or name}.yaml"),
        tags=sorted(tags),
    )


def _project(tmp_path, *, config: str, suites: dict) -> Path:
    """Write a minimal project. *suites* maps a suite name to
    ``{"suite.yml": <text>, "tasks": {<name>: <text>}}``."""
    root = tmp_path / "proj"
    (root / "evals" / "fixtures").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(
        dedent(
            """\
            default_harness: dummy
            skills_dirs: []
            providers:
              dummy:
                judge_model: haiku
            """
        )
        + config
    )
    for suite, content in suites.items():
        suite_dir = root / "evals" / "suites" / suite
        (suite_dir / "tasks").mkdir(parents=True)
        if "suite.yml" in content:
            (suite_dir / "suite.yml").write_text(dedent(content["suite.yml"]))
        for name, body in content["tasks"].items():
            (suite_dir / "tasks" / f"{name}.yaml").write_text(dedent(body))
    return root


_TASK_YAML = "prompt: do the thing\nassertions: []\n"


# --- declaration and union -------------------------------------------------


def test_suite_tags_union_with_task_tags(tmp_path):
    root = _project(
        tmp_path,
        config="tags:\n  expensive: {exclude_by_default: true}\n  network: {}\n",
        suites={
            "s": {
                "suite.yml": "tags: [expensive]\n",
                "tasks": {
                    "plain": _TASK_YAML,
                    "tagged": _TASK_YAML + "tags: [network]\n",
                },
            }
        },
    )
    tasks = {t.name: t.tags for t in load_suite(root / "evals", "s")}
    assert tasks == {"plain": ["expensive"], "tagged": ["expensive", "network"]}


def test_trigger_cases_inherit_the_file_tags(tmp_path):
    root = _project(
        tmp_path,
        config="tags:\n  network: {}\n",
        suites={
            "s": {
                "tasks": {
                    "trig": """\
                        kind: trigger
                        skill: s
                        tags: [network]
                        positive: [ping, pong]
                    """
                }
            }
        },
    )
    tasks = load_suite(root / "evals", "s")
    assert [t.tags for t in tasks] == [["network"], ["network"]]


def test_undeclared_tag_fails_validation(tmp_path, monkeypatch):
    root = _project(
        tmp_path,
        config="tags:\n  expensive: {exclude_by_default: true}\n",
        suites={"s": {"tasks": {"t": _TASK_YAML + "tags: [expensiv]\n"}}},
    )
    monkeypatch.chdir(root)
    fails = [c for c in validate_suite(load_config(root), "s") if c.status == "FAIL"]
    assert [c.name for c in fails] == ["s: tags declared"]
    assert "expensiv" in fails[0].hint


# --- selection -------------------------------------------------------------

_SPECS = [("cheap", None), ("pricey", None)]


def _tasks():
    return [
        _task("cheap", "quick"),
        _task("pricey", "slow", ["expensive"]),
        _task("pricey", "remote", ["expensive", "remote-account"]),
    ]


def test_default_excluded_tags_drop_tasks_across_suites():
    kept, dropped = select_by_tags(
        _tasks(), _SPECS, default_excluded=["expensive", "remote-account"]
    )
    assert [t.name for t in kept] == ["quick"]
    # Each task counts once, under its first tag.
    assert dropped == {"expensive": 2}


def test_single_target_suite_lifts_the_tags_the_suite_itself_declares():
    kept, dropped = select_by_tags(
        _tasks()[1:],
        [("pricey", None)],
        default_excluded=["expensive", "remote-account"],
        suite_tags={"pricey": ["expensive"]},
    )
    # `expensive` is what the suite is, so naming it asks for those tasks;
    # the one also tagged `remote-account` stays out.
    assert [t.name for t in kept] == ["slow"]
    assert dropped == {"remote-account": 1}


def test_single_target_suite_still_drops_its_individually_tagged_tasks():
    kept, dropped = select_by_tags(
        _tasks()[1:], [("pricey", None)], default_excluded=["expensive"]
    )
    assert [t.name for t in kept] == []
    assert dropped == {"expensive": 2}


def test_named_task_survives_its_suite_being_excluded():
    kept, _ = select_by_tags(
        _tasks(), [("cheap", None), ("pricey", "slow")], default_excluded=["expensive"]
    )
    assert [t.name for t in kept] == ["quick", "slow"]


def test_naming_a_trigger_file_exempts_all_its_cases():
    tasks = [
        _task("cheap", "quick"),
        _task("pricey", "trig-0", ["expensive"], source="trig"),
        _task("pricey", "trig-1", ["expensive"], source="trig"),
    ]
    kept, _ = select_by_tags(
        tasks, [("cheap", None), ("pricey", "trig")], default_excluded=["expensive"]
    )
    assert [t.name for t in kept] == ["quick", "trig-0", "trig-1"]


def test_include_lifts_one_tag_only():
    kept, dropped = select_by_tags(
        _tasks(),
        _SPECS,
        default_excluded=["expensive", "remote-account"],
        include=["expensive"],
    )
    assert [t.name for t in kept] == ["quick", "slow"]
    assert dropped == {"remote-account": 1}


def test_all_tags_lifts_every_default_exclusion():
    kept, dropped = select_by_tags(
        _tasks(),
        _SPECS,
        default_excluded=["expensive", "remote-account"],
        all_tags=True,
    )
    assert [t.name for t in kept] == ["quick", "slow", "remote"]
    assert dropped == {}


def test_exclude_tag_applies_even_to_a_named_task():
    kept, dropped = select_by_tags(
        _tasks(),
        [("cheap", None), ("pricey", "slow")],
        default_excluded=[],
        exclude=["expensive"],
    )
    assert [t.name for t in kept] == ["quick"]
    assert dropped == {"expensive": 2}


def test_exclude_tag_applies_to_a_tag_the_suite_declares():
    kept, _ = select_by_tags(
        _tasks()[1:],
        [("pricey", None)],
        default_excluded=[],
        suite_tags={"pricey": ["expensive"]},
        exclude=["expensive"],
    )
    assert [t.name for t in kept] == []


def test_all_tags_does_not_undo_exclude_tag():
    kept, _ = select_by_tags(
        _tasks(),
        _SPECS,
        default_excluded=["expensive"],
        exclude=["remote-account"],
        all_tags=True,
    )
    assert [t.name for t in kept] == ["quick", "slow"]


# --- the runner end to end -------------------------------------------------


def _tagged_project(tmp_path):
    """`cheap` wears no tag but holds one tagged task; `pricey` is tagged
    whole."""
    return _project(
        tmp_path,
        config=(
            "tags:\n"
            "  expensive: {exclude_by_default: true}\n"
            "  remote-account: {exclude_by_default: true}\n"
        ),
        suites={
            "cheap": {
                "tasks": {
                    "quick": _TASK_YAML,
                    "remote": _TASK_YAML + "tags: [remote-account]\n",
                }
            },
            "pricey": {
                "suite.yml": "tags: [expensive]\n",
                "tasks": {"slow": _TASK_YAML},
            },
        },
    )


def _tasks_ran(run_dir):
    return sorted(
        f"{suite.name}::{task.name}"
        for suite in (run_dir / "artifacts").iterdir()
        for task in suite.iterdir()
    )


def _run(root, **kwargs):
    from agent_exam.runner import RunRequest, run

    return run(
        load_config(root),
        RunRequest(provider="dummy", model="", k=1, n_parallel=1, **kwargs),
    )


def _last_run_json(root):
    run_dir = max((root / "evals" / "runs").iterdir())
    return json.loads((run_dir / "run.json").read_text()), run_dir


def test_wildcard_run_skips_every_tagged_task(tmp_path, monkeypatch):
    root = _tagged_project(tmp_path)
    monkeypatch.chdir(root)
    _run(root, specs=[("*", None)], without_skill=False, no_skills=False)

    rj, run_dir = _last_run_json(root)
    assert rj["config"]["tasks_excluded_by_tag"] == {
        "expensive": 1,
        "remote-account": 1,
    }
    assert _tasks_ran(run_dir) == ["cheap::quick"]


def test_naming_the_tagged_suite_runs_it(tmp_path, monkeypatch):
    root = _tagged_project(tmp_path)
    monkeypatch.chdir(root)
    _run(root, specs=[("pricey", None)], without_skill=False, no_skills=False)

    rj, run_dir = _last_run_json(root)
    assert rj["config"]["tasks_excluded_by_tag"] == {}
    assert _tasks_ran(run_dir) == ["pricey::slow"]


def test_naming_an_untagged_suite_still_skips_its_tagged_task(tmp_path, monkeypatch):
    root = _tagged_project(tmp_path)
    monkeypatch.chdir(root)
    _run(root, specs=[("cheap", None)], without_skill=False, no_skills=False)

    rj, run_dir = _last_run_json(root)
    assert rj["config"]["tasks_excluded_by_tag"] == {"remote-account": 1}
    assert _tasks_ran(run_dir) == ["cheap::quick"]


def test_naming_the_tagged_task_itself_runs_it(tmp_path, monkeypatch):
    root = _tagged_project(tmp_path)
    monkeypatch.chdir(root)
    _run(root, specs=[("cheap", "remote")], without_skill=False, no_skills=False)

    _, run_dir = _last_run_json(root)
    assert _tasks_ran(run_dir) == ["cheap::remote"]


def test_all_tags_runs_everything(tmp_path, monkeypatch):
    root = _tagged_project(tmp_path)
    monkeypatch.chdir(root)
    _run(
        root,
        specs=[("*", None)],
        without_skill=False,
        no_skills=False,
        all_tags=True,
    )

    rj, run_dir = _last_run_json(root)
    assert rj["config"]["tasks_excluded_by_tag"] == {}
    assert _tasks_ran(run_dir) == ["cheap::quick", "cheap::remote", "pricey::slow"]


def test_unknown_tag_on_the_command_line_is_an_error(tmp_path, monkeypatch):
    root = _tagged_project(tmp_path)
    monkeypatch.chdir(root)
    with pytest.raises(UsageError, match="unknown tag"):
        _run(
            root,
            specs=[("*", None)],
            without_skill=False,
            no_skills=False,
            tags=["expensiv"],
        )


def test_excluding_everything_names_the_flag_that_brings_it_back(tmp_path, monkeypatch):
    root = _tagged_project(tmp_path)
    monkeypatch.chdir(root)
    with pytest.raises(UsageError, match=r"--all-tags"):
        _run(
            root,
            specs=[("pricey", None)],
            without_skill=False,
            no_skills=False,
            exclude_tags=["expensive"],
        )
