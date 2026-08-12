"""Tests for validation.py — static suite validation — plus the
load-time assertion-type check it relies on."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest
from conftest import git_init

from agent_exam.config import load_config
from agent_exam.errors import UsageError
from agent_exam.tasks import load_suite_config, load_task
from agent_exam.validation import validate_suite

if TYPE_CHECKING:
    from pathlib import Path


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "evals" / "suites" / "skill-a" / "tasks").mkdir(parents=True)
    (root / "evals" / "fixtures").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(
        "default_harness: dummy\nproviders:\n  dummy:\n    judge_model: haiku\n"
    )
    return root


def _task(root: Path, name: str, body: str) -> None:
    (root / "evals" / "suites" / "skill-a" / "tasks" / name).write_text(dedent(body))


# --- load-time assertion-type validation -----------------------------------


def test_unknown_assertion_type_rejected_at_load(tmp_path):
    """A typo'd assertion type fails at task-load time, not after an
    agent run during scoring."""
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions:\n  - judege: typo\n")
    with pytest.raises(UsageError, match="unknown assertion type"):
        load_task(p, "s")


def test_known_assertion_type_accepted_at_load(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions:\n  - judge: ok\n")
    [task] = load_task(p, "s")
    assert task.assertions[0].type == "judge"


def test_bad_assertion_config_rejected_at_load(tmp_path):
    """A malformed assertion config fails at task-load time, via the
    assertion's shared `validate` — not silently at scoring time."""
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions:
              - tool_count:
                  name: Bash
            """
        )
    )
    with pytest.raises(UsageError, match="tool_count"):
        load_task(p, "s")


def test_unknown_provider_in_assertion_rejected(tmp_path):
    """A typo'd harness name in an assertion's `providers:` meta-field
    fails at load — otherwise the assertion silently skips on every run."""
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions:
              - tool_called: Bash
                providers: [claude_codeX]
            """
        )
    )
    with pytest.raises(UsageError, match="unknown harness name"):
        load_task(p, "s")


def test_codex_cli_provider_section_and_filter_accepted(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            codex_cli:
              sandbox: workspace-write
              network_access: true
            assertions:
              - tool_called: command_execution
                providers: [codex_cli]
            """
        )
    )
    [task] = load_task(p, "s")
    assert task.provider_configs["codex_cli"].network_access is True
    assert task.assertions[0].providers == ["codex_cli"]


def test_unknown_top_level_key_rejected(tmp_path):
    """A misspelled top-level key fails at load — otherwise it's silently
    ignored (a typo'd `assertions:` → a task with zero assertions that
    'passes')."""
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertons:        # typo
              - judge: ok
            """
        )
    )
    with pytest.raises(UsageError, match=r"assertons.*Extra inputs"):
        load_task(p, "s")


def test_trigger_only_key_rejected_on_execute_task(tmp_path):
    """The schema is kind-specific (discriminated union on `kind`): a
    trigger-only key on an execute task surfaces as a forbidden extra."""
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions: []\npositive: [a]\n")
    with pytest.raises(UsageError, match=r"positive.*Extra inputs"):
        load_task(p, "s")


def test_valid_but_unused_top_level_key_accepted(tmp_path):
    """The allowlist comes from what the loader reads, not what current
    YAMLs use — `known_issue` at the top level is valid even if no
    shipped task happens to set it."""
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nknown_issue: tracked\nassertions: []\n")
    [task] = load_task(p, "s")
    assert task.known_issue == "tracked"


# --- load-time field validation --------------------------------------------


def test_setup_must_be_mapping(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions: []\nsetup: nope\n")
    with pytest.raises(UsageError, match=r"setup.*valid dictionary"):
        load_task(p, "s")


def test_unknown_setup_key_rejected(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions: []
            setup:
              fixutre: typo
            """
        )
    )
    with pytest.raises(UsageError, match=r"setup.fixutre.*Extra inputs"):
        load_task(p, "s")


def test_setup_fixture_must_be_string(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions: []
            setup:
              fixture: [a, b]
            """
        )
    )
    with pytest.raises(UsageError, match=r"setup.fixture.*valid string"):
        load_task(p, "s")


def test_timeout_seconds_rejects_non_numbers(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text('kind: execute\nprompt: x\nassertions: []\ntimeout_seconds: "60"\n')
    with pytest.raises(UsageError, match=r"timeout_seconds.*must be a number"):
        load_task(p, "s")


def test_timeout_seconds_rejects_non_positive(tmp_path):
    for bad in ("-5", "0"):
        p = tmp_path / "t.yaml"
        p.write_text(
            f"kind: execute\nprompt: x\nassertions: []\ntimeout_seconds: {bad}\n"
        )
        with pytest.raises(UsageError, match="greater than 0"):
            load_task(p, "s")


def test_timeout_seconds_accepts_float(tmp_path):
    """Floats are valid — `subprocess` timeouts accept them, and
    sub-second granularity is meaningful for fast probes."""
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions: []\ntimeout_seconds: 60.5\n")
    [task] = load_task(p, "s")
    assert task.timeout_seconds == 60.5


def test_concurrency_group_must_be_string(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions: []\nconcurrency_group: [a]\n")
    with pytest.raises(UsageError, match=r"concurrency_group.*valid string"):
        load_task(p, "s")


def test_unknown_suite_yml_key_rejected(tmp_path):
    root = _project(tmp_path)
    (root / "evals" / "suites" / "skill-a" / "suite.yml").write_text(
        "evaluated_skils: [skill-a]\n"  # typo
    )
    with pytest.raises(UsageError, match="Extra inputs"):
        load_suite_config(root / "evals", "skill-a")


# --- validate_suite --------------------------------------------------------


def test_validate_clean_suite(tmp_path):
    root = _project(tmp_path)
    _task(root, "basic.yaml", "kind: execute\nprompt: x\nassertions: []\n")
    results = validate_suite(load_config(root), "skill-a")
    assert all(r.status == "OK" for r in results)
    assert any(r.name == "skill-a: task files parse" for r in results)


def test_validate_counts_files_not_expanded_tasks(tmp_path):
    """A trigger file fans out into one task per case, but the check
    counts source files — that's the unit it actually validates."""
    root = _project(tmp_path)
    _task(root, "exec.yaml", "kind: execute\nprompt: x\nassertions: []\n")
    _task(
        root,
        "trig.yaml",
        """
        kind: trigger
        skill: skill-a
        positive: [a, b, c]
        negative: [d, e]
        """,
    )
    results = validate_suite(load_config(root), "skill-a")
    parse = next(r for r in results if r.name == "skill-a: task files parse")
    # 2 files, even though trig.yaml expands to 5 tasks.
    assert parse.hint == "2 file(s)"


def test_validate_catches_parse_error(tmp_path):
    root = _project(tmp_path)
    _task(root, "bad.yaml", "kind: execute\nprompt: x\nassertions:\n  - judege: typo\n")
    results = validate_suite(load_config(root), "skill-a")
    assert any(r.status == "FAIL" for r in results)


def test_validate_catches_bad_assertion_config(tmp_path):
    root = _project(tmp_path)
    _task(
        root,
        "bad.yaml",
        """
        kind: execute
        prompt: x
        assertions:
          - file_contains:
              path: a.py
        """,
    )
    results = validate_suite(load_config(root), "skill-a")
    assert any(r.status == "FAIL" for r in results)


def test_validate_catches_missing_fixture(tmp_path):
    root = _project(tmp_path)
    _task(
        root,
        "fix.yaml",
        """
        kind: execute
        prompt: x
        setup:
          fixture: does-not-exist
        assertions: []
        """,
    )
    results = validate_suite(load_config(root), "skill-a")
    fixture_check = [r for r in results if "fixtures exist" in r.name]
    assert fixture_check
    assert fixture_check[0].status == "FAIL"
    assert "does-not-exist" in fixture_check[0].hint


def _fixture_task(root: Path, fixture: str = "myfix") -> None:
    _task(
        root,
        "fix.yaml",
        f"""
        kind: execute
        prompt: x
        setup:
          fixture: {fixture}
        assertions: []
        """,
    )


def _git_repo(root: Path, gitignore: str) -> None:
    """A repo with the fixture tree already committed — the state a real
    checkout is in, and what `git ls-files --others` semantics assume."""
    git_init(root, gitignore, commit=True)


def _dummy_provider_config(root: Path) -> None:
    (root / "evals" / "config.yaml").write_text(
        "default_harness: dummy\nskills_dirs: []\n"
        "providers:\n  dummy:\n    judge_model: haiku\n"
    )


def _fixture_project(root: Path, fixture: str = "myfix") -> Path:
    proj = root / "evals" / "fixtures" / fixture / "proj"
    proj.mkdir(parents=True)
    (proj / "spider.py").write_text("# spider")
    return proj


def _ignored_check(results) -> object | None:
    hits = [r for r in results if "have git-ignored files" in r.name]
    return hits[0] if hits else None


def _listed(check) -> list[str]:
    """The paths a check lists, one per line after its first line."""
    return [line.strip() for line in check.hint.splitlines()[1:]]


def _rel(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def test_validate_fails_on_gitignored_file_in_fixture(tmp_path):
    """A fixture whose content only exists on one machine is not the same
    eval anywhere else, so an ignored path is a FAIL — the runner refuses
    to spend tokens on it. The listing is one path per line, collapsed to
    the `.venv` rather than each file inside it."""
    root = _project(tmp_path)
    proj = _fixture_project(root)
    _fixture_task(root)
    _git_repo(root, ".venv/\n")

    (proj / ".venv" / "bin").mkdir(parents=True)
    (proj / ".venv" / "bin" / "python").write_text("#!/bin/sh\n")
    (proj / ".venv" / "pyvenv.cfg").write_text("home = /usr\n")

    check = _ignored_check(validate_suite(load_config(root), "skill-a"))

    assert check is not None
    assert check.status == "FAIL"
    assert _listed(check) == [_rel(root, proj / ".venv")]


def test_validate_finds_ignored_content_under_a_new_subdirectory(tmp_path):
    """Regression: `git ls-files --ignored --directory` collapses a
    wholly-untracked directory, and because the collapsed directory itself
    isn't ignored it then vanishes from the listing — a `.venv` under a
    brand-new subdir would go unreported."""
    root = _project(tmp_path)
    _fixture_project(root)
    _fixture_task(root)
    _git_repo(root, ".venv/\n")

    newsub = root / "evals" / "fixtures" / "myfix" / "newsub"
    (newsub / ".venv").mkdir(parents=True)
    (newsub / ".venv" / "pyvenv.cfg").write_text("home = /usr\n")
    (newsub / "page.py").write_text("# uncommitted but not ignored")

    check = _ignored_check(validate_suite(load_config(root), "skill-a"))

    assert check is not None
    assert check.status == "FAIL"
    assert _listed(check) == [_rel(root, newsub / ".venv")]


def test_validate_does_not_collapse_past_wanted_content(tmp_path):
    """The listing must be safe to hand to `rm -rf`. `proj/` holds a
    committed file and an uncommitted one, so collapsing the ignored
    `*.pyc` files up to `proj/` would delete both — list the files."""
    root = _project(tmp_path)
    proj = _fixture_project(root)
    _fixture_task(root)
    _git_repo(root, "*.pyc\n")

    (proj / "a.pyc").write_bytes(b"\x00")
    (proj / "b.pyc").write_bytes(b"\x00")
    (proj / "in_progress.py").write_text("# uncommitted, wanted")

    check = _ignored_check(validate_suite(load_config(root), "skill-a"))

    assert check is not None
    assert check.status == "FAIL"
    assert _listed(check) == [
        _rel(root, proj / "a.pyc"),
        _rel(root, proj / "b.pyc"),
    ]


def test_validate_collapses_each_offender_independently(tmp_path):
    """Several ignored trees in one fixture each collapse to their own
    root — one line per thing to delete, at whatever depth it sits."""
    root = _project(tmp_path)
    proj = _fixture_project(root)
    _fixture_task(root)
    _git_repo(root, ".venv/\n__pycache__/\n")

    (proj / ".venv" / "lib").mkdir(parents=True)
    (proj / ".venv" / "lib" / "x.so").write_bytes(b"\x00")
    (proj / "pages" / "__pycache__").mkdir(parents=True)
    (proj / "pages" / "__pycache__" / "p.pyc").write_bytes(b"\x00")
    (proj / "pages" / "product.py").write_text("# page object")

    check = _ignored_check(validate_suite(load_config(root), "skill-a"))

    assert check is not None
    assert check.status == "FAIL"
    assert _listed(check) == [
        _rel(root, proj / ".venv"),
        _rel(root, proj / "pages" / "__pycache__"),
    ]


def _empty_dir_check(results) -> object | None:
    hits = [r for r in results if "empty directories" in r.name]
    return hits[0] if hits else None


def test_validate_fails_on_empty_directory_in_fixture(tmp_path):
    """An empty directory can't be committed, so it won't exist on a fresh
    checkout — but it *is* staged into the attempt cwd, where the agent
    sees it. Reported separately from ignored files: the fix differs."""
    root = _project(tmp_path)
    proj = _fixture_project(root)
    _fixture_task(root)
    _git_repo(root, ".venv/\n")

    (proj / "output").mkdir()

    check = _empty_dir_check(validate_suite(load_config(root), "skill-a"))

    assert check is not None
    assert check.status == "FAIL"
    assert _listed(check) == [_rel(root, proj / "output")]
    assert ".gitkeep" in check.hint


def test_validate_reports_empty_directory_chain_once(tmp_path):
    """`out/logs/` with neither holding a file is one problem — one
    `.gitkeep` or one delete fixes it, so report the shallowest."""
    root = _project(tmp_path)
    proj = _fixture_project(root)
    _fixture_task(root)
    _git_repo(root, ".venv/\n")

    (proj / "out" / "logs").mkdir(parents=True)

    check = _empty_dir_check(validate_suite(load_config(root), "skill-a"))

    assert check is not None
    assert _listed(check) == [_rel(root, proj / "out")]


def test_validate_reports_empty_ignored_directory_as_empty(tmp_path):
    """An ignored directory holding no files has nothing for the
    ignored-files check to list, so it lands in the empty-directory one.
    Adding the `.gitkeep` then trips the ignored-files check, which is the
    nudge to narrow the ignore rule."""
    root = _project(tmp_path)
    proj = _fixture_project(root)
    _fixture_task(root)
    _git_repo(root, ".venv/\n")

    (proj / ".venv" / "bin").mkdir(parents=True)

    results = validate_suite(load_config(root), "skill-a")

    empty = _empty_dir_check(results)
    assert empty is not None
    assert _listed(empty) == [_rel(root, proj / ".venv")]
    assert _ignored_check(results) is None

    (proj / ".venv" / "bin" / ".gitkeep").write_text("")

    results = validate_suite(load_config(root), "skill-a")
    assert _empty_dir_check(results) is None
    assert _ignored_check(results) is not None


def test_validate_ignores_empty_dirs_outside_the_suites_fixtures(tmp_path):
    root = _project(tmp_path)
    _fixture_project(root)
    _fixture_project(root, "otherfix")
    _fixture_task(root)
    _git_repo(root, ".venv/\n")

    (root / "evals" / "fixtures" / "otherfix" / "empty").mkdir()

    assert not [
        r for r in validate_suite(load_config(root), "skill-a") if r.status == "FAIL"
    ]


def test_validate_allows_untracked_unignored_fixture_files(tmp_path):
    """Files that are merely uncommitted must not block a run — a fixture
    is untracked while it's being authored, and evals get run before the
    commit."""
    root = _project(tmp_path)
    proj = _fixture_project(root)
    _fixture_task(root)
    _git_repo(root, ".venv/\n")

    (proj / "brand_new.py").write_text("# not committed yet")

    results = validate_suite(load_config(root), "skill-a")

    assert not [r for r in results if r.status == "FAIL"]


def test_validate_ignores_paths_outside_the_suites_fixtures(tmp_path):
    """Scoped to the fixtures this suite references — an unrelated
    fixture's cruft shouldn't block a run that doesn't stage it."""
    root = _project(tmp_path)
    _fixture_project(root)
    _fixture_project(root, "otherfix")
    _fixture_task(root)
    _git_repo(root, ".venv/\n")

    other_venv = root / "evals" / "fixtures" / "otherfix" / ".venv"
    other_venv.mkdir(parents=True)
    (other_venv / "x").write_text("x")

    results = validate_suite(load_config(root), "skill-a")

    assert not [r for r in results if r.status == "FAIL"]


def test_validate_silent_when_git_cannot_be_asked(tmp_path):
    """Not a work tree (or no git binary): the framework must work for
    projects that don't use git at all, so fixture hygiene says nothing —
    neither finding means anything without a checkout to be missing from,
    and nothing gets filtered out of the fixture either."""
    root = _project(tmp_path)  # deliberately not a git repo
    proj = _fixture_project(root)
    _fixture_task(root)

    # Both offenders present: an ignorable-looking tree and an empty dir.
    (proj / ".venv" / "bin").mkdir(parents=True)
    (proj / ".venv" / "bin" / "python").write_text("#!/bin/sh\n")
    (proj / "output").mkdir()

    results = validate_suite(load_config(root), "skill-a")

    assert all(r.status == "OK" for r in results)
    assert not [r for r in results if "fixtures have" in r.name]


def test_runner_refuses_to_run_with_gitignored_fixture_content(tmp_path, monkeypatch):
    root = _project(tmp_path)
    proj = _fixture_project(root)
    _fixture_task(root)
    _dummy_provider_config(root)
    _git_repo(root, ".venv/\n")

    (proj / ".venv").mkdir()
    (proj / ".venv" / "x").write_text("x")
    monkeypatch.chdir(root)

    from agent_exam.runner import RunRequest, run

    with pytest.raises(UsageError, match="stop ignoring them"):
        run(
            load_config(root),
            RunRequest(
                specs=[("skill-a", None)],
                provider="dummy",
                model="",
                k=1,
                n_parallel=1,
                without_skill=False,
            ),
        )

    # Refused before creating a run dir — no tokens, no artifacts.
    assert not (root / "evals" / "runs").exists()


def test_validate_passes_with_existing_fixture(tmp_path):
    root = _project(tmp_path)
    _fixture_project(root)  # a fixture with a file in it, not an empty dir
    _git_repo(root, ".venv/\n")  # so git can vet it
    _fixture_task(root)
    results = validate_suite(load_config(root), "skill-a")
    assert all(r.status == "OK" for r in results)


def test_validate_task_filter_narrows_scope(tmp_path):
    """task_filter narrows validation — a sibling task's bad fixture
    doesn't fail a single-task scope."""
    root = _project(tmp_path)
    _task(root, "good.yaml", "kind: execute\nprompt: x\nassertions: []\n")
    _task(
        root,
        "bad-fixture.yaml",
        """
        kind: execute
        prompt: x
        setup:
          fixture: missing
        assertions: []
        """,
    )
    results = validate_suite(load_config(root), "skill-a", task_filter="good")
    assert all(r.status == "OK" for r in results)


def test_validate_catches_undeclared_concurrency_group(tmp_path):
    """A `concurrency_group` not declared in config.yaml is caught by
    validate_suite — not just mid-run when the pool builds semaphores."""
    root = _project(tmp_path)
    _task(
        root,
        "t.yaml",
        """
        kind: execute
        prompt: x
        concurrency_group: not-declared
        assertions: []
        """,
    )
    results = validate_suite(load_config(root), "skill-a")
    cg = [r for r in results if "concurrency" in r.name]
    assert cg
    assert cg[0].status == "FAIL"
    assert "not-declared" in cg[0].hint


def test_validate_catches_bad_suite_yml(tmp_path):
    """A malformed suite.yml surfaces in validate_suite, not only when
    the runner loads it."""
    root = _project(tmp_path)
    _task(root, "basic.yaml", "kind: execute\nprompt: x\nassertions: []\n")
    (root / "evals" / "suites" / "skill-a" / "suite.yml").write_text(
        "evaluated_skils: [skill-a]\n"  # typo
    )
    results = validate_suite(load_config(root), "skill-a")
    suite_yml = [r for r in results if "suite.yml" in r.name]
    assert suite_yml
    assert suite_yml[0].status == "FAIL"


# --- skills_dirs default ----------------------------------------------------


def _minimal_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "evals").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\nversion = '0'\n")
    return root


def test_skills_dir_at_project_root_is_the_default(tmp_path, monkeypatch):
    root = _minimal_project(tmp_path)
    (root / "skills").mkdir()
    monkeypatch.chdir(root)

    assert load_config().skills_dirs == [root / "skills"]


def test_no_skills_dir_leaves_skills_dirs_unset(tmp_path, monkeypatch):
    root = _minimal_project(tmp_path)
    monkeypatch.chdir(root)

    assert load_config().skills_dirs is None


def test_explicit_skills_dirs_wins_over_the_default(tmp_path, monkeypatch):
    root = _minimal_project(tmp_path)
    (root / "skills").mkdir()
    (root / "elsewhere").mkdir()
    (root / "evals" / "config.yaml").write_text("skills_dirs:\n  - ./elsewhere\n")
    monkeypatch.chdir(root)

    assert load_config().skills_dirs == [root / "elsewhere"]


def test_default_does_not_lock_out_the_pre_run_hook(tmp_path, monkeypatch):
    """The default must behave like a config.yaml value, not like a
    config.local.yaml one: a pre-run hook still overrides it."""
    root = _minimal_project(tmp_path)
    (root / "skills").mkdir()
    monkeypatch.chdir(root)

    assert load_config()._skills_dirs_locked is False
