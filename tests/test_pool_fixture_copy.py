"""Fixture staging copies the fixture into the attempt cwd verbatim, minus the
`.gitkeep` markers that only exist to store empty directories in git.

Nothing else is filtered. Where a project keeps its fixtures in git,
`validate_suite` refuses the run outright if a staged fixture holds anything
git ignores, so cruft can't get this far; where it doesn't, copying the fixture
as-is is the intended behaviour."""

from __future__ import annotations

from agent_exam.pool import _copy_fixture


def test_copy_fixture_copies_everything_else(tmp_path):
    """Including paths that *look* like build artifacts — filtering by name
    would silently change the scenario for a project that doesn't use git,
    where nothing has vetted the fixture and nothing should."""
    src = tmp_path / "fixture"
    dst = tmp_path / "cwd"

    (src / "proj").mkdir(parents=True)
    (src / "proj" / "spider.py").write_text("# spider")
    (src / "pyproject.toml").write_text("[project]\nname = 'proj'\n")
    (src / "samples" / "nested").mkdir(parents=True)
    (src / "samples" / "nested" / "page.html").write_text("<html></html>")
    (src / "proj" / ".venv" / "bin").mkdir(parents=True)
    (src / "proj" / ".venv" / "bin" / "python").write_text("#!/bin/sh\n")

    _copy_fixture(src, dst)

    assert (dst / "proj" / "spider.py").read_text() == "# spider"
    assert (dst / "pyproject.toml").exists()
    assert (dst / "samples" / "nested" / "page.html").read_text() == "<html></html>"
    assert (dst / "proj" / ".venv" / "bin" / "python").exists()


def test_copy_fixture_drops_gitkeep_but_keeps_its_directory(tmp_path):
    """A `.gitkeep` says "git, retain this empty directory". Once staged the
    directory exists, so the marker has done its job — the agent should see
    the empty directory the fixture author meant, not the bookkeeping."""
    src = tmp_path / "fixture"
    dst = tmp_path / "cwd"

    (src / "proj" / "output" / "logs").mkdir(parents=True)
    (src / "proj" / "output" / "logs" / ".gitkeep").write_text("")
    (src / "proj" / "spider.py").write_text("# spider")

    _copy_fixture(src, dst)

    assert (dst / "proj" / "output" / "logs").is_dir()
    assert not (dst / "proj" / "output" / "logs" / ".gitkeep").exists()
    assert list((dst / "proj" / "output" / "logs").iterdir()) == []
    assert (dst / "proj" / "spider.py").read_text() == "# spider"
