"""_mirror_cwd archives an attempt's cwd, pruning only the skill-discovery
symlink dirs it stages (e.g. `.agents/skills/`) so archives stay small
without silently dropping unrelated fixture/agent content."""

from __future__ import annotations

from agent_exam.pool import _mirror_cwd


def test_mirror_cwd_prunes_only_skills_subdir_of_discovery_dirs(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    (src / ".agents" / "skills" / "probe-skill").mkdir(parents=True)
    (src / ".agents" / "skills" / "probe-skill" / "SKILL.md").write_text("# skill")
    (src / ".agents" / "notes.txt").write_text("agent deliverable")
    (src / "output.txt").write_text("agent output")

    _mirror_cwd(src, dst)

    assert not (dst / ".agents" / "skills").exists()
    assert (dst / ".agents" / "notes.txt").read_text() == "agent deliverable"
    assert (dst / "output.txt").read_text() == "agent output"


def test_mirror_cwd_ignores_nested_dirs_named_like_discovery_dirs(tmp_path):
    """Only the top-level `<discovery-dir>/skills` is pruned — a nested
    directory that happens to share a discovery-dir name is fixture
    content, not something the provider staged, and must be archived."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    (src / "some_project" / ".claude" / "skills").mkdir(parents=True)
    (src / "some_project" / ".claude" / "skills" / "marker.txt").write_text("keep me")

    _mirror_cwd(src, dst)

    assert (
        dst / "some_project" / ".claude" / "skills" / "marker.txt"
    ).read_text() == "keep me"


def test_mirror_cwd_archives_agent_created_build_artifacts(tmp_path):
    """Whatever the agent created stays in the archive — a `.venv` from its
    own `uv run` records what actually got installed, which is not always
    what a rebuild would produce, and that's evidence when reconstructing
    what went wrong."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    (src / "proj" / ".venv" / "lib").mkdir(parents=True)
    (src / "proj" / ".venv" / "lib" / "installed.so").write_bytes(b"\x00")
    (src / "proj" / "__pycache__").mkdir()
    (src / "proj" / "__pycache__" / "m.cpython-314.pyc").write_bytes(b"\x00")
    (src / "proj" / "items.jsonl").write_text('{"name": "x"}\n')

    _mirror_cwd(src, dst)

    assert (dst / "proj" / ".venv" / "lib" / "installed.so").exists()
    assert (dst / "proj" / "__pycache__" / "m.cpython-314.pyc").exists()
    assert (dst / "proj" / "items.jsonl").read_text() == '{"name": "x"}\n'


def test_mirror_cwd_survives_dangling_symlinks(tmp_path):
    """A timed-out attempt gets killed mid-write, so the cwd can hold a
    symlink whose target never appeared."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    (src / "half-built").mkdir(parents=True)
    (src / "half-built" / "lib64").symlink_to("lib")
    (src / "output.txt").write_text("agent output")

    _mirror_cwd(src, dst)

    assert (dst / "output.txt").read_text() == "agent output"
    assert not (dst / "half-built" / "lib64").exists()
