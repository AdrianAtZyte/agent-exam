"""Tests for the staged-skill description check."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agent_exam.providers.claude_code.session_checks import (
    skill_descriptions_in_session,
)

if TYPE_CHECKING:
    from pathlib import Path


def _transcript(tmp_path: Path, listing: str) -> Path:
    p = tmp_path / "transcript.jsonl"
    p.write_text(
        json.dumps(
            {
                "type": "attachment",
                "attachment": {"type": "skill_listing", "content": listing},
            }
        )
        + "\n"
    )
    return p


def test_ok_when_every_staged_skill_is_described(tmp_path):
    transcript = _transcript(
        tmp_path,
        "- alpha: Does alpha.\n- beta: Does beta.\n- builtin: Does builtin things.",
    )
    result = skill_descriptions_in_session(transcript, ["alpha", "beta"])
    assert result.status == "OK"


def test_warns_about_the_skills_listed_without_one(tmp_path):
    transcript = _transcript(
        tmp_path, "- alpha: Does alpha.\n- beta\n- builtin: Does builtin."
    )
    result = skill_descriptions_in_session(transcript, ["alpha", "beta"])
    assert result.status == "WARN"
    assert "1 skill (beta)" in result.hint


def test_unstaged_skills_are_reported_separately(tmp_path):
    result = skill_descriptions_in_session(
        _transcript(tmp_path, "- builtin: Does builtin things."), ["alpha"]
    )
    assert result.status == "WARN"
    assert "did not list alpha at all" in result.hint


def test_skipped_without_a_listing(tmp_path):
    assert (
        skill_descriptions_in_session(_transcript(tmp_path, ""), ["alpha"]).status
        == "WARN"
    )
    assert skill_descriptions_in_session(None, ["alpha"]).status == "WARN"
