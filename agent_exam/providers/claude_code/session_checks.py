from __future__ import annotations

from typing import TYPE_CHECKING

from ...schemas import CheckResult
from .transcripts import skill_listing

if TYPE_CHECKING:
    from pathlib import Path


def skill_descriptions_in_session(
    transcript_path: Path | None, skill_names: list[str]
) -> CheckResult:
    """Check that every staged skill reached the model with its description.

    Claude Code lists the skills it can see with their descriptions and caps
    that listing at a fixed size. Skills past the cap are listed by name alone,
    which leaves the model nothing to route on, so it never invokes them. It
    reports neither the cap nor what it left out — hence "most likely" in the
    warning — but the listing shows which descriptions arrived. Room is made by
    shortening any description, not those of the skills named: the cap is on
    their combined length, and it does not shed the longest ones first.

    Distinct from the doctor probes, which are about the developer's setup:
    what this finds is fixed by editing the skills under test, so the runner
    reports it.
    """
    name = "staged skills have descriptions"
    if transcript_path is None or not transcript_path.exists():
        return CheckResult(
            name=name, status="WARN", hint="transcript path missing — skipped"
        )
    if not skill_names:
        return CheckResult(name=name, status="OK", hint="no staged skills")
    listing = skill_listing(transcript_path)
    if not listing:
        return CheckResult(
            name=name, status="WARN", hint="no skill listing in session transcript"
        )

    entries = [line[2:] for line in listing.splitlines() if line.startswith("- ")]
    listed = {entry.split(":", 1)[0] for entry in entries}
    described = {entry.split(":", 1)[0] for entry in entries if ": " in entry}

    absent = [n for n in skill_names if n not in listed]
    if absent:
        # Nothing to do with description length: the session never loaded them.
        return CheckResult(
            name=name,
            status="WARN",
            hint=(
                f"Claude Code did not list {', '.join(absent)} at all — they "
                "were not staged, so nothing could invoke them"
            ),
        )

    dropped = [n for n in skill_names if n not in described]
    if not dropped:
        return CheckResult(
            name=name,
            status="OK",
            hint=f"all {len(skill_names)} staged skills listed with descriptions",
        )
    count = "1 skill" if len(dropped) == 1 else f"{len(dropped)} skills"
    return CheckResult(
        name=name,
        status="WARN",
        hint=(
            f"Claude Code ignored the descriptions of {count} "
            f"({', '.join(dropped)}), most likely due to the total length of "
            "all skill descriptions. Consider shortening skill descriptions."
        ),
    )
