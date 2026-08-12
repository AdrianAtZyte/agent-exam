from __future__ import annotations

import re
from typing import Literal

Verdict = Literal["YES", "NO", "UNCLEAR"]

_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(YES|NO|UNCLEAR)\b", re.IGNORECASE)


def parse_verdict(response: str) -> tuple[Verdict, str]:
    """Parse a judge response into (verdict, reasoning).

    - Looks for the *last* line containing a `VERDICT:` marker. Judges are
      asked to put the verdict on the final line, but they sometimes state it
      mid-response and then keep writing (a trailing summary paragraph); taking
      the last match honours the intended verdict instead of falling back to
      UNCLEAR because a prose line came after it.
    - Everything before that line is reasoning (whitespace-trimmed).
    - If no `VERDICT:` line found, returns UNCLEAR with the raw response as
      the reasoning (for debugging).
    """
    if not response or not response.strip():
        return "UNCLEAR", response or ""

    lines = response.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        match = _VERDICT_RE.search(lines[i])
        if match:
            verdict: Verdict = match.group(1).upper()  # type: ignore[assignment]
            reasoning = "\n".join(lines[:i]).strip()
            return verdict, reasoning

    return "UNCLEAR", response.strip()
