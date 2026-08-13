from __future__ import annotations

import io
import sys

from agent_exam.cli import main


def test_main_forces_utf8_output(monkeypatch):
    """A Windows console hands Python a cp1252 stdout, which every arrow in the
    reports would fail to encode.
    """
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "argv", ["agent-exam"])

    main()

    stream.write("→")
    assert stream.encoding == "utf-8"
