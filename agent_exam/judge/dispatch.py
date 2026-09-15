from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..providers import get_provider
from ..providers.base import Provider
from ..schemas import TextBlock

if TYPE_CHECKING:
    from ..config import Config

CallFn = Callable[[str], str]
"""Abstract "call the judge with this prompt, get its text response" function.

Parameterized here so tests can substitute a stub without building a Provider.
"""


@dataclass
class JudgeCall:
    """Everything the judge needs to make one LLM call.

    Constructed by :func:`build_judge_call` once per run and passed into
    the scoring context.

    Two timeouts: ``timeout_seconds`` for the plain ``judge`` assertion
    (one-shot LLM call), ``agent_timeout_seconds`` for ``judge_agent``
    (multi-turn tool loop, needs more headroom).
    """

    provider: Provider
    judge_model: str
    provider_options: dict = field(default_factory=dict)
    timeout_seconds: int = 60
    agent_timeout_seconds: int = 300


def build_judge_call(cfg: Config, harness: Provider) -> JudgeCall:
    """Resolve the judge provider and model from *cfg*.

    The judge runs on ``judge.provider`` when configured, otherwise on
    *harness*, the provider under evaluation. The model is ``judge.model``,
    falling back to that provider's ``judge_model`` and then to its
    ``default_model``; providers that accept an omitted model may receive an
    empty string and use their own default.
    """
    if cfg.judge.provider and cfg.judge.provider != harness.name:
        provider = get_provider(cfg.judge.provider)
    else:
        provider = harness
    provider_cfg = cfg.provider(provider.name)
    judge_model = (
        cfg.judge.model or provider_cfg.judge_model or provider_cfg.default_model or ""
    )
    return JudgeCall(
        provider=provider,
        judge_model=provider_cfg.resolve_model(judge_model),
        provider_options={"extra_args": list(provider_cfg.extra_args)},
        timeout_seconds=cfg.judge.timeout_seconds,
        agent_timeout_seconds=cfg.judge.agent_timeout_seconds,
    )


def _last_assistant_text(turns) -> str:
    for turn in reversed(turns):
        if turn.role != "assistant":
            continue
        parts = [b.text for b in turn.content if isinstance(b, TextBlock)]
        return "\n".join(p for p in parts if p is not None)
    return ""


def call_judge(jc: JudgeCall, prompt: str) -> str:
    """Invoke the judge, return the last assistant turn's text."""
    with tempfile.TemporaryDirectory(prefix="agent-exam-judge-") as tmp:
        run_result = jc.provider.invoke(
            prompt=prompt,
            model=jc.judge_model,
            cwd=Path(tmp),
            provider_options=jc.provider_options,
            stop_on_first_trigger=False,
            timeout_seconds=jc.timeout_seconds,
        )
    return _last_assistant_text(run_result.trajectory)


def call_judge_agent(jc: JudgeCall, prompt: str, attempt_cwd: Path) -> str:
    """Invoke an agentic judge that can read the attempt's archived cwd.

    Differs from :func:`call_judge` in three ways:

    1. The judge runs against a fresh copy of ``attempt_cwd`` (the
       archived cwd from the original run) rather than an empty tempdir.
       Copying isolates the archive from any side-effect files the
       harness writes — keeps ``cwd_hash`` stable across rescores.
    2. The provider's :meth:`judge_agent_options` are merged on top of
       any pre-set ``provider_options`` (e.g. ``extra_args`` from
       config), so each provider can use its native permission mechanism
       for read-only cwd inspection.
    3. Uses :attr:`JudgeCall.agent_timeout_seconds` — a separate budget
       from the plain-judge timeout because multi-turn tool use
       routinely exceeds the latter.
    """
    options = dict(jc.provider_options)
    options.update(jc.provider.judge_agent_options())
    with tempfile.TemporaryDirectory(prefix="agent-exam-judge-agent-") as tmp:
        judge_cwd = Path(tmp) / "cwd"
        if attempt_cwd.is_dir():
            shutil.copytree(attempt_cwd, judge_cwd, symlinks=False)
        else:
            judge_cwd.mkdir()
        run_result = jc.provider.invoke(
            prompt=prompt,
            model=jc.judge_model,
            cwd=judge_cwd,
            provider_options=options,
            stop_on_first_trigger=False,
            timeout_seconds=jc.agent_timeout_seconds,
        )
    return _last_assistant_text(run_result.trajectory)
