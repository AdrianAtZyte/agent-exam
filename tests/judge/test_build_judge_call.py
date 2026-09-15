from __future__ import annotations

from pathlib import Path

from agent_exam.config import Config
from agent_exam.judge import build_judge_call
from agent_exam.providers import get_provider


def _config(**raw) -> Config:
    return Config.model_validate(
        {"project_root": Path("/p"), "evals_dir": Path("/p/evals"), **raw}
    )


def test_judge_defaults_to_harness():
    cfg = _config(providers={"dummy": {"judge_model": "judge", "extra_args": ["-x"]}})
    jc = build_judge_call(cfg, get_provider("dummy"))
    assert jc.provider.name == "dummy"
    assert jc.judge_model == "judge"
    assert jc.provider_options == {"extra_args": ["-x"]}


def test_judge_provider_uses_its_own_block():
    cfg = _config(
        judge={"provider": "dummy"},
        providers={
            "claude_code": {"judge_model": "haiku"},
            "dummy": {"default_model": "d", "model_aliases": {"d": "dummy-1"}},
        },
    )
    jc = build_judge_call(cfg, get_provider("claude_code"))
    assert jc.provider.name == "dummy"
    assert jc.judge_model == "dummy-1"
