# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Inference environment and output-format compliance tests."""

import pytest

from inference import _read_required_env
from inference_runner import log_end, log_start, log_step


def test_read_required_env_requires_hf_token(monkeypatch):
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(ValueError, match="HF_TOKEN"):
        _read_required_env()


def test_read_required_env_uses_defaults(monkeypatch):
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.setenv("HF_TOKEN", "token")

    api_base_url, model_name, hf_token = _read_required_env()

    assert api_base_url == "https://router.huggingface.co/v1"
    assert model_name == "Qwen/Qwen2.5-72B-Instruct"
    assert hf_token == "token"


def test_inference_logs_match_required_format(capsys):
    log_start(task="easy_access_recovery", env="Proj_Scale", model="demo-model")
    log_step(step=1, action="submit", reward=0.0, done=False, error=None)
    log_step(step=2, action="submit", reward=1.0, done=True, error="boom")
    log_end(success=True, steps=2, rewards=[0.0, 1.0])

    out_lines = capsys.readouterr().out.strip().splitlines()

    assert out_lines[0] == "[START] task=easy_access_recovery env=Proj_Scale model=demo-model"
    assert out_lines[1] == "[STEP] step=1 action=submit reward=0.00 done=false error=null"
    assert out_lines[2] == "[STEP] step=2 action=submit reward=1.00 done=true error=boom"
    assert out_lines[3] == "[END] success=true steps=2 rewards=0.00,1.00"
