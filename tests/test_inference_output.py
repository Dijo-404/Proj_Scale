# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Inference environment and output-format compliance tests."""

import asyncio

import pytest

from inference_config import InferenceSettings
from inference import _read_required_env
from inference_runner import log_end, log_start, log_step, run_inference


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
    log_end(success=True, steps=2, score=0.5, rewards=[0.0, 1.0])

    out_lines = capsys.readouterr().out.strip().splitlines()

    assert out_lines[0] == "[START] task=easy_access_recovery env=Proj_Scale model=demo-model"
    assert out_lines[1] == "[STEP] step=1 action=submit reward=0.00 done=false error=null"
    assert out_lines[2] == "[STEP] step=2 action=submit reward=1.00 done=true error=boom"
    assert out_lines[3] == "[END] success=true steps=2 score=0.50 rewards=0.00,1.00"


def test_log_end_clamps_score_to_open_interval(capsys):
    log_end(success=False, steps=0, score=0.0, rewards=[])
    log_end(success=True, steps=1, score=1.0, rewards=[1.0])

    out_lines = capsys.readouterr().out.strip().splitlines()

    assert out_lines[0] == "[END] success=false steps=0 score=0.01 rewards="
    assert out_lines[1] == "[END] success=true steps=1 score=0.99 rewards=1.00"


def test_run_inference_runs_all_tasks_when_requested(monkeypatch, capsys):
    class _DummyEnv:
        async def close(self):
            return None

    async def _fake_from_docker_image(_image_name):
        return _DummyEnv()

    called_tasks = []

    async def _fake_run_task(_env, settings, _llm_client):
        called_tasks.append(settings.task_name)
        return True, 1, 0.5, [0.5]

    monkeypatch.setattr("inference_runner.SupportOpsEnv.from_docker_image", _fake_from_docker_image)
    monkeypatch.setattr("inference_runner.run_task", _fake_run_task)

    settings = InferenceSettings.from_env(
        api_base_url="https://example.com/v1",
        model_name="demo-model",
        hf_token="token",
    ).with_overrides(force_heuristic=True)

    exit_code = asyncio.run(run_inference(settings, run_all_tasks=True))

    assert exit_code == 0
    assert called_tasks == list(settings.task_names)

    out_lines = capsys.readouterr().out.strip().splitlines()
    start_lines = [line for line in out_lines if line.startswith("[START]")]
    end_lines = [line for line in out_lines if line.startswith("[END]")]

    assert len(start_lines) == len(settings.task_names)
    assert len(end_lines) == len(settings.task_names)
