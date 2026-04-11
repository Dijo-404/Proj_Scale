# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Inference strategy regression tests."""

from types import SimpleNamespace

from inference_config import InferenceSettings
from inference_strategies import _classify_category, choose_action
from models import SupportOpsAction


def _settings(force_heuristic: bool = False) -> InferenceSettings:
    return InferenceSettings(
        api_base_url="https://example.com/v1",
        model_name="demo-model",
        reasoning_model="demo-model",
        api_key="token",
        env_base_url=None,
        local_image_name="proj_scale-env:test",
        benchmark="Proj_Scale",
        success_score_threshold=0.75,
        max_steps=20,
        llm_max_retries=1,
        force_heuristic=force_heuristic,
        task_name="easy_access_recovery",
        task_names=("easy_access_recovery",),
    )


def _ticket(**overrides):
    base = {
        "ticket_id": "ACC-1001",
        "priority": "high",
        "category": "access",
        "team": "tier1",
        "status": "resolved",
        "last_reply": "Please verify identity and complete MFA reset; access should return in 15 minutes.",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_choose_action_triggers_recovery_when_plan_submits_with_missing_targets(monkeypatch):
    fallback_action = SupportOpsAction(command="set_status", ticket_id="ACC-1001", value="resolved")

    monkeypatch.setattr(
        "inference_strategies.model_action_per_step",
        lambda *args, **kwargs: fallback_action,
    )

    observation = SimpleNamespace(tickets=[_ticket()])

    action = choose_action(
        observation=observation,
        client=object(),
        settings=_settings(force_heuristic=False),
        targets={},
        replies={},
        order=[],
        action_history=[],
    )

    assert action.command == "set_status"
    assert action.ticket_id == "ACC-1001"


def test_choose_action_keeps_submit_when_targets_are_complete(monkeypatch):
    def _unexpected(*_args, **_kwargs):
        raise AssertionError("model_action_per_step should not run for complete plans")

    monkeypatch.setattr("inference_strategies.model_action_per_step", _unexpected)

    observation = SimpleNamespace(tickets=[_ticket()])
    targets = {
        "ACC-1001": {
            "priority": "high",
            "category": "access",
            "team": "tier1",
            "status": "resolved",
        }
    }

    action = choose_action(
        observation=observation,
        client=object(),
        settings=_settings(force_heuristic=False),
        targets=targets,
        replies={"ACC-1001": observation.tickets[0].last_reply},
        order=["ACC-1001"],
        action_history=[],
    )

    assert action.command == "submit"


def test_classify_category_avoids_false_outage_for_plain_500_request():
    category, team = _classify_category("Account tier 500 upgrade request")

    assert category == "feature_request"
    assert team == "product"


def test_classify_category_marks_api_500_incident_as_outage():
    category, team = _classify_category("EU API 500 errors impacting checkout flow")

    assert category == "outage"
    assert team == "sre"


def test_classify_category_keeps_non_incident_api_request_as_feature():
    category, team = _classify_category("API documentation request")

    assert category == "feature_request"
    assert team == "product"
