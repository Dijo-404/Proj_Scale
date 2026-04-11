# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Scenario configuration loading and validation tests."""

import pytest

from tasks import ScenarioConfigError, clear_task_cache, validate_scenario_config


@pytest.fixture(autouse=True)
def reset_task_cache():
    clear_task_cache()
    yield
    clear_task_cache()


def test_validate_scenario_config_missing_file(monkeypatch, tmp_path):
    missing = tmp_path / "missing_scenario_config.json"
    monkeypatch.setenv("PROJ_SCALE_SCENARIO_CONFIG", str(missing))

    with pytest.raises(ScenarioConfigError, match="scenario config not found"):
        validate_scenario_config()


def test_validate_scenario_config_rejects_invalid_json(monkeypatch, tmp_path):
    config_file = tmp_path / "scenario_config.json"
    config_file.write_text("{ invalid json", encoding="utf-8")
    monkeypatch.setenv("PROJ_SCALE_SCENARIO_CONFIG", str(config_file))

    with pytest.raises(ScenarioConfigError, match="not valid JSON"):
        validate_scenario_config()


def test_validate_scenario_config_rejects_unknown_task_order(monkeypatch, tmp_path):
    config_file = tmp_path / "scenario_config.json"
    config_file.write_text(
        """
{
  "task_order": ["unknown_task"],
  "tasks": [
    {
      "name": "easy_access_recovery",
      "difficulty": "easy",
      "description": "desc",
      "max_steps": 8,
      "tickets": [
        {
          "ticket_id": "ACC-1001",
          "subject": "login issue",
          "customer_tier": "business",
          "sla_hours": 8
        }
      ],
      "goals": {
        "ACC-1001": {
          "priority": "high",
          "category": "access",
          "team": "tier1",
          "status": "resolved",
          "reply_rule": {
            "required_keywords": ["verify"],
            "min_length": 20
          }
        }
      }
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROJ_SCALE_SCENARIO_CONFIG", str(config_file))

    with pytest.raises(ScenarioConfigError, match="task_order references unknown tasks"):
        validate_scenario_config()


def test_validate_scenario_config_returns_diagnostics_for_valid_file(monkeypatch, tmp_path):
    config_file = tmp_path / "scenario_config.json"
    config_file.write_text(
        """
{
  "task_order": ["easy_access_recovery"],
  "tasks": [
    {
      "name": "easy_access_recovery",
      "difficulty": "easy",
      "description": "desc",
      "max_steps": 8,
      "tickets": [
        {
          "ticket_id": "ACC-1001",
          "subject": "login issue",
          "customer_tier": "business",
          "sla_hours": 8
        }
      ],
      "goals": {
        "ACC-1001": {
          "priority": "high",
          "category": "access",
          "team": "tier1",
          "status": "resolved",
          "reply_rule": {
            "required_keywords": ["verify"],
            "min_length": 20
          }
        }
      }
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROJ_SCALE_SCENARIO_CONFIG", str(config_file))

    details = validate_scenario_config()

    assert details["config_path"] == str(config_file)
    assert details["task_count"] == 1
    assert details["task_order_count"] == 1
