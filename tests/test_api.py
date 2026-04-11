# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""API endpoint tests for the Proj_Scale FastAPI app."""

from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["name"] == "Proj_Scale"
    assert payload["message"] == "Proj_Scale OpenEnv API is running"


def test_tasks_endpoint_lists_all_benchmark_tasks():
    response = client.get("/tasks")

    assert response.status_code == 200
    payload = response.json()
    names = {task["name"] for task in payload["tasks"]}

    assert {
        "easy_access_recovery",
        "medium_billing_dispute",
        "hard_incident_swarm",
    }.issubset(names)
    assert all(task["has_grader"] is True for task in payload["tasks"])


def test_task_detail_endpoint_does_not_leak_goal_answers():
    response = client.get("/tasks/easy_access_recovery")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "easy_access_recovery"
    assert payload["has_grader"] is True
    assert payload["ticket_count"] == 1
    assert "goals" not in payload


def test_unknown_task_detail_returns_not_found():
    response = client.get("/tasks/does_not_exist")

    assert response.status_code == 404
