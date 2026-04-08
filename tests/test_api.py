from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["name"] == "Proj_Scale"


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


def test_unknown_task_detail_returns_not_found():
    response = client.get("/tasks/does_not_exist")

    assert response.status_code == 404
