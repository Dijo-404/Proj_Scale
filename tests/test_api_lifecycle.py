from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_http_reset_step_state_lifecycle():
    reset = client.post("/reset", json={"task_name": "easy_access_recovery"})
    assert reset.status_code == 200

    payload = reset.json()
    assert payload["done"] is False
    assert payload["observation"]["task_name"] == "easy_access_recovery"

    step = client.post(
        "/step",
        json={
            "action": {
                "command": "set_priority",
                "ticket_id": "ACC-1001",
                "value": "high",
            }
        },
    )
    assert step.status_code == 200
    step_payload = step.json()
    assert step_payload["done"] is False

    state = client.get("/state")
    assert state.status_code == 200
    state_payload = state.json()
    assert "episode_id" in state_payload
    assert "step_count" in state_payload


def test_http_submit_completes_episode():
    client.post("/reset", json={"task_name": "easy_access_recovery"})

    submit = client.post("/step", json={"action": {"command": "submit"}})
    assert submit.status_code == 200

    payload = submit.json()
    assert payload["done"] is True
    assert payload["observation"]["last_action_summary"].startswith("Submitted")
