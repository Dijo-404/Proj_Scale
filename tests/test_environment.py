import pytest

from models import SupportOpsAction
from server.support_ops_environment import SupportOpsEnvironment


def _ticket_by_id(observation, ticket_id):
    for ticket in observation.tickets:
        if ticket.ticket_id == ticket_id:
            return ticket
    raise AssertionError(f"Ticket {ticket_id} not found")


def test_reset_uses_requested_task_and_episode_id():
    env = SupportOpsEnvironment()

    observation = env.reset(task_name="easy_access_recovery", episode_id="ep-test-1")

    assert observation.task_name == "easy_access_recovery"
    assert observation.remaining_steps == 8
    assert len(observation.tickets) == 1
    assert env.state.active_task == "easy_access_recovery"
    assert env.state.episode_id == "ep-test-1"


def test_invalid_priority_action_sets_error_and_penalty():
    env = SupportOpsEnvironment()
    env.reset(task_name="easy_access_recovery")

    observation = env.step(
        SupportOpsAction(
            command="set_priority",
            ticket_id="ACC-1001",
            value="urgent",
        )
    )

    assert observation.done is False
    assert observation.last_action_error is not None
    assert "Invalid priority" in observation.last_action_error
    assert observation.reward_details.invalid_action_penalty == pytest.approx(0.05)


def test_reply_moves_new_ticket_to_in_progress():
    env = SupportOpsEnvironment()
    env.reset(task_name="easy_access_recovery")

    observation = env.step(
        SupportOpsAction(
            command="reply",
            ticket_id="ACC-1001",
            message=(
                "Please verify your identity, complete MFA reset steps, and "
                "confirm once access is restored."
            ),
        )
    )

    ticket = _ticket_by_id(observation, "ACC-1001")
    assert ticket.status == "in_progress"
    assert ticket.last_reply is not None


def test_submit_ends_episode():
    env = SupportOpsEnvironment()
    env.reset(task_name="easy_access_recovery")

    observation = env.step(SupportOpsAction(command="submit"))

    assert observation.done is True
    assert env.state.done is True
