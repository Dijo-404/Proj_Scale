# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Direct environment state-machine behavior tests."""

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


def test_full_medium_lifecycle_reaches_high_score():
    env = SupportOpsEnvironment()
    env.reset(task_name="medium_billing_dispute")

    # Critical billing dispute first.
    env.step(
        SupportOpsAction(
            command="set_priority",
            ticket_id="BILL-2044",
            value="critical",
        )
    )
    env.step(
        SupportOpsAction(
            command="set_category",
            ticket_id="BILL-2044",
            value="billing",
        )
    )
    env.step(
        SupportOpsAction(
            command="assign_team",
            ticket_id="BILL-2044",
            value="billing",
        )
    )
    env.step(
        SupportOpsAction(
            command="reply",
            ticket_id="BILL-2044",
            message=(
                "We apologize for the duplicate charge. Billing is reviewing your invoice and "
                "we will process the refund. You will receive confirmation within 48 hours."
            ),
        )
    )
    env.step(
        SupportOpsAction(
            command="set_status",
            ticket_id="BILL-2044",
            value="escalated",
        )
    )

    # Routine invoice request second.
    env.step(
        SupportOpsAction(
            command="set_priority",
            ticket_id="BILL-2058",
            value="medium",
        )
    )
    env.step(
        SupportOpsAction(
            command="set_category",
            ticket_id="BILL-2058",
            value="billing",
        )
    )
    env.step(
        SupportOpsAction(
            command="assign_team",
            ticket_id="BILL-2058",
            value="billing",
        )
    )
    env.step(
        SupportOpsAction(
            command="reply",
            ticket_id="BILL-2058",
            message=(
                "Your invoice PDF is attached in this thread. If your finance team needs a "
                "different format, let us know and we can provide it today."
            ),
        )
    )
    env.step(
        SupportOpsAction(
            command="set_status",
            ticket_id="BILL-2058",
            value="resolved",
        )
    )

    observation = env.step(SupportOpsAction(command="submit"))

    assert observation.done is True
    assert observation.score > 0.95
    assert observation.grader_breakdown["routing"] == pytest.approx(1.0)


def test_step_limit_enforced_without_submit():
    env = SupportOpsEnvironment()
    env.reset(task_name="easy_access_recovery")

    observation = None
    for _ in range(8):
        observation = env.step(
            SupportOpsAction(
                command="set_priority",
                ticket_id="ACC-1001",
                value="high",
            )
        )

    assert observation is not None
    assert observation.done is True
    assert env.state.done is True
    assert env.state.step_count == 8


def test_state_tracks_selected_ticket_and_step_count():
    env = SupportOpsEnvironment()
    env.reset(task_name="easy_access_recovery")

    env.step(
        SupportOpsAction(
            command="set_category",
            ticket_id="ACC-1001",
            value="access",
        )
    )

    assert env.state.step_count == 1
    assert env.state.selected_ticket == "ACC-1001"


def test_short_reply_is_rejected():
    env = SupportOpsEnvironment()
    env.reset(task_name="easy_access_recovery")

    observation = env.step(
        SupportOpsAction(
            command="reply",
            ticket_id="ACC-1001",
            message="too short",
        )
    )

    assert observation.last_action_error is not None
    assert "at least 20 characters" in observation.last_action_error


def test_action_after_done_returns_error():
    env = SupportOpsEnvironment()
    env.reset(task_name="easy_access_recovery")
    env.step(SupportOpsAction(command="submit"))

    observation = env.step(SupportOpsAction(command="submit"))

    assert observation.done is True
    assert observation.last_action_error is not None
    assert "Episode already completed" in observation.last_action_error
