# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Deterministic grader unit tests."""

import pytest

from graders import (
    STRICT_SCORE_EPSILON,
    grade_easy_access_recovery,
    grade_for_task,
)


def test_easy_grader_returns_perfect_score_for_perfect_ticket():
    tickets = {
        "ACC-1001": {
            "priority": "high",
            "category": "access",
            "team": "tier1",
            "status": "resolved",
            "last_reply": (
                "Please verify your identity and complete the MFA reset flow. "
                "Access should be restored within 15 minutes after reset "
                "completion."
            ),
        }
    }
    action_history = [{"command": "set_priority", "ticket_id": "ACC-1001"}]

    result = grade_easy_access_recovery(tickets, action_history)

    assert 1.0 - STRICT_SCORE_EPSILON <= result["routing"] < 1.0
    assert 1.0 - STRICT_SCORE_EPSILON <= result["communication"] < 1.0
    assert 1.0 - STRICT_SCORE_EPSILON <= result["process"] < 1.0
    assert result["raw_total"] == pytest.approx(1.0)
    assert 1.0 - STRICT_SCORE_EPSILON <= result["total"] < 1.0


def test_easy_grader_clamps_zero_to_strict_open_interval():
    tickets = {
        "ACC-1001": {
            "priority": "low",
            "category": "billing",
            "team": "product",
            "status": "new",
            "last_reply": "",
        }
    }

    result = grade_easy_access_recovery(tickets, action_history=[])

    assert 0.0 < result["routing"] <= STRICT_SCORE_EPSILON
    assert 0.0 < result["communication"] <= STRICT_SCORE_EPSILON
    assert 0.0 < result["process"] <= STRICT_SCORE_EPSILON
    assert result["raw_total"] == pytest.approx(0.0)
    assert 0.0 < result["total"] <= STRICT_SCORE_EPSILON


def test_grade_for_task_raises_for_unknown_task():
    with pytest.raises(KeyError):
        grade_for_task("unknown_task", tickets={}, action_history=[])
