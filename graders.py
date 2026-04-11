# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Deterministic grading logic for Proj_Scale tasks."""

from __future__ import annotations

from typing import Dict, List

from tasks import TASK_LIBRARY, TaskSpec


STRICT_SCORE_EPSILON = 1e-4


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _safe_text(value: object) -> str:
    return str(value) if value is not None else ""


def _keyword_coverage(reply: str, keywords: tuple[str, ...]) -> float:
    if not keywords:
        return 1.0
    norm_reply = _normalize(reply)
    hits = sum(1 for keyword in keywords if _normalize(keyword) in norm_reply)
    return hits / float(len(keywords))


def _strict_unit_interval(value: float) -> float:
    # OpenEnv task validation expects totals strictly inside (0, 1).
    return max(STRICT_SCORE_EPSILON, min(1.0 - STRICT_SCORE_EPSILON, value))


def _grade_process(
    task: TaskSpec, tickets: Dict[str, Dict], action_history: List[Dict]
) -> float:
    checks: List[float] = []

    if task.process_rule.first_action_ticket:
        first_ticket_action = next(
            (
                entry.get("ticket_id")
                for entry in action_history
                if entry.get("command") != "submit" and entry.get("ticket_id")
            ),
            None,
        )
        checks.append(
            1.0 if first_ticket_action == task.process_rule.first_action_ticket else 0.0
        )

    for ticket_id in task.process_rule.must_escalate:
        status = _safe_text(tickets.get(ticket_id, {}).get("status"))
        checks.append(1.0 if status == "escalated" else 0.0)

    for ticket_id in task.process_rule.must_resolve:
        status = _safe_text(tickets.get(ticket_id, {}).get("status"))
        checks.append(1.0 if status == "resolved" else 0.0)

    if not checks:
        return 1.0
    return sum(checks) / len(checks)


def _grade_task(
    task: TaskSpec, tickets: Dict[str, Dict], action_history: List[Dict]
) -> Dict[str, float]:
    routing_total = 0.0
    communication_total = 0.0
    ticket_count = max(len(task.goals), 1)

    for ticket_id, goal in task.goals.items():
        ticket = tickets.get(ticket_id, {})

        routing_checks = [
            _safe_text(ticket.get("priority")) == goal.priority,
            _safe_text(ticket.get("category")) == goal.category,
            _safe_text(ticket.get("team")) == goal.team,
            _safe_text(ticket.get("status")) == goal.status,
        ]
        routing_total += sum(1.0 for ok in routing_checks if ok) / len(routing_checks)

        reply = _safe_text(ticket.get("last_reply"))
        coverage = _keyword_coverage(reply, goal.reply_rule.required_keywords)
        length_score = min(len(reply) / float(goal.reply_rule.min_length), 1.0)
        communication_total += (0.8 * coverage) + (0.2 * length_score)

    routing_score = routing_total / ticket_count
    communication_score = communication_total / ticket_count
    process_score = _grade_process(task, tickets, action_history)

    raw_total = (0.5 * routing_score) + (0.3 * communication_score) + (0.2 * process_score)

    # Clamp all scores to open interval (0, 1) — the evaluator rejects
    # exact 0.0 and 1.0 values.
    routing_score = _strict_unit_interval(routing_score)
    communication_score = _strict_unit_interval(communication_score)
    process_score = _strict_unit_interval(process_score)
    total = _strict_unit_interval(raw_total)

    return {
        "routing": round(routing_score, 4),
        "communication": round(communication_score, 4),
        "process": round(process_score, 4),
        "raw_total": round(raw_total, 6),
        "total": round(total, 6),
    }


def grade_easy_access_recovery(
    tickets: Dict[str, Dict], action_history: List[Dict]
) -> Dict[str, float]:
    return _grade_task(TASK_LIBRARY["easy_access_recovery"], tickets, action_history)


def grade_medium_billing_dispute(
    tickets: Dict[str, Dict], action_history: List[Dict]
) -> Dict[str, float]:
    return _grade_task(TASK_LIBRARY["medium_billing_dispute"], tickets, action_history)


def grade_hard_incident_swarm(
    tickets: Dict[str, Dict], action_history: List[Dict]
) -> Dict[str, float]:
    return _grade_task(TASK_LIBRARY["hard_incident_swarm"], tickets, action_history)


TASK_GRADERS = {
    "easy_access_recovery": grade_easy_access_recovery,
    "medium_billing_dispute": grade_medium_billing_dispute,
    "hard_incident_swarm": grade_hard_incident_swarm,
}


def grade_for_task(
    task_name: str, tickets: Dict[str, Dict], action_history: List[Dict]
) -> Dict[str, float]:
    grader = TASK_GRADERS.get(task_name)
    if grader is None:
        raise KeyError(f"No grader registered for task: {task_name}")
    return grader(tickets, action_history)
