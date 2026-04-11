# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Deterministic grading logic for Proj_Scale tasks."""

from __future__ import annotations

import re
from typing import Dict, List

from tasks import TaskSpec, get_task


STRICT_SCORE_EPSILON = 1e-4


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _safe_text(value: object) -> str:
    return str(value) if value is not None else ""


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _has_keyword_phrase(normalized_reply: str, keyword: str) -> bool:
    normalized_keyword = _normalize(keyword)
    if not normalized_keyword:
        return False

    pattern = r"\b" + r"\s+".join(
        re.escape(token) for token in normalized_keyword.split()
    ) + r"\b"
    return re.search(pattern, normalized_reply) is not None


def _keyword_coverage(reply: str, keywords: tuple[str, ...]) -> float:
    if not keywords:
        return 1.0

    norm_reply = _normalize(reply)
    normalized_keywords = [
        _normalize(keyword)
        for keyword in keywords
        if _normalize(keyword)
    ]
    unique_keywords = list(dict.fromkeys(normalized_keywords))
    if not unique_keywords:
        return 1.0

    hits = sum(
        1 for keyword in unique_keywords if _has_keyword_phrase(norm_reply, keyword)
    )
    return hits / float(len(unique_keywords))


def _structure_score(reply: str) -> float:
    normalized_reply = reply.strip()
    if not normalized_reply:
        return 0.0

    sentence_chunks = [
        chunk.strip() for chunk in re.split(r"[.!?]+", normalized_reply) if chunk.strip()
    ]
    if len(sentence_chunks) >= 2:
        return 1.0

    token_count = len(_tokenize(normalized_reply))
    return 0.7 if token_count >= 20 else 0.3


def _anti_stuffing_score(reply: str, keywords: tuple[str, ...]) -> float:
    tokens = _tokenize(reply)
    if not tokens:
        return 0.0

    unique_ratio = len(set(tokens)) / float(len(tokens))
    # Scores below 0.25 often indicate low-information repetition.
    diversity_score = max(0.0, min(1.0, (unique_ratio - 0.25) / 0.5))

    keyword_tokens = {
        token
        for keyword in keywords
        for token in _tokenize(keyword)
    }
    if not keyword_tokens:
        density_score = 1.0
    else:
        keyword_hits = sum(1 for token in tokens if token in keyword_tokens)
        keyword_density = keyword_hits / float(len(tokens))
        if keyword_density <= 0.35:
            density_score = 1.0
        elif keyword_density >= 0.75:
            density_score = 0.0
        else:
            density_score = 1.0 - ((keyword_density - 0.35) / 0.40)

    return max(0.0, min(1.0, (0.6 * diversity_score) + (0.4 * density_score)))


def _communication_score(reply: str, keywords: tuple[str, ...], min_length: int) -> float:
    coverage = _keyword_coverage(reply, keywords)
    length_score = min(len(reply) / float(max(min_length, 1)), 1.0)
    structure = _structure_score(reply)
    anti_stuffing = _anti_stuffing_score(reply, keywords)
    combined = (
        (0.5 * coverage)
        + (0.15 * length_score)
        + (0.15 * structure)
        + (0.2 * anti_stuffing)
    )
    return max(0.0, min(1.0, combined))


def _extract_history_value(entry: object, aliases: tuple[str, ...]) -> str:
    for key in aliases:
        value = None
        if isinstance(entry, dict):
            value = entry.get(key)
        elif hasattr(entry, key):
            value = getattr(entry, key)

        text = _safe_text(value).strip()
        if text:
            return text
    return ""


def _strict_unit_interval(value: float) -> float:
    # OpenEnv validation expects scores strictly inside (0, 1),
    # so a perfect run is represented as 1 - epsilon (0.9999), not 1.0.
    return max(STRICT_SCORE_EPSILON, min(1.0 - STRICT_SCORE_EPSILON, value))


def _grade_process(
    task: TaskSpec, tickets: Dict[str, Dict], action_history: List[object]
) -> float:
    checks: List[float] = []

    if task.process_rule.first_action_ticket:
        first_ticket_action = None
        for entry in action_history:
            command = _extract_history_value(entry, ("command", "cmd", "action"))
            ticket_id = _extract_history_value(
                entry,
                ("ticket_id", "ticketId", "tid", "ticket"),
            )
            if command.lower() != "submit" and ticket_id:
                first_ticket_action = ticket_id
                break

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
    task: TaskSpec, tickets: Dict[str, Dict], action_history: List[object]
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
        communication_total += _communication_score(
            reply,
            goal.reply_rule.required_keywords,
            goal.reply_rule.min_length,
        )

    routing_score = max(0.0, min(1.0, routing_total / ticket_count))
    communication_score = max(0.0, min(1.0, communication_total / ticket_count))
    process_score = max(0.0, min(1.0, _grade_process(task, tickets, action_history)))

    raw_total = (0.5 * routing_score) + (0.3 * communication_score) + (0.2 * process_score)

    total = _strict_unit_interval(raw_total)

    return {
        "routing": round(_strict_unit_interval(routing_score), 4),
        "communication": round(_strict_unit_interval(communication_score), 4),
        "process": round(_strict_unit_interval(process_score), 4),
        "raw_total": round(raw_total, 6),
        "total": round(total, 6),
    }


def grade_easy_access_recovery(
    tickets: Dict[str, Dict], action_history: List[object]
) -> Dict[str, float]:
    return _grade_task(get_task("easy_access_recovery"), tickets, action_history)


def grade_medium_billing_dispute(
    tickets: Dict[str, Dict], action_history: List[object]
) -> Dict[str, float]:
    return _grade_task(get_task("medium_billing_dispute"), tickets, action_history)


def grade_hard_incident_swarm(
    tickets: Dict[str, Dict], action_history: List[object]
) -> Dict[str, float]:
    return _grade_task(get_task("hard_incident_swarm"), tickets, action_history)


TASK_GRADERS = {
    "easy_access_recovery": grade_easy_access_recovery,
    "medium_billing_dispute": grade_medium_billing_dispute,
    "hard_incident_swarm": grade_hard_incident_swarm,
}


def grade_for_task(
    task_name: str, tickets: Dict[str, Dict], action_history: List[object]
) -> Dict[str, float]:
    grader = TASK_GRADERS.get(task_name)
    if grader is None:
        raise KeyError(f"No grader registered for task: {task_name}")
    return grader(tickets, action_history)
