# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Action planning and selection strategies for Proj_Scale inference."""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from inference_config import (
    COMMAND_ENUM_MAP,
    VALID_CATEGORIES,
    VALID_COMMANDS,
    VALID_PRIORITIES,
    VALID_STATUSES,
    VALID_TEAMS,
    InferenceSettings,
)
from inference_prompts import PLANNING_SYSTEM_PROMPT, TRIAGE_SYSTEM_PROMPT
from models import SupportOpsAction

_DEFAULT_REPLY = (
    "Thank you for contacting support. We are reviewing your request now and will "
    "share the next concrete step and timeline shortly in this thread."
)


def _priority_rank(priority: str) -> int:
    rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return rank.get(priority, 4)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _classify_category(subject: str) -> Tuple[str, str]:
    normalized = _normalize(subject)

    if any(token in normalized for token in ("login", "mfa", "password", "access", "locked out")):
        return "access", "tier1"
    if any(token in normalized for token in ("billing", "invoice", "refund", "charge", "payment")):
        return "billing", "billing"
    if any(token in normalized for token in ("500", "outage", "downtime", "api", "checkout")):
        return "outage", "sre"
    if any(token in normalized for token in ("oauth", "token", "security", "suspicious", "compromised")):
        return "security", "security"
    if any(token in normalized for token in ("feature", "request", "enhancement", "roadmap")):
        return "feature_request", "product"

    return "access", "tier1"


def _infer_priority(sla_hours: int, customer_tier: str, category: str) -> str:
    if sla_hours <= 1:
        return "critical"
    if sla_hours <= 2:
        if customer_tier == "enterprise" and category == "billing":
            return "critical"
        return "high"
    if sla_hours <= 8:
        return "high"
    if sla_hours <= 24:
        return "medium"

    if customer_tier == "enterprise" and category in {"outage", "security"}:
        return "high"
    return "low"


def _infer_status(category: str, priority: str, customer_tier: str, subject: str) -> str:
    normalized = _normalize(subject)
    if category in {"outage", "security"}:
        return "escalated"
    if (
        category == "billing"
        and customer_tier == "enterprise"
        and any(token in normalized for token in ("refund", "double", "charged twice"))
    ):
        return "escalated"
    if category == "feature_request":
        return "in_progress"
    if priority in {"critical", "high"} and category == "billing":
        return "escalated"
    return "resolved"


def _reply_for_ticket(category: str, subject: str) -> str:
    normalized = _normalize(subject)

    if category == "access":
        return (
            "Thanks for reporting this access issue. Please verify your identity in the secure portal, "
            "complete the MFA reset sequence, and confirm sign-in. We have queued the reset and expect "
            "normal access within 15 minutes after verification."
        )
    if category == "billing":
        if any(token in normalized for token in ("invoice", "pdf", "audit", "copy")):
            return (
                "Your invoice PDF is attached to this ticket. If your finance team needs a signed "
                "copy or additional invoice documents, reply here and we can provide them today."
            )
        return (
            "We apologize for the billing issue and are reviewing invoice history now. We will process "
            "any required refund and confirm correction in this thread. You can expect a billing update "
            "within 48 hours with the invoice details."
        )
    if category == "outage":
        return (
            "We have opened an incident and SRE is applying mitigation now. We will post progress on the "
            "status page and send your next update in 30 minutes. If impact changes, we will escalate "
            "communication frequency immediately."
        )
    if category == "security":
        return (
            "Security has escalated this case and started an investigation. Please revoke affected tokens, "
            "rotate credentials, and share any suspicious events you observed. We will continue updates "
            "as containment and analysis progress."
        )
    if category == "feature_request":
        return (
            "Thank you for the feature request. We added this item to roadmap tracking and linked it to "
            "the current product planning cycle. We will share updates here when prioritization for this "
            "feature request changes."
        )
    return _DEFAULT_REPLY


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, ValueError):
        pass

    for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            continue
        try:
            payload = json.loads(match.group(1))
            if isinstance(payload, dict):
                return payload
        except (json.JSONDecodeError, ValueError):
            continue

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, ValueError):
        return None

    return None


def _validate_action(raw: Dict[str, Any], observation: Any) -> Optional[SupportOpsAction]:
    command = str(raw.get("command", "")).strip().lower()
    if command not in VALID_COMMANDS:
        return None

    ticket_id = raw.get("ticket_id")
    value = raw.get("value")
    message = raw.get("message")

    ticket_ids = {ticket.ticket_id for ticket in observation.tickets}
    if command != "submit":
        if not ticket_id or ticket_id not in ticket_ids:
            return None

    enum_values = COMMAND_ENUM_MAP.get(command)
    if enum_values is not None:
        if value is None:
            return None
        value = str(value).strip().lower()
        if value not in enum_values:
            return None

    if command == "reply":
        if message is None or len(str(message).strip()) < 20:
            return None
        message = str(message).strip()

    return SupportOpsAction(
        command=command,
        ticket_id=ticket_id,
        value=value,
        message=message,
    )


def _llm_call(
    client: OpenAI,
    *,
    settings: InferenceSettings,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> Optional[str]:
    for attempt in range(settings.llm_max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
            )
            content = (completion.choices[0].message.content or "").strip()
            if content:
                return content
        except Exception as exc:
            print(
                f"[WARN] LLM call failed on attempt {attempt + 1}/{settings.llm_max_retries + 1}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < settings.llm_max_retries:
                time.sleep(min(2**attempt, 4))

    return None


def build_baseline_plan(observation: Any) -> Dict[str, Any]:
    """Build deterministic non-cheating targets from observable ticket content."""

    tickets_plan: Dict[str, Dict[str, str]] = {}
    ordering: List[Tuple[int, int, str]] = []

    for ticket in observation.tickets:
        category, team = _classify_category(ticket.subject)
        priority = _infer_priority(ticket.sla_hours, ticket.customer_tier, category)
        status = _infer_status(category, priority, ticket.customer_tier, ticket.subject)
        reply_text = _reply_for_ticket(category, ticket.subject)

        tickets_plan[ticket.ticket_id] = {
            "priority": priority,
            "category": category,
            "team": team,
            "status": status,
            "reply_text": reply_text,
        }

        ordering.append((_priority_rank(priority), ticket.sla_hours, ticket.ticket_id))

    ordering.sort()

    return {
        "ticket_order": [ticket_id for _, _, ticket_id in ordering],
        "tickets": tickets_plan,
    }


def plan_episode(
    client: Optional[OpenAI],
    settings: InferenceSettings,
    observation: Any,
) -> Optional[Dict[str, Any]]:
    """Generate an LLM triage plan from the initial observation."""

    if client is None:
        return None

    ticket_info = [
        {
            "ticket_id": ticket.ticket_id,
            "subject": ticket.subject,
            "customer_tier": ticket.customer_tier,
            "sla_hours": ticket.sla_hours,
            "current_status": ticket.status,
        }
        for ticket in observation.tickets
    ]

    user_prompt = json.dumps(
        {
            "task": observation.task_name,
            "difficulty": observation.difficulty,
            "description": observation.task_description,
            "max_steps": observation.remaining_steps,
            "tickets": ticket_info,
            "hints": observation.action_hints,
        },
        ensure_ascii=True,
    )

    content = _llm_call(
        client,
        settings=settings,
        model=settings.reasoning_model,
        system_prompt=PLANNING_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=1024,
    )
    if content is None:
        return None

    payload = _extract_json(content)
    if payload is None or not isinstance(payload.get("tickets"), dict):
        print(
            "[WARN] Unable to parse episode planning output; falling back.",
            file=sys.stderr,
            flush=True,
        )
        return None

    return payload


def merge_targets(
    observation: Any,
    baseline_plan: Dict[str, Any],
    llm_plan: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], List[str]]:
    """Merge baseline and LLM plans into normalized actionable targets."""

    targets: Dict[str, Dict[str, str]] = {}
    replies: Dict[str, str] = {}

    baseline_tickets = baseline_plan.get("tickets", {})
    baseline_order = list(baseline_plan.get("ticket_order", []))

    llm_tickets = llm_plan.get("tickets", {}) if llm_plan else {}
    llm_order = list(llm_plan.get("ticket_order", [])) if llm_plan else []

    for ticket in observation.tickets:
        ticket_id = ticket.ticket_id

        base_spec = baseline_tickets.get(ticket_id, {})
        llm_spec = llm_tickets.get(ticket_id, {})
        if not isinstance(llm_spec, dict):
            llm_spec = {}

        priority = str(llm_spec.get("priority", base_spec.get("priority", "medium"))).strip().lower()
        category = str(llm_spec.get("category", base_spec.get("category", "access"))).strip().lower()
        team = str(llm_spec.get("team", base_spec.get("team", "tier1"))).strip().lower()
        status = str(llm_spec.get("status", base_spec.get("status", "in_progress"))).strip().lower()
        reply_text = str(llm_spec.get("reply_text", base_spec.get("reply_text", _DEFAULT_REPLY))).strip()

        if priority not in VALID_PRIORITIES:
            priority = str(base_spec.get("priority", "medium"))
        if category not in VALID_CATEGORIES:
            category = str(base_spec.get("category", "access"))
        if team not in VALID_TEAMS:
            team = str(base_spec.get("team", "tier1"))
        if status not in VALID_STATUSES:
            status = str(base_spec.get("status", "in_progress"))
        if len(reply_text) < 20:
            reply_text = str(base_spec.get("reply_text", _DEFAULT_REPLY))

        targets[ticket_id] = {
            "priority": priority,
            "category": category,
            "team": team,
            "status": status,
        }
        replies[ticket_id] = reply_text

    order: List[str] = []
    for ticket_id in llm_order + baseline_order:
        if ticket_id in targets and ticket_id not in order:
            order.append(ticket_id)
    for ticket_id in targets:
        if ticket_id not in order:
            order.append(ticket_id)

    return targets, replies, order


def next_action_from_plan(
    observation: Any,
    targets: Dict[str, Dict[str, str]],
    replies: Dict[str, str],
    order: List[str],
) -> SupportOpsAction:
    current = {ticket.ticket_id: ticket for ticket in observation.tickets}

    for ticket_id in order:
        ticket = current.get(ticket_id)
        if ticket is None:
            continue

        target = targets.get(ticket_id)
        if target is None:
            continue

        if ticket.priority != target["priority"]:
            return SupportOpsAction(
                command="set_priority",
                ticket_id=ticket_id,
                value=target["priority"],
            )

        if ticket.category != target["category"]:
            return SupportOpsAction(
                command="set_category",
                ticket_id=ticket_id,
                value=target["category"],
            )

        if ticket.team != target["team"]:
            return SupportOpsAction(
                command="assign_team",
                ticket_id=ticket_id,
                value=target["team"],
            )

        if not ticket.last_reply:
            return SupportOpsAction(
                command="reply",
                ticket_id=ticket_id,
                message=replies.get(ticket_id, _DEFAULT_REPLY),
            )

        if ticket.status != target["status"]:
            return SupportOpsAction(
                command="set_status",
                ticket_id=ticket_id,
                value=target["status"],
            )

    return SupportOpsAction(command="submit")


def model_action_per_step(
    client: OpenAI,
    settings: InferenceSettings,
    observation: Any,
    action_history: List[Dict[str, Any]],
    fallback_action: SupportOpsAction,
) -> SupportOpsAction:
    """Ask the policy model for one next action and validate the output."""

    ticket_summaries = [
        {
            "ticket_id": ticket.ticket_id,
            "subject": ticket.subject,
            "tier": ticket.customer_tier,
            "sla_hours": ticket.sla_hours,
            "priority": ticket.priority,
            "category": ticket.category,
            "team": ticket.team,
            "status": ticket.status,
            "has_reply": bool(ticket.last_reply),
        }
        for ticket in observation.tickets
    ]

    payload: Dict[str, Any] = {
        "task": observation.task_name,
        "difficulty": observation.difficulty,
        "remaining_steps": observation.remaining_steps,
        "current_score": observation.score,
        "grader_breakdown": getattr(observation, "grader_breakdown", {}),
        "last_error": getattr(observation, "last_action_error", None),
        "tickets": ticket_summaries,
        "hints": observation.action_hints,
    }
    if action_history:
        payload["recent_actions"] = action_history[-5:]

    content = _llm_call(
        client,
        settings=settings,
        model=settings.model_name,
        system_prompt=TRIAGE_SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, ensure_ascii=True),
        max_tokens=320,
    )
    if content is None:
        return fallback_action

    parsed = _extract_json(content)
    if parsed is None:
        return fallback_action

    action = _validate_action(parsed, observation)
    return action or fallback_action


def choose_action(
    observation: Any,
    client: Optional[OpenAI],
    settings: InferenceSettings,
    targets: Dict[str, Dict[str, str]],
    replies: Dict[str, str],
    order: List[str],
    action_history: List[Dict[str, Any]],
) -> SupportOpsAction:
    """Select the next action using plan-first orchestration."""

    planned_action = next_action_from_plan(observation, targets, replies, order)

    if settings.force_heuristic:
        return planned_action

    if planned_action.command != "submit":
        return planned_action

    if client is None:
        return planned_action

    has_incomplete_ticket = any(
        ticket.priority is None
        or ticket.category is None
        or ticket.team is None
        or not ticket.last_reply
        for ticket in observation.tickets
    )

    if not has_incomplete_ticket:
        return planned_action

    return model_action_per_step(
        client,
        settings,
        observation,
        action_history,
        fallback_action=planned_action,
    )
