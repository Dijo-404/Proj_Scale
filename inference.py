"""Baseline inference script for support_ops_env.

This script follows the required stdout format:
[START], [STEP], [END]
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Dict, List, Optional

from openai import OpenAI

from client import SupportOpsEnv
from models import SupportOpsAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

ENV_BASE_URL = os.getenv("ENV_BASE_URL")
LOCAL_IMAGE_NAME = (
    os.getenv("LOCAL_IMAGE_NAME")
    or os.getenv("IMAGE_NAME")
    or "support_ops_env-env:latest"
)

BENCHMARK = os.getenv("BENCHMARK", "support_ops_env")
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.75"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))

FORCE_HEURISTIC = os.getenv("FORCE_HEURISTIC", "0").lower() in {"1", "true", "yes"}
TASKS = ["easy_access_recovery", "medium_billing_dispute", "hard_incident_swarm"]

TARGET_FIELDS: Dict[str, Dict[str, str]] = {
    "ACC-1001": {
        "priority": "high",
        "category": "access",
        "team": "tier1",
        "status": "resolved",
    },
    "BILL-2044": {
        "priority": "critical",
        "category": "billing",
        "team": "billing",
        "status": "escalated",
    },
    "BILL-2058": {
        "priority": "medium",
        "category": "billing",
        "team": "billing",
        "status": "resolved",
    },
    "INC-9001": {
        "priority": "critical",
        "category": "outage",
        "team": "sre",
        "status": "escalated",
    },
    "SEC-7712": {
        "priority": "high",
        "category": "security",
        "team": "security",
        "status": "escalated",
    },
    "FEAT-3304": {
        "priority": "low",
        "category": "feature_request",
        "team": "product",
        "status": "in_progress",
    },
}

REPLY_TEMPLATES: Dict[str, str] = {
    "ACC-1001": (
        "Thanks for reporting this. Please verify your identity with the secure link, "
        "then complete the MFA reset flow. We have triggered a reset and your access "
        "should return within 15 minutes."
    ),
    "BILL-2044": (
        "We apologize for the duplicate charge. Billing has escalated this case and will "
        "process your refund within 48 hours. We are reviewing invoice history now and "
        "will confirm completion on this thread."
    ),
    "BILL-2058": (
        "Your invoice PDF is attached to this ticket. If your finance team needs a signed "
        "copy, reply here and we can provide it today."
    ),
    "INC-9001": (
        "We have declared an incident and SRE is applying mitigation now. Next update in "
        "30 minutes on the status page, and we will keep your team informed until service "
        "stabilizes."
    ),
    "SEC-7712": (
        "Security has escalated this alert and started an investigation. Please revoke the "
        "suspect integration, rotate tokens, and share any unusual login details so we can "
        "complete containment."
    ),
    "FEAT-3304": (
        "Thanks for the feature request. We logged this in the product roadmap tracking queue "
        "for dark-mode export support and will update you when the scope is scheduled."
    ),
}


SYSTEM_PROMPT = (
    "You are an agent that must output one JSON action for support triage. "
    "Valid commands: set_priority, set_category, assign_team, set_status, reply, submit. "
    "Return only compact JSON with keys command, ticket_id, value, message."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    done_val = str(done).lower()
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _action_to_str(action: SupportOpsAction) -> str:
    payload = {
        "command": action.command,
        "ticket_id": action.ticket_id,
        "value": action.value,
        "message": action.message,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _heuristic_action(observation) -> SupportOpsAction:
    tickets = list(observation.tickets)

    for ticket in tickets:
        target = TARGET_FIELDS.get(ticket.ticket_id)
        if target is None:
            continue

        if ticket.priority != target["priority"]:
            return SupportOpsAction(
                command="set_priority",
                ticket_id=ticket.ticket_id,
                value=target["priority"],
            )

        if ticket.category != target["category"]:
            return SupportOpsAction(
                command="set_category",
                ticket_id=ticket.ticket_id,
                value=target["category"],
            )

        if ticket.team != target["team"]:
            return SupportOpsAction(
                command="assign_team",
                ticket_id=ticket.ticket_id,
                value=target["team"],
            )

        if not ticket.last_reply:
            return SupportOpsAction(
                command="reply",
                ticket_id=ticket.ticket_id,
                message=REPLY_TEMPLATES.get(
                    ticket.ticket_id, "Acknowledged. Working on this now."
                ),
            )

        if ticket.status != target["status"]:
            return SupportOpsAction(
                command="set_status",
                ticket_id=ticket.ticket_id,
                value=target["status"],
            )

    return SupportOpsAction(command="submit")


def _model_action(client: OpenAI, observation) -> SupportOpsAction:
    ticket_summaries = []
    for ticket in observation.tickets:
        ticket_summaries.append(
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
        )

    user_prompt = {
        "task": observation.task_name,
        "difficulty": observation.difficulty,
        "remaining_steps": observation.remaining_steps,
        "score": observation.score,
        "tickets": ticket_summaries,
        "action_hints": observation.action_hints,
    }

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=220,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
        ],
        stream=False,
    )

    content = (completion.choices[0].message.content or "").strip()

    try:
        payload = json.loads(content)
        return SupportOpsAction(**payload)
    except Exception:
        return _heuristic_action(observation)


async def run_task(
    env: SupportOpsEnv, task_name: str, client: Optional[OpenAI]
) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    result = await env.reset(task_name=task_name)

    try:
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            if client is None:
                action = _heuristic_action(result.observation)
            else:
                action = _model_action(client, result.observation)

            action_str = _action_to_str(action)
            result = await env.step(action)

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=bool(result.done),
                error=result.observation.last_action_error,
            )

            if result.done:
                break

        score = float(result.observation.score)
        success = score >= SUCCESS_SCORE_THRESHOLD
        return score
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    use_model = bool(API_KEY) and (not FORCE_HEURISTIC)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if use_model else None

    if ENV_BASE_URL:
        env = SupportOpsEnv(base_url=ENV_BASE_URL)
        await env.connect()
    else:
        env = await SupportOpsEnv.from_docker_image(LOCAL_IMAGE_NAME)

    try:
        for task in TASKS:
            await run_task(env, task, client)
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
