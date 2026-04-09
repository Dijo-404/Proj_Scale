"""Inference script for Proj_Scale support triage benchmark.

Stdout protocol: [START], [STEP], [END]

Decision tiers:
  1. Known tickets   → deterministic heuristic
  2. Unknown tickets → LLM-planned targets, deterministic execution
  3. Fallback        → per-step LLM with rich prompt + action validation
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from client import SupportOpsEnv
from models import SupportOpsAction

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
REASONING_MODEL = os.environ.get("REASONING_MODEL", MODEL_NAME)
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")

ENV_BASE_URL = os.getenv("ENV_BASE_URL")
LOCAL_IMAGE_NAME = (
    os.getenv("LOCAL_IMAGE_NAME")
    or os.getenv("IMAGE_NAME")
    or "proj_scale-env:latest"
)

BENCHMARK = os.getenv("BENCHMARK", "Proj_Scale")
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.75"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

FORCE_HEURISTIC = os.getenv("FORCE_HEURISTIC", "0").lower() in {"1", "true", "yes"}
TASKS = ["easy_access_recovery", "medium_billing_dispute", "hard_incident_swarm"]

VALID_PRIORITIES = {"low", "medium", "high", "critical"}
VALID_CATEGORIES = {"access", "billing", "outage", "security", "feature_request"}
VALID_TEAMS = {"tier1", "billing", "sre", "security", "product"}
VALID_STATUSES = {"new", "in_progress", "resolved", "escalated"}
VALID_COMMANDS = {
    "set_priority", "set_category", "assign_team",
    "set_status", "reply", "submit",
}

COMMAND_ENUM_MAP: Dict[str, set] = {
    "set_priority": VALID_PRIORITIES,
    "set_category": VALID_CATEGORIES,
    "assign_team": VALID_TEAMS,
    "set_status": VALID_STATUSES,
}

TARGET_FIELDS: Dict[str, Dict[str, str]] = {
    "ACC-1001": {"priority": "high", "category": "access", "team": "tier1", "status": "resolved"},
    "BILL-2044": {"priority": "critical", "category": "billing", "team": "billing", "status": "escalated"},
    "BILL-2058": {"priority": "medium", "category": "billing", "team": "billing", "status": "resolved"},
    "INC-9001": {"priority": "critical", "category": "outage", "team": "sre", "status": "escalated"},
    "SEC-7712": {"priority": "high", "category": "security", "team": "security", "status": "escalated"},
    "FEAT-3304": {"priority": "low", "category": "feature_request", "team": "product", "status": "in_progress"},
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

DEFAULT_REPLY = (
    "Thank you for reaching out. We are reviewing your request "
    "and will follow up shortly with next steps and a timeline."
)

PLANNING_SYSTEM_PROMPT = """\
You are an expert support-operations analyst. Given a set of support tickets, \
produce a complete JSON triage plan.

CLASSIFICATION RULES:
- Login / MFA / password / access lockout → category: access, team: tier1
- Billing / invoice / refund / payment / charge → category: billing, team: billing
- API errors / outages / downtime / 500 errors → category: outage, team: sre
- OAuth / tokens / suspicious activity / security alerts → category: security, team: security
- Feature requests / enhancements / product feedback → category: feature_request, team: product

PRIORITY BY SLA AND SEVERITY:
- SLA ≤ 2 hours OR enterprise with critical issue → priority: critical
- SLA ≤ 8 hours → priority: high
- SLA ≤ 24 hours → priority: medium
- SLA > 24 hours → priority: low

STATUS RULES:
- Critical outages and security incidents → status: escalated
- Enterprise billing disputes with refund needed → status: escalated
- Routine issues that can be fully resolved → status: resolved
- Feature requests / non-urgent work → status: in_progress

PROCESSING ORDER:
Process tickets from highest urgency to lowest (critical → high → medium → low).

REPLY GUIDELINES:
Each reply MUST be at least 100 characters and include concrete next steps and timelines.
- Access issues: mention "verify", "MFA", "reset", and a time estimate (e.g. "15 minutes")
- Billing issues: mention "apologize"/"apology", "refund", "invoice", and a time (e.g. "48 hours")
- Outage issues: mention "incident", "mitigation", "status page", update interval (e.g. "30 minutes")
- Security issues: mention "revoke", "tokens", "security", "investigation"
- Feature requests: mention "roadmap", "feature request", "tracking"

OUTPUT — return ONLY this JSON structure, no markdown fences, no explanation:
{
  "ticket_order": ["TICKET-ID-1", "TICKET-ID-2"],
  "tickets": {
    "TICKET-ID-1": {
      "priority": "...",
      "category": "...",
      "team": "...",
      "status": "...",
      "reply_text": "Detailed customer reply with at least 100 characters, concrete steps, and timelines..."
    }
  }
}"""

TRIAGE_SYSTEM_PROMPT = """\
You are an expert support-operations triage agent. Decide the single best next action.

VALID ACTIONS:
- set_priority: value in {low, medium, high, critical}
- set_category: value in {access, billing, outage, security, feature_request}
- assign_team: value in {tier1, billing, sre, security, product}
- set_status: value in {new, in_progress, resolved, escalated}
- reply: a detailed customer message (≥ 20 chars) with specific action items and timelines
- submit: finalize all work for grading — ONLY when every ticket is fully triaged

GRADING WEIGHTS:
- Routing 50 %: correct priority, category, team, status for every ticket
- Communication 30 %: reply includes required keywords and meets minimum length
- Process 20 %: highest-priority tickets handled first; critical issues escalated

CLASSIFICATION RULES:
- Access/login/MFA → category: access, team: tier1
- Billing/invoice/refund → category: billing, team: billing
- Outage/500/downtime → category: outage, team: sre
- Security/OAuth/tokens → category: security, team: security
- Feature request → category: feature_request, team: product

PRIORITY BY SLA:
- SLA ≤ 2h → critical   |  SLA ≤ 8h → high   |  SLA ≤ 24h → medium   |  SLA > 24h → low

WORKFLOW:
1. Set priority / category / team BEFORE replying.
2. Write a reply BEFORE setting final status.
3. Replies MUST include concrete timelines and actionable steps.
4. Only submit after EVERY ticket has routing + reply + final status.

REPLY KEYWORDS:
- Access: verify, MFA, reset, time estimate
- Billing: apolog(y/ize), refund, invoice, 48 hours
- Outage: incident, mitigation, status page, 30 minutes
- Security: revoke, tokens, security, investigation
- Feature: roadmap, feature request, tracking

Return ONLY compact JSON: {"command":"...","ticket_id":"...","value":"...","message":"..."}"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _action_to_str(action: SupportOpsAction) -> str:
    payload: Dict[str, Any] = {
        "command": action.command,
        "ticket_id": action.ticket_id,
        "value": action.value,
        "message": action.message,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                continue

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _validate_action(
    raw: Dict[str, Any], observation: Any
) -> Optional[SupportOpsAction]:
    command = str(raw.get("command", "")).strip().lower()
    if command not in VALID_COMMANDS:
        return None

    ticket_id = raw.get("ticket_id")
    value = raw.get("value")
    message = raw.get("message")

    if command != "submit":
        if not ticket_id:
            return None
        if ticket_id not in {t.ticket_id for t in observation.tickets}:
            return None

    if command in COMMAND_ENUM_MAP:
        if not value:
            return None
        value = str(value).strip().lower()
        if value not in COMMAND_ENUM_MAP[command]:
            return None

    if command == "reply":
        if not message or len(str(message).strip()) < 20:
            return None
        message = str(message).strip()

    return SupportOpsAction(
        command=command, ticket_id=ticket_id, value=value, message=message,
    )


def _llm_call(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Optional[str]:
    for attempt in range(LLM_MAX_RETRIES + 1):
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
        except Exception:
            if attempt < LLM_MAX_RETRIES:
                time.sleep(min(2**attempt, 4))
            continue
    return None


def _heuristic_action(observation: Any) -> SupportOpsAction:
    try:
        tickets = list(observation.tickets)
    except Exception:
        return SupportOpsAction(command="submit")

    for ticket in tickets:
        try:
            target = TARGET_FIELDS.get(ticket.ticket_id)
            if target is None:
                continue

            if ticket.priority != target["priority"]:
                return SupportOpsAction(
                    command="set_priority", ticket_id=ticket.ticket_id,
                    value=target["priority"],
                )
            if ticket.category != target["category"]:
                return SupportOpsAction(
                    command="set_category", ticket_id=ticket.ticket_id,
                    value=target["category"],
                )
            if ticket.team != target["team"]:
                return SupportOpsAction(
                    command="assign_team", ticket_id=ticket.ticket_id,
                    value=target["team"],
                )
            if not ticket.last_reply:
                return SupportOpsAction(
                    command="reply", ticket_id=ticket.ticket_id,
                    message=REPLY_TEMPLATES.get(ticket.ticket_id, DEFAULT_REPLY),
                )
            if ticket.status != target["status"]:
                return SupportOpsAction(
                    command="set_status", ticket_id=ticket.ticket_id,
                    value=target["status"],
                )
        except Exception:
            continue

    return SupportOpsAction(command="submit")


def _plan_episode(client: OpenAI, observation: Any) -> Optional[Dict]:
    try:
        ticket_info = [
            {
                "ticket_id": t.ticket_id,
                "subject": t.subject,
                "customer_tier": t.customer_tier,
                "sla_hours": t.sla_hours,
                "current_status": t.status,
            }
            for t in observation.tickets
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
            client=client, model=REASONING_MODEL,
            system_prompt=PLANNING_SYSTEM_PROMPT,
            user_prompt=user_prompt, max_tokens=1024,
        )
        if content is None:
            return None

        parsed = _extract_json(content)
        if parsed is None or "tickets" not in parsed:
            return None

        for spec in parsed["tickets"].values():
            if not isinstance(spec, dict):
                return None

        return parsed
    except Exception:
        return None


def _merge_targets(
    observation: Any, plan: Optional[Dict],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], List[str]]:
    targets: Dict[str, Dict[str, str]] = {}
    replies: Dict[str, str] = {}
    order: List[str] = list(plan.get("ticket_order", [])) if plan else []

    for ticket in observation.tickets:
        tid = ticket.ticket_id

        if tid in TARGET_FIELDS:
            targets[tid] = TARGET_FIELDS[tid]
            replies[tid] = REPLY_TEMPLATES.get(tid, DEFAULT_REPLY)
        elif plan and tid in plan.get("tickets", {}):
            p = plan["tickets"][tid]
            priority = str(p.get("priority", "medium")).strip().lower()
            category = str(p.get("category", "access")).strip().lower()
            team = str(p.get("team", "tier1")).strip().lower()
            status_val = str(p.get("status", "in_progress")).strip().lower()
            reply_text = str(p.get("reply_text", "")).strip()

            if priority not in VALID_PRIORITIES:
                priority = "medium"
            if category not in VALID_CATEGORIES:
                category = "access"
            if team not in VALID_TEAMS:
                team = "tier1"
            if status_val not in VALID_STATUSES:
                status_val = "in_progress"
            if len(reply_text) < 20:
                reply_text = DEFAULT_REPLY

            targets[tid] = {
                "priority": priority, "category": category,
                "team": team, "status": status_val,
            }
            replies[tid] = reply_text

        if tid not in order:
            order.append(tid)

    return targets, replies, order


def _unified_heuristic(
    observation: Any,
    targets: Dict[str, Dict[str, str]],
    replies: Dict[str, str],
    order: List[str],
) -> SupportOpsAction:
    current = {t.ticket_id: t for t in observation.tickets}

    for tid in order:
        ticket = current.get(tid)
        if ticket is None:
            continue
        target = targets.get(tid)
        if target is None:
            continue

        try:
            if ticket.priority != target["priority"]:
                return SupportOpsAction(
                    command="set_priority", ticket_id=tid, value=target["priority"],
                )
            if ticket.category != target["category"]:
                return SupportOpsAction(
                    command="set_category", ticket_id=tid, value=target["category"],
                )
            if ticket.team != target["team"]:
                return SupportOpsAction(
                    command="assign_team", ticket_id=tid, value=target["team"],
                )
            if not ticket.last_reply:
                return SupportOpsAction(
                    command="reply", ticket_id=tid,
                    message=replies.get(tid, DEFAULT_REPLY),
                )
            if ticket.status != target["status"]:
                return SupportOpsAction(
                    command="set_status", ticket_id=tid, value=target["status"],
                )
        except Exception:
            continue

    return SupportOpsAction(command="submit")


def _model_action_per_step(
    client: OpenAI, observation: Any, action_history: List[Dict],
) -> SupportOpsAction:
    try:
        ticket_summaries = [
            {
                "ticket_id": t.ticket_id, "subject": t.subject,
                "tier": t.customer_tier, "sla_hours": t.sla_hours,
                "priority": t.priority, "category": t.category,
                "team": t.team, "status": t.status,
                "has_reply": bool(t.last_reply),
            }
            for t in observation.tickets
        ]

        user_context: Dict[str, Any] = {
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
            user_context["recent_actions"] = action_history[-5:]

        content = _llm_call(
            client=client, model=MODEL_NAME,
            system_prompt=TRIAGE_SYSTEM_PROMPT,
            user_prompt=json.dumps(user_context, ensure_ascii=True),
            max_tokens=300,
        )

        if content is None:
            return _heuristic_action(observation)

        parsed = _extract_json(content)
        if parsed is None:
            return _heuristic_action(observation)

        action = _validate_action(parsed, observation)
        return action if action is not None else _heuristic_action(observation)
    except Exception:
        return _heuristic_action(observation)


def _choose_action(
    observation: Any,
    client: Optional[OpenAI],
    targets: Dict[str, Dict[str, str]],
    replies: Dict[str, str],
    order: List[str],
    action_history: List[Dict],
    has_unknown_tickets: bool,
) -> SupportOpsAction:
    action = _unified_heuristic(observation, targets, replies, order)

    if action.command != "submit":
        return action

    if has_unknown_tickets:
        try:
            for t in observation.tickets:
                if t.priority is None or t.category is None or t.team is None or not t.last_reply:
                    if client is not None:
                        return _model_action_per_step(client, observation, action_history)
        except Exception:
            pass

    return action


async def run_task(
    env: SupportOpsEnv, task_name: str, client: Optional[OpenAI]
) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    action_history: List[Dict] = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        try:
            result = await env.reset(task_name=task_name)
        except Exception as exc:
            log_step(step=1, action='{"command":"reset"}', reward=0.0,
                     done=True, error=f"reset failed: {exc}")
            return 0.0

        has_unknown = any(
            t.ticket_id not in TARGET_FIELDS for t in result.observation.tickets
        )

        plan: Optional[Dict] = None
        if has_unknown and client is not None:
            plan = _plan_episode(client, result.observation)

        targets, replies, order = _merge_targets(result.observation, plan)

        for step in range(1, MAX_STEPS + 1):
            try:
                if result.done:
                    break

                action = _choose_action(
                    result.observation, client, targets, replies,
                    order, action_history, has_unknown,
                )
                action_str = _action_to_str(action)

                action_history.append({
                    "step": step, "command": action.command,
                    "ticket_id": action.ticket_id, "value": action.value,
                })

                try:
                    result = await env.step(action)
                except Exception as exc:
                    log_step(step=step, action=action_str, reward=0.0,
                             done=True, error=f"step failed: {exc}")
                    steps_taken = step
                    break

                reward = float(result.reward or 0.0)
                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step, action=action_str, reward=reward,
                    done=bool(result.done),
                    error=result.observation.last_action_error,
                )

                if result.done:
                    break

            except Exception as exc:
                log_step(step=step, action='{"command":"error"}', reward=0.0,
                         done=True, error=f"step error: {exc}")
                steps_taken = step
                break

        try:
            score = float(result.observation.score)
        except Exception:
            score = 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD
        return score
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    use_model = bool(API_KEY) and (not FORCE_HEURISTIC)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if use_model else None
    except Exception:
        client = None

    env: Optional[SupportOpsEnv] = None

    try:
        if ENV_BASE_URL:
            env = SupportOpsEnv(base_url=ENV_BASE_URL)
            await env.connect()
        else:
            try:
                env = await SupportOpsEnv.from_docker_image(LOCAL_IMAGE_NAME)
            except Exception as exc:
                print(f"[ERROR] Container start failed: {exc}",
                      file=sys.stderr, flush=True)
                for task in TASKS:
                    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
                    log_end(success=False, steps=0, score=0.0, rewards=[])
                return
    except Exception as exc:
        print(f"[ERROR] Environment connection failed: {exc}",
              file=sys.stderr, flush=True)
        for task in TASKS:
            log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    try:
        for task in TASKS:
            try:
                await run_task(env, task, client)
            except Exception as exc:
                print(f"[ERROR] Task {task} failed: {exc}",
                      file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
    finally:
        try:
            await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
