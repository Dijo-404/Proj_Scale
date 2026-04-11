from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class ReplyRule:
    required_keywords: Tuple[str, ...]
    min_length: int = 80


@dataclass(frozen=True)
class TicketSeed:
    ticket_id: str
    subject: str
    customer_tier: Literal["standard", "business", "enterprise"]
    sla_hours: int


@dataclass(frozen=True)
class TicketGoal:
    priority: str
    category: str
    team: str
    status: str
    reply_rule: ReplyRule


@dataclass(frozen=True)
class ProcessRule:
    first_action_ticket: Optional[str] = None
    must_escalate: Tuple[str, ...] = ()
    must_resolve: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    name: str
    difficulty: Difficulty
    description: str
    max_steps: int
    tickets: Tuple[TicketSeed, ...]
    goals: Dict[str, TicketGoal]
    process_rule: ProcessRule
    action_hints: Tuple[str, ...]


def _build_task_library() -> Dict[str, TaskSpec]:
    return {
        "easy_access_recovery": TaskSpec(
            name="easy_access_recovery",
            difficulty="easy",
            description=(
                "A business customer is locked out after changing their phone. "
                "Classify, route, reply with concrete recovery steps, and close the ticket."
            ),
            max_steps=8,
            tickets=(
                TicketSeed(
                    ticket_id="ACC-1001",
                    subject="Payroll dashboard login blocked after MFA reset",
                    customer_tier="business",
                    sla_hours=8,
                ),
            ),
            goals={
                "ACC-1001": TicketGoal(
                    priority="high",
                    category="access",
                    team="tier1",
                    status="resolved",
                    reply_rule=ReplyRule(
                        required_keywords=(
                            "verify",
                            "mfa",
                            "reset",
                            "15 minutes",
                        ),
                        min_length=70,
                    ),
                ),
            },
            process_rule=ProcessRule(
                first_action_ticket="ACC-1001", must_resolve=("ACC-1001",)
            ),
            action_hints=(
                "Set priority/category/team before submitting.",
                "Write a practical customer reply that includes concrete next steps.",
                "Mark status as resolved only after a complete reply.",
            ),
        ),
        "medium_billing_dispute": TaskSpec(
            name="medium_billing_dispute",
            difficulty="medium",
            description=(
                "Handle two billing tickets: a critical enterprise double-charge dispute and "
                "a routine invoice request. Prioritize correctly and route both tickets."
            ),
            max_steps=12,
            tickets=(
                TicketSeed(
                    ticket_id="BILL-2044",
                    subject="Enterprise account billed twice, refund failed",
                    customer_tier="enterprise",
                    sla_hours=2,
                ),
                TicketSeed(
                    ticket_id="BILL-2058",
                    subject="Need April invoice PDF for finance audit",
                    customer_tier="business",
                    sla_hours=24,
                ),
            ),
            goals={
                "BILL-2044": TicketGoal(
                    priority="critical",
                    category="billing",
                    team="billing",
                    status="escalated",
                    reply_rule=ReplyRule(
                        required_keywords=(
                            "apolog",
                            "refund",
                            "invoice",
                            "48 hours",
                        ),
                        min_length=100,
                    ),
                ),
                "BILL-2058": TicketGoal(
                    priority="medium",
                    category="billing",
                    team="billing",
                    status="resolved",
                    reply_rule=ReplyRule(
                        required_keywords=(
                            "invoice",
                            "pdf",
                            "attached",
                        ),
                        min_length=60,
                    ),
                ),
            },
            process_rule=ProcessRule(
                first_action_ticket="BILL-2044",
                must_escalate=("BILL-2044",),
                must_resolve=("BILL-2058",),
            ),
            action_hints=(
                "Prioritize the double-charge dispute first.",
                "Critical enterprise billing issues should be escalated, not auto-closed.",
                "Provide explicit timelines in customer communication.",
            ),
        ),
        "hard_incident_swarm": TaskSpec(
            name="hard_incident_swarm",
            difficulty="hard",
            description=(
                "Coordinate triage for outage, security, and feature tickets during a support surge. "
                "Protect SLA-critical work first while still handling lower-priority requests."
            ),
            max_steps=16,
            tickets=(
                TicketSeed(
                    ticket_id="INC-9001",
                    subject="EU API 500 errors impacting checkout flow",
                    customer_tier="enterprise",
                    sla_hours=1,
                ),
                TicketSeed(
                    ticket_id="SEC-7712",
                    subject="Suspicious OAuth app connected to admin account",
                    customer_tier="enterprise",
                    sla_hours=2,
                ),
                TicketSeed(
                    ticket_id="FEAT-3304",
                    subject="Request: dark-mode CSV export",
                    customer_tier="standard",
                    sla_hours=72,
                ),
            ),
            goals={
                "INC-9001": TicketGoal(
                    priority="critical",
                    category="outage",
                    team="sre",
                    status="escalated",
                    reply_rule=ReplyRule(
                        required_keywords=(
                            "incident",
                            "mitigation",
                            "30 minutes",
                            "status page",
                        ),
                        min_length=110,
                    ),
                ),
                "SEC-7712": TicketGoal(
                    priority="high",
                    category="security",
                    team="security",
                    status="escalated",
                    reply_rule=ReplyRule(
                        required_keywords=(
                            "revoke",
                            "tokens",
                            "security",
                            "investigation",
                        ),
                        min_length=100,
                    ),
                ),
                "FEAT-3304": TicketGoal(
                    priority="low",
                    category="feature_request",
                    team="product",
                    status="in_progress",
                    reply_rule=ReplyRule(
                        required_keywords=(
                            "roadmap",
                            "feature request",
                            "tracking",
                        ),
                        min_length=75,
                    ),
                ),
            },
            process_rule=ProcessRule(
                first_action_ticket="INC-9001",
                must_escalate=("INC-9001", "SEC-7712"),
            ),
            action_hints=(
                "Address the outage first, then security, then feature work.",
                "Escalate outage and security incidents with explicit follow-up timelines.",
                "Do not mark feature requests resolved in the same incident-response episode.",
            ),
        ),
    }


TASK_LIBRARY = _build_task_library()
TASK_ORDER = tuple(TASK_LIBRARY.keys())


def get_task(task_name: str) -> TaskSpec:
    if task_name not in TASK_LIBRARY:
        raise KeyError(f"Unknown task: {task_name}")
    return TASK_LIBRARY[task_name]


def available_tasks() -> Tuple[str, ...]:
    return TASK_ORDER
