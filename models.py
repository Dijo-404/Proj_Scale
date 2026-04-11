# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Typed data models for the Proj_Scale OpenEnv benchmark."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field

Priority = Literal["low", "medium", "high", "critical"]
Category = Literal["access", "billing", "outage", "security", "feature_request"]
Team = Literal["tier1", "billing", "sre", "security", "product"]
TicketStatus = Literal["new", "in_progress", "resolved", "escalated"]
Command = Literal[
    "set_priority",
    "set_category",
    "assign_team",
    "set_status",
    "reply",
    "submit",
]


class TicketView(BaseModel):
    """Observed ticket state exposed to the agent."""

    ticket_id: str
    subject: str
    customer_tier: Literal["standard", "business", "enterprise"]
    sla_hours: int = Field(..., ge=1)
    status: TicketStatus = "new"
    priority: Optional[Priority] = None
    category: Optional[Category] = None
    team: Optional[Team] = None
    last_reply: Optional[str] = None


class SupportOpsReward(BaseModel):
    """Typed reward diagnostics for transparent shaping."""

    total: float = 0.0
    progress_delta: float = 0.0
    step_penalty: float = 0.0
    invalid_action_penalty: float = 0.0


class SupportOpsAction(Action):
    """Action model for ticket triage and response operations."""

    command: Command
    ticket_id: Optional[str] = None
    value: Optional[str] = None
    message: Optional[str] = None


class SupportOpsObservation(Observation):
    """Observation model returned on reset and step."""

    task_name: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    task_description: str = ""
    remaining_steps: int = 0
    score: float = Field(default=1e-4, gt=0.0, lt=1.0)
    grader_breakdown: Dict[str, float] = Field(default_factory=dict)
    reward_details: SupportOpsReward = Field(default_factory=SupportOpsReward)
    tickets: List[TicketView] = Field(default_factory=list)
    current_ticket: Optional[str] = None
    action_hints: List[str] = Field(default_factory=list)
    last_action_summary: str = ""
    last_action_error: Optional[str] = None


class SupportOpsState(State):
    """Internal state object surfaced via state()."""

    # Declared explicitly for clarity even though these come from the OpenEnv base State.
    episode_id: Optional[str] = None
    step_count: int = 0
    active_task: str = ""
    selected_ticket: Optional[str] = None
    score: float = Field(default=1e-4, gt=0.0, lt=1.0)
    done: bool = False
