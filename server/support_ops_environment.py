# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Core environment state machine for the Proj_Scale support-ops benchmark."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from graders import STRICT_SCORE_EPSILON, grade_for_task
from models import (
    SupportOpsAction,
    SupportOpsObservation,
    SupportOpsReward,
    SupportOpsState,
    TicketView,
)
from tasks import (
    ScenarioConfigError,
    TaskSpec,
    available_tasks,
    get_task,
    has_task,
)


class SupportOpsEnvironment(Environment):
    """Real-world customer support triage simulation for agentic RL."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    STEP_PENALTY = 0.01
    INVALID_ACTION_PENALTY = 0.05

    def __init__(self) -> None:
        task_order = available_tasks()
        if not task_order:
            raise ScenarioConfigError("no tasks available from scenario config")

        self._task_idx = -1
        self._task: TaskSpec = get_task(task_order[0])
        self._tickets: Dict[str, Dict] = {}
        self._history: list[Dict] = []
        self._selected_ticket: Optional[str] = None
        self._last_score = 0.0
        self._done = False
        self._state = SupportOpsState(episode_id=str(uuid4()), step_count=0)
        self._set_task(task_order[0], rotate=False)

    def _new_ticket_state(self, ticket_seed) -> Dict:
        return {
            "ticket_id": ticket_seed.ticket_id,
            "subject": ticket_seed.subject,
            "customer_tier": ticket_seed.customer_tier,
            "sla_hours": ticket_seed.sla_hours,
            "status": "new",
            "priority": None,
            "category": None,
            "team": None,
            "last_reply": None,
        }

    def _set_task(self, task_name: str, rotate: bool = False) -> None:
        self._task = get_task(task_name)
        self._tickets = {
            seed.ticket_id: self._new_ticket_state(seed) for seed in self._task.tickets
        }
        self._history = []
        self._selected_ticket = None
        self._last_score = 0.0
        self._done = False
        self._state = SupportOpsState(
            episode_id=str(uuid4()),
            step_count=0,
            active_task=self._task.name,
            selected_ticket=None,
            score=STRICT_SCORE_EPSILON,
            done=False,
        )
        if rotate:
            task_order = available_tasks()
            self._task_idx = (self._task_idx + 1) % len(task_order)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs,
    ) -> SupportOpsObservation:
        task_order = available_tasks()
        if not task_order:
            raise ScenarioConfigError("no tasks available from scenario config")

        requested_task = task_name or kwargs.get("task") or kwargs.get("task_name")

        if requested_task is None:
            next_idx = (self._task_idx + 1) % len(task_order)
            requested_task = task_order[next_idx]

        if not has_task(requested_task):
            requested_task = task_order[0]

        self._set_task(requested_task, rotate=True)

        if episode_id:
            self._state.episode_id = episode_id

        return self._build_observation(
            reward=0.0,
            done=False,
            summary=f"Reset on task {self._task.name}",
            error=None,
            progress_delta=0.0,
            invalid_penalty=0.0,
        )

    def step(
        self, action: SupportOpsAction, timeout_s: Optional[float] = None, **kwargs
    ) -> SupportOpsObservation:  # type: ignore[override]
        if self._done:
            return self._build_observation(
                reward=-0.1,
                done=True,
                summary="Ignored action after episode completion",
                error="Episode already completed. Call reset() to start a new task.",
                progress_delta=0.0,
                invalid_penalty=0.1,
            )

        self._state.step_count += 1
        error = self._apply_action(action)

        breakdown = grade_for_task(self._task.name, self._tickets, self._history)
        score = breakdown["total"]
        progress_delta = score - self._last_score

        invalid_penalty = self.INVALID_ACTION_PENALTY if error else 0.0
        reward = progress_delta - self.STEP_PENALTY - invalid_penalty

        done = (
            action.command == "submit" or self._state.step_count >= self._task.max_steps
        )
        if done and action.command != "submit":
            reward -= 0.02
        if done and score >= 0.95:
            reward += 0.05

        reward = max(-1.0, min(1.0, reward))

        self._last_score = score
        self._done = done
        self._state.score = score
        self._state.done = done
        self._state.selected_ticket = self._selected_ticket

        summary = self._summarize_action(action, error)
        return self._build_observation(
            reward=reward,
            done=done,
            summary=summary,
            error=error,
            progress_delta=progress_delta,
            invalid_penalty=invalid_penalty,
            breakdown=breakdown,
        )

    @property
    def state(self) -> SupportOpsState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Proj_Scale",
            description=(
                "A real-world support operations environment where agents triage tickets, "
                "route work, communicate with customers, and optimize SLA outcomes."
            ),
            version="1.0.0",
        )

    def _apply_action(self, action: SupportOpsAction) -> Optional[str]:
        entry = {
            "step": self._state.step_count,
            "command": action.command,
            "ticket_id": action.ticket_id,
            "value": action.value,
            "message": action.message,
        }
        self._history.append(deepcopy(entry))

        if action.command == "submit":
            return None

        if not action.ticket_id:
            return "ticket_id is required for this command"

        ticket = self._tickets.get(action.ticket_id)
        if ticket is None:
            return f"Unknown ticket_id: {action.ticket_id}"

        self._selected_ticket = action.ticket_id

        if action.command == "set_priority":
            allowed = {"low", "medium", "high", "critical"}
            if action.value not in allowed:
                return f"Invalid priority '{action.value}'. Allowed: {sorted(allowed)}"
            ticket["priority"] = action.value
            return None

        if action.command == "set_category":
            allowed = {"access", "billing", "outage", "security", "feature_request"}
            if action.value not in allowed:
                return f"Invalid category '{action.value}'. Allowed: {sorted(allowed)}"
            ticket["category"] = action.value
            return None

        if action.command == "assign_team":
            allowed = {"tier1", "billing", "sre", "security", "product"}
            if action.value not in allowed:
                return f"Invalid team '{action.value}'. Allowed: {sorted(allowed)}"
            ticket["team"] = action.value
            return None

        if action.command == "set_status":
            allowed = {"new", "in_progress", "resolved", "escalated"}
            if action.value not in allowed:
                return f"Invalid status '{action.value}'. Allowed: {sorted(allowed)}"
            ticket["status"] = action.value
            return None

        if action.command == "reply":
            message = (action.message or "").strip()
            if len(message) < 20:
                return "Reply message must be at least 20 characters"
            ticket["last_reply"] = message
            if ticket["status"] == "new":
                ticket["status"] = "in_progress"
            return None

        return f"Unsupported command: {action.command}"

    def _ticket_views(self) -> list[TicketView]:
        ordered_ids = [seed.ticket_id for seed in self._task.tickets]
        return [TicketView(**self._tickets[ticket_id]) for ticket_id in ordered_ids]

    def _summarize_action(self, action: SupportOpsAction, error: Optional[str]) -> str:
        if error:
            return f"{action.command} failed: {error}"

        if action.command == "submit":
            return "Submitted current triage plan for grading"

        parts = [action.command]
        if action.ticket_id:
            parts.append(action.ticket_id)
        if action.value:
            parts.append(action.value)
        if action.message:
            parts.append("reply_updated")
        return " ".join(parts)

    def _build_observation(
        self,
        reward: float,
        done: bool,
        summary: str,
        error: Optional[str],
        progress_delta: float,
        invalid_penalty: float,
        breakdown: Optional[Dict[str, float]] = None,
    ) -> SupportOpsObservation:
        if breakdown is None:
            breakdown = grade_for_task(self._task.name, self._tickets, self._history)

        score = breakdown["total"]
        self._state.score = score
        self._state.done = done
        self._state.selected_ticket = self._selected_ticket

        return SupportOpsObservation(
            task_name=self._task.name,
            difficulty=self._task.difficulty,
            task_description=self._task.description,
            remaining_steps=max(0, self._task.max_steps - self._state.step_count),
            score=score,
            grader_breakdown=breakdown,
            reward_details=SupportOpsReward(
                total=reward,
                progress_delta=progress_delta,
                step_penalty=self.STEP_PENALTY,
                invalid_action_penalty=invalid_penalty,
            ),
            tickets=self._ticket_views(),
            current_ticket=self._selected_ticket,
            action_hints=list(self._task.action_hints),
            last_action_summary=summary,
            last_action_error=error,
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
            },
        )
