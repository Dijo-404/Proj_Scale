"""Support Ops environment client."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        SupportOpsAction,
        SupportOpsObservation,
        SupportOpsReward,
        SupportOpsState,
    )
except ImportError:
    from models import (
        SupportOpsAction,
        SupportOpsObservation,
        SupportOpsReward,
        SupportOpsState,
    )


class SupportOpsEnv(EnvClient[SupportOpsAction, SupportOpsObservation, SupportOpsState]):
    """Typed OpenEnv client for the support triage benchmark."""

    def _step_payload(self, action: SupportOpsAction) -> Dict:
        payload = {
            "command": action.command,
            "ticket_id": action.ticket_id,
            "value": action.value,
            "message": action.message,
            "metadata": action.metadata,
        }
        return {k: v for k, v in payload.items() if v is not None}

    def _parse_result(self, payload: Dict) -> StepResult[SupportOpsObservation]:
        obs_data = payload.get("observation", {})

        reward_details = obs_data.get("reward_details", {})
        if not isinstance(reward_details, dict):
            reward_details = {}

        observation = SupportOpsObservation(
            benchmark=obs_data.get("benchmark", "support_ops_env"),
            task_name=obs_data.get("task_name", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            task_description=obs_data.get("task_description", ""),
            remaining_steps=obs_data.get("remaining_steps", 0),
            score=obs_data.get("score", 0.0),
            progress=obs_data.get("progress", 0.0),
            grader_breakdown=obs_data.get("grader_breakdown", {}),
            reward_details=SupportOpsReward(**reward_details),
            tickets=obs_data.get("tickets", []),
            current_ticket=obs_data.get("current_ticket"),
            available_tasks=obs_data.get("available_tasks", []),
            action_hints=obs_data.get("action_hints", []),
            last_action_summary=obs_data.get("last_action_summary", ""),
            last_action_error=obs_data.get("last_action_error"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SupportOpsState:
        return SupportOpsState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            active_task=payload.get("active_task", ""),
            selected_ticket=payload.get("selected_ticket"),
            score=payload.get("score", 0.0),
            done=payload.get("done", False),
        )
