# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Typed client for interacting with the Proj_Scale OpenEnv server."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import (
    SupportOpsAction,
    SupportOpsObservation,
    SupportOpsReward,
    SupportOpsState,
)


class SupportOpsEnv(
    EnvClient[SupportOpsAction, SupportOpsObservation, SupportOpsState]
):
    """Typed OpenEnv client for the support triage benchmark.

    Examples:
        Direct HTTP usage:
            env = SupportOpsEnv(base_url="http://localhost:8000")
            await env.connect()
            result = await env.reset(task_name="easy_access_recovery")
            await env.close()

        WebSocket-friendly context manager:
            async with SupportOpsEnv(base_url="http://localhost:8000") as env:
                await env.reset(task_name="medium_billing_dispute")

        Local image bootstrap:
            env = await SupportOpsEnv.from_docker_image("proj_scale-env:latest")
    """

    async def __aenter__(self) -> "SupportOpsEnv":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _step_payload(self, action: SupportOpsAction) -> Dict:
        """Serialize a typed action into the OpenEnv step payload format."""

        payload = {
            "command": action.command,
            "ticket_id": action.ticket_id,
            "value": action.value,
            "message": action.message,
        }
        return {k: v for k, v in payload.items() if v is not None}

    def _parse_result(self, payload: Dict) -> StepResult[SupportOpsObservation]:
        """Parse OpenEnv step/reset response payload into typed observation + result."""

        obs_data = payload.get("observation", {})

        reward_details = obs_data.get("reward_details", {})
        if not isinstance(reward_details, dict):
            reward_details = {}

        observation = SupportOpsObservation(
            task_name=obs_data.get("task_name", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            task_description=obs_data.get("task_description", ""),
            remaining_steps=obs_data.get("remaining_steps", 0),
            score=obs_data.get("score", 0.0),
            grader_breakdown=obs_data.get("grader_breakdown", {}),
            reward_details=SupportOpsReward(**reward_details),
            tickets=obs_data.get("tickets", []),
            current_ticket=obs_data.get("current_ticket"),
            action_hints=obs_data.get("action_hints", []),
            last_action_summary=obs_data.get("last_action_summary", ""),
            last_action_error=obs_data.get("last_action_error"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SupportOpsState:
        """Parse OpenEnv state response payload into typed environment state."""

        return SupportOpsState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            active_task=payload.get("active_task", ""),
            selected_ticket=payload.get("selected_ticket"),
            score=payload.get("score", 0.0),
            done=payload.get("done", False),
        )
