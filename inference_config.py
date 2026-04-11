# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Configuration for Proj_Scale inference runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from tasks import available_tasks

VALID_PRIORITIES = {"low", "medium", "high", "critical"}
VALID_CATEGORIES = {"access", "billing", "outage", "security", "feature_request"}
VALID_TEAMS = {"tier1", "billing", "sre", "security", "product"}
VALID_STATUSES = {"new", "in_progress", "resolved", "escalated"}
VALID_COMMANDS = {
    "set_priority",
    "set_category",
    "assign_team",
    "set_status",
    "reply",
    "submit",
}

COMMAND_ENUM_MAP = {
    "set_priority": VALID_PRIORITIES,
    "set_category": VALID_CATEGORIES,
    "assign_team": VALID_TEAMS,
    "set_status": VALID_STATUSES,
}


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class InferenceSettings:
    """Runtime settings for one inference execution."""

    api_base_url: str
    model_name: str
    reasoning_model: str
    api_key: Optional[str]
    env_base_url: Optional[str]
    local_image_name: str
    benchmark: str
    success_score_threshold: float
    max_steps: int
    llm_max_retries: int
    force_heuristic: bool
    task_name: str
    task_names: Tuple[str, ...]

    @property
    def use_llm(self) -> bool:
        return not self.force_heuristic

    @classmethod
    def from_env(
        cls,
        *,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> "InferenceSettings":
        task_names = tuple(available_tasks())
        default_task = task_names[0] if task_names else "easy_access_recovery"
        task_name = os.getenv("TASK_NAME", default_task)
        if task_name not in task_names:
            task_name = default_task

        resolved_model_name = model_name or os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        resolved_api_base_url = api_base_url or os.getenv(
            "API_BASE_URL",
            "https://router.huggingface.co/v1",
        )
        resolved_hf_token = hf_token if hf_token is not None else (os.getenv("API_KEY") or os.getenv("HF_TOKEN"))

        return cls(
            api_base_url=resolved_api_base_url,
            model_name=resolved_model_name,
            reasoning_model=os.getenv("REASONING_MODEL", resolved_model_name),
            api_key=resolved_hf_token,
            env_base_url=os.getenv("ENV_BASE_URL"),
            local_image_name=(
                os.getenv("LOCAL_IMAGE_NAME")
                or os.getenv("IMAGE_NAME")
                or "proj_scale-env:latest"
            ),
            benchmark=os.getenv("BENCHMARK", "Proj_Scale"),
            success_score_threshold=float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.75")),
            max_steps=int(os.getenv("MAX_STEPS", "20")),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            force_heuristic=_parse_bool(os.getenv("FORCE_HEURISTIC", "0")),
            task_name=task_name,
            task_names=task_names,
        )

    def with_overrides(
        self,
        *,
        task_name: Optional[str] = None,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        reasoning_model: Optional[str] = None,
        env_base_url: Optional[str] = None,
        local_image_name: Optional[str] = None,
        max_steps: Optional[int] = None,
        success_score_threshold: Optional[float] = None,
        force_heuristic: Optional[bool] = None,
    ) -> "InferenceSettings":
        next_model = model_name or self.model_name
        next_reasoning_model = reasoning_model or self.reasoning_model
        next_task = task_name or self.task_name
        if next_task not in self.task_names and self.task_names:
            next_task = self.task_names[0]

        return InferenceSettings(
            api_base_url=api_base_url or self.api_base_url,
            model_name=next_model,
            reasoning_model=next_reasoning_model,
            api_key=self.api_key,
            env_base_url=env_base_url if env_base_url is not None else self.env_base_url,
            local_image_name=local_image_name or self.local_image_name,
            benchmark=self.benchmark,
            success_score_threshold=(
                success_score_threshold
                if success_score_threshold is not None
                else self.success_score_threshold
            ),
            max_steps=max_steps if max_steps is not None else self.max_steps,
            llm_max_retries=self.llm_max_retries,
            force_heuristic=(
                force_heuristic if force_heuristic is not None else self.force_heuristic
            ),
            task_name=next_task,
            task_names=self.task_names,
        )
