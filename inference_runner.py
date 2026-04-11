# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Async runner for Proj_Scale inference executions."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from client import SupportOpsEnv
from inference_config import InferenceSettings
from inference_strategies import (
    build_baseline_plan,
    choose_action,
    merge_targets,
    plan_episode,
)
from models import SupportOpsAction


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    display_error = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={display_error}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # Task validation expects score strictly within (0, 1) after formatting.
    display_score = max(0.01, min(0.99, float(score)))
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={display_score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _action_to_str(action: SupportOpsAction) -> str:
    payload: Dict[str, Any] = {
        "command": action.command,
        "ticket_id": action.ticket_id,
        "value": action.value,
        "message": action.message,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _build_llm_client(settings: InferenceSettings) -> Optional[OpenAI]:
    if not settings.use_llm:
        return None

    if not settings.api_key:
        raise ValueError("HF_TOKEN environment variable is required")

    return OpenAI(base_url=settings.api_base_url, api_key=settings.api_key)


async def run_task(
    env: SupportOpsEnv,
    settings: InferenceSettings,
    llm_client: Optional[OpenAI],
) -> Tuple[bool, int, float, List[float]]:
    rewards: List[float] = []
    steps_taken = 0
    action_history: List[Dict[str, Any]] = []

    reset_result = await env.reset(task_name=settings.task_name)

    baseline_plan = build_baseline_plan(reset_result.observation)
    llm_plan = (
        plan_episode(llm_client, settings, reset_result.observation)
        if settings.use_llm and llm_client is not None
        else None
    )

    targets, replies, order = merge_targets(
        reset_result.observation,
        baseline_plan,
        llm_plan,
    )

    result = reset_result
    for step in range(1, settings.max_steps + 1):
        if result.done:
            break

        action = choose_action(
            result.observation,
            llm_client,
            settings,
            targets,
            replies,
            order,
            action_history,
        )
        action_history.append(
            {
                "step": step,
                "command": action.command,
                "ticket_id": action.ticket_id,
                "value": action.value,
            }
        )

        result = await env.step(action)
        reward = float(result.reward or 0.0)
        rewards.append(reward)
        steps_taken = step

        log_step(
            step=step,
            action=_action_to_str(action),
            reward=reward,
            done=bool(result.done),
            error=result.observation.last_action_error,
        )

        if result.done:
            break

    final_score = float(getattr(result.observation, "score", 0.0))
    final_score = max(0.0, min(1.0, final_score))
    success = final_score >= settings.success_score_threshold
    return success, steps_taken, final_score, rewards


async def run_inference(settings: InferenceSettings, run_all_tasks: bool = False) -> int:
    task_names = settings.task_names if run_all_tasks else (settings.task_name,)
    llm_client: Optional[OpenAI] = None

    try:
        llm_client = _build_llm_client(settings)
    except Exception as exc:
        print(f"[ERROR] Execution failed: {exc}", file=sys.stderr, flush=True)
        for task_name in task_names:
            log_start(task=task_name, env=settings.benchmark, model=settings.model_name)
            log_end(success=False, steps=0, score=0.01, rewards=[])
        return 1

    all_success = True

    for task_name in task_names:
        success = False
        steps = 0
        score = 0.01
        rewards: List[float] = []
        env: Optional[SupportOpsEnv] = None

        log_start(task=task_name, env=settings.benchmark, model=settings.model_name)

        try:
            if settings.env_base_url:
                env = SupportOpsEnv(base_url=settings.env_base_url)
                await env.connect()
            else:
                env = await SupportOpsEnv.from_docker_image(settings.local_image_name)

            task_settings = settings.with_overrides(task_name=task_name)
            success, steps, score, rewards = await run_task(env, task_settings, llm_client)
        except Exception as exc:
            print(f"[ERROR] Execution failed: {exc}", file=sys.stderr, flush=True)
            all_success = False
        else:
            all_success = all_success and success
        finally:
            if env is not None:
                try:
                    await env.close()
                except Exception as close_exc:
                    print(
                        f"[WARN] Failed to close environment cleanly: {close_exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            # Emit END after env.close to satisfy strict output-order contract.
            log_end(success=success, steps=steps, score=score, rewards=rewards)

    return 0 if all_success else 1
