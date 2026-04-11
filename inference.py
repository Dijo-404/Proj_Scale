# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""CLI entrypoint for running Proj_Scale inference experiments."""

from __future__ import annotations

import argparse
import asyncio
import os

from inference_config import InferenceSettings
from inference_runner import run_inference


def _build_parser(settings: InferenceSettings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Proj_Scale inference in heuristic or LLM mode.",
    )
    parser.add_argument("--task", help="Task name to run")
    parser.add_argument(
        "--api-base-url",
        help="Base URL for the chat-completions-compatible LLM API",
    )
    parser.add_argument("--model", help="Primary model name for per-step action calls")
    parser.add_argument(
        "--reasoning-model",
        help="Model used for initial episode planning (defaults to --model)",
    )
    parser.add_argument("--env-base-url", help="Connect to an already running OpenEnv server")
    parser.add_argument(
        "--image-name",
        help="Docker image name for local container execution when --env-base-url is omitted",
    )
    parser.add_argument("--max-steps", type=int, help="Maximum number of environment steps")
    parser.add_argument(
        "--success-threshold",
        type=float,
        help="Minimum score required to consider the run successful",
    )
    parser.add_argument(
        "--force-heuristic",
        action="store_true",
        help="Disable LLM calls and use deterministic baseline triage rules only",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List known tasks and exit",
    )
    parser.set_defaults(force_heuristic=settings.force_heuristic)
    return parser


def _resolve_settings(base: InferenceSettings, args: argparse.Namespace) -> InferenceSettings:
    reasoning_model = args.reasoning_model
    if args.model and not reasoning_model:
        reasoning_model = args.model

    return base.with_overrides(
        task_name=args.task,
        api_base_url=args.api_base_url,
        model_name=args.model,
        reasoning_model=reasoning_model,
        env_base_url=args.env_base_url,
        local_image_name=args.image_name,
        max_steps=args.max_steps,
        success_score_threshold=args.success_threshold,
        force_heuristic=args.force_heuristic,
    )


def _read_required_env() -> tuple[str, str, str]:
    """Read required runtime env vars for hackathon submission compatibility."""

    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")

    return api_base_url, model_name, hf_token


async def _async_main() -> int:
    api_base_url, model_name, hf_token = _read_required_env()
    base_settings = InferenceSettings.from_env(
        api_base_url=api_base_url,
        model_name=model_name,
        hf_token=hf_token,
    )
    parser = _build_parser(base_settings)
    args = parser.parse_args()

    if args.list_tasks:
        for task_name in base_settings.task_names:
            print(task_name)
        return 0

    settings = _resolve_settings(base_settings, args)
    return await run_inference(settings)


def main() -> int:
    return asyncio.run(_async_main())


if __name__ == "__main__":
    raise SystemExit(main())
