# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""FastAPI application entrypoint for the Proj_Scale OpenEnv server."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from openenv.core.env_server.http_server import create_app

from graders import TASK_GRADERS
from models import SupportOpsAction, SupportOpsObservation
from tasks import ScenarioConfigError, get_task_library, validate_scenario_config
from server.support_ops_environment import SupportOpsEnvironment


logger = logging.getLogger(__name__)


app = create_app(
    SupportOpsEnvironment,
    SupportOpsAction,
    SupportOpsObservation,
    env_name="Proj_Scale",
    max_concurrent_envs=4,
)


def validate_config_on_startup() -> None:
    """Fail fast with clear diagnostics when scenario config is invalid."""
    try:
        details = validate_scenario_config()
    except ScenarioConfigError as exc:
        raise RuntimeError(f"Proj_Scale startup validation failed: {exc}") from exc

    logger.info(
        "Loaded scenario config at %s with %s tasks",
        details["config_path"],
        details["task_count"],
    )


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    validate_config_on_startup()
    yield


app.router.lifespan_context = _lifespan


@app.get("/", tags=["Environment Info"])
def root() -> dict:
    """Basic root endpoint for Space-level health checks and App tab rendering."""
    return {
        "status": "ok",
        "name": "Proj_Scale",
        "message": "Proj_Scale OpenEnv API is running",
    }


@app.get("/tasks", tags=["Environment Info"])
def list_tasks() -> dict:
    """List available benchmark tasks and their difficulty."""
    task_library = get_task_library()
    return {
        "tasks": [
            {
                "name": task.name,
                "difficulty": task.difficulty,
                "description": task.description,
                "max_steps": task.max_steps,
                "has_grader": task.name in TASK_GRADERS,
            }
            for task in task_library.values()
        ]
    }


@app.get("/tasks/{task_name}", tags=["Environment Info"])
def get_task(task_name: str) -> dict:
    """Return safe task details for a specific task.

    This endpoint intentionally excludes exact grading targets to avoid leaking
    an answer sheet through the API.
    """

    task = get_task_library().get(task_name)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_name}")

    return {
        "name": task.name,
        "difficulty": task.difficulty,
        "description": task.description,
        "max_steps": task.max_steps,
        "has_grader": task.name in TASK_GRADERS,
        "tickets": [seed.__dict__ for seed in task.tickets],
        "ticket_count": len(task.tickets),
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the OpenEnv server locally."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
