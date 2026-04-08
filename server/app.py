"""FastAPI app for support operations triage environment."""

from __future__ import annotations

from fastapi import HTTPException

from openenv.core.env_server.http_server import create_app

try:
    from ..models import SupportOpsAction, SupportOpsObservation
    from ..tasks import TASK_LIBRARY
    from .support_ops_environment import SupportOpsEnvironment
except ImportError:
    from models import SupportOpsAction, SupportOpsObservation
    from tasks import TASK_LIBRARY
    from server.support_ops_environment import SupportOpsEnvironment


app = create_app(
    SupportOpsEnvironment,
    SupportOpsAction,
    SupportOpsObservation,
    env_name="Proj_Scale",
    max_concurrent_envs=4,
)


@app.get("/tasks", tags=["Environment Info"])
def list_tasks() -> dict:
    """List available benchmark tasks and their difficulty."""
    return {
        "tasks": [
            {
                "name": task.name,
                "difficulty": task.difficulty,
                "description": task.description,
                "max_steps": task.max_steps,
            }
            for task in TASK_LIBRARY.values()
        ]
    }


@app.get("/tasks/{task_name}", tags=["Environment Info"])
def get_task(task_name: str) -> dict:
    """Return details for a specific task, including target outcomes."""
    task = TASK_LIBRARY.get(task_name)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_name}")

    return {
        "name": task.name,
        "difficulty": task.difficulty,
        "description": task.description,
        "max_steps": task.max_steps,
        "tickets": [seed.__dict__ for seed in task.tickets],
        "goals": {
            ticket_id: {
                "priority": goal.priority,
                "category": goal.category,
                "team": goal.team,
                "status": goal.status,
                "reply_keywords": list(goal.reply_rule.required_keywords),
            }
            for ticket_id, goal in task.goals.items()
        },
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the OpenEnv server locally."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
