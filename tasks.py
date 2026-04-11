# Copyright (c) 2026 Proj_Scale contributors.
# SPDX-License-Identifier: MIT

"""Task specification loading for Proj_Scale benchmark scenarios."""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, cast

Difficulty = Literal["easy", "medium", "hard"]
CustomerTier = Literal["standard", "business", "enterprise"]

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_CUSTOMER_TIERS = {"standard", "business", "enterprise"}


class ScenarioConfigError(RuntimeError):
    """Raised when the benchmark scenario config cannot be loaded safely."""


@dataclass(frozen=True)
class ReplyRule:
    required_keywords: Tuple[str, ...]
    min_length: int = 80


@dataclass(frozen=True)
class TicketSeed:
    ticket_id: str
    subject: str
    customer_tier: CustomerTier
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


def _config_path() -> Path:
    configured = os.getenv("PROJ_SCALE_SCENARIO_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()

    repo_root = Path(__file__).resolve().parent
    preferred = repo_root / "config" / "scenario_config.json"
    legacy = repo_root / "scenario_config.json"
    if preferred.exists():
        return preferred.resolve()
    return legacy.resolve()


def _load_raw_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"scenario config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "scenario config is not valid JSON "
                f"(line {exc.lineno}, column {exc.colno})"
            ) from exc

    if not isinstance(payload, dict):
        raise ValueError("scenario config must be a JSON object")
    if not isinstance(payload.get("tasks"), list):
        raise ValueError("scenario config must include a 'tasks' list")

    return payload


def _parse_reply_rule(raw: Dict) -> ReplyRule:
    keywords = tuple(str(value).strip().lower() for value in raw.get("required_keywords", []))
    min_length = int(raw.get("min_length", 80))
    return ReplyRule(required_keywords=keywords, min_length=min_length)


def _parse_ticket_seed(raw: Dict) -> TicketSeed:
    customer_tier = str(raw["customer_tier"])
    if customer_tier not in VALID_CUSTOMER_TIERS:
        raise ValueError(f"Invalid customer_tier: {customer_tier}")

    return TicketSeed(
        ticket_id=str(raw["ticket_id"]),
        subject=str(raw["subject"]),
        customer_tier=cast(CustomerTier, customer_tier),
        sla_hours=int(raw["sla_hours"]),
    )


def _parse_ticket_goal(raw: Dict) -> TicketGoal:
    return TicketGoal(
        priority=str(raw["priority"]),
        category=str(raw["category"]),
        team=str(raw["team"]),
        status=str(raw["status"]),
        reply_rule=_parse_reply_rule(raw.get("reply_rule", {})),
    )


def _parse_process_rule(raw: Dict) -> ProcessRule:
    return ProcessRule(
        first_action_ticket=raw.get("first_action_ticket"),
        must_escalate=tuple(str(value) for value in raw.get("must_escalate", [])),
        must_resolve=tuple(str(value) for value in raw.get("must_resolve", [])),
    )


def _parse_task(raw: Dict) -> TaskSpec:
    difficulty = str(raw["difficulty"])
    if difficulty not in VALID_DIFFICULTIES:
        raise ValueError(f"Invalid difficulty: {difficulty}")

    tickets = tuple(_parse_ticket_seed(item) for item in raw.get("tickets", []))
    goals = {
        str(ticket_id): _parse_ticket_goal(goal)
        for ticket_id, goal in raw.get("goals", {}).items()
    }

    return TaskSpec(
        name=str(raw["name"]),
        difficulty=cast(Difficulty, difficulty),
        description=str(raw["description"]),
        max_steps=int(raw["max_steps"]),
        tickets=tickets,
        goals=goals,
        process_rule=_parse_process_rule(raw.get("process_rule", {})),
        action_hints=tuple(str(value) for value in raw.get("action_hints", [])),
    )


def _build_task_library(config_payload: Dict) -> Tuple[Dict[str, TaskSpec], Tuple[str, ...]]:
    specs = [_parse_task(raw_task) for raw_task in config_payload["tasks"]]
    library = {spec.name: spec for spec in specs}

    task_order = tuple(str(name) for name in config_payload.get("task_order", []))
    if task_order:
        invalid = [name for name in task_order if name not in library]
        if invalid:
            raise ValueError(f"task_order references unknown tasks: {invalid}")
    else:
        task_order = tuple(library.keys())

    return library, task_order


@functools.lru_cache(maxsize=1)
def _load_task_catalog() -> Tuple[Dict[str, TaskSpec], Tuple[str, ...], Path]:
    config_path = _config_path()
    try:
        config_payload = _load_raw_config(config_path)
        task_library, task_order = _build_task_library(config_payload)
    except (FileNotFoundError, ValueError, TypeError, KeyError) as exc:
        raise ScenarioConfigError(
            f"failed to load scenario config from '{config_path}': {exc}"
        ) from exc

    if not task_library:
        raise ScenarioConfigError(
            f"scenario config '{config_path}' must define at least one task"
        )

    if not task_order:
        raise ScenarioConfigError(
            f"scenario config '{config_path}' produced an empty task order"
        )

    return task_library, task_order, config_path


def clear_task_cache() -> None:
    """Clear cached task catalog data (used by tests and dynamic config changes)."""

    _load_task_catalog.cache_clear()


def get_task_library() -> Dict[str, TaskSpec]:
    return _load_task_catalog()[0]


def get_task_order() -> Tuple[str, ...]:
    return _load_task_catalog()[1]


def validate_scenario_config() -> Dict[str, object]:
    """Validate config eagerly and return a small diagnostics payload."""

    task_library, task_order, config_path = _load_task_catalog()
    return {
        "config_path": str(config_path),
        "task_count": len(task_library),
        "task_order_count": len(task_order),
    }


def has_task(task_name: str) -> bool:
    return task_name in get_task_library()


def __getattr__(name: str):
    """Backward-compatible lazy access for legacy module globals."""

    if name == "TASK_LIBRARY":
        return get_task_library()
    if name == "TASK_ORDER":
        return get_task_order()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_task(task_name: str) -> TaskSpec:
    task_library = get_task_library()
    if task_name not in task_library:
        raise KeyError(f"Unknown task: {task_name}")
    return task_library[task_name]


def available_tasks() -> Tuple[str, ...]:
    return get_task_order()
