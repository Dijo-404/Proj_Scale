---
title: Support Ops OpenEnv
emoji: 📮
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - customer-support
  - reinforcement-learning
---

## Support Ops OpenEnv

`support_ops_env` is a real-world OpenEnv environment that simulates customer support operations work: ticket triage, queue routing, escalation, and customer communication under SLA pressure.

It is designed for training and evaluating agents on realistic service-ops behavior, not toy games.

## Why This Environment

Modern support teams continuously balance:

- SLA deadlines and customer tier commitments.
- Correct routing across billing, security, SRE, and product teams.
- Clear customer communication with explicit timelines.
- Multi-ticket prioritization during incident load.

This environment models those tradeoffs with deterministic graders and dense rewards.

## OpenEnv API Compliance

This repo implements the OpenEnv server and client pattern with typed models and standard APIs:

- `reset(...)`
- `step(action)`
- `state`
- `openenv.yaml`
- FastAPI server app at `server.app:app`

## Action Space

`SupportOpsAction` fields:

- `command`: one of `set_priority`, `set_category`, `assign_team`, `set_status`, `reply`, `submit`.
- `ticket_id`: required for all commands except `submit`.
- `value`: payload for classification and routing commands.
- `message`: payload for `reply`.

## Observation Space

`SupportOpsObservation` includes:

- `task_name`, `difficulty`, `task_description`.
- `tickets`: ticket states (priority, category, team, status, last_reply).
- `score` in `[0.0, 1.0]`.
- `grader_breakdown`: routing, communication, process subtotals.
- `reward_details`: shaped reward decomposition.
- `remaining_steps`, `current_ticket`, `action_hints`, `last_action_error`.

## Tasks And Graders

1. `easy_access_recovery` (easy)

- Single account-access incident.
- Tests basic routing, practical customer reply, and resolution.

1. `medium_billing_dispute` (medium)

- Critical enterprise billing dispute plus routine invoice request.
- Tests prioritization, escalation, and parallel handling.

1. `hard_incident_swarm` (hard)

- Outage, security alert, and feature request in one episode.
- Tests incident-first sequencing and cross-team triage.

Deterministic grader functions:

- `grade_easy_access_recovery`
- `grade_medium_billing_dispute`
- `grade_hard_incident_swarm`

All graders return scores in `[0.0, 1.0]`.

## Reward Shaping

Per-step reward is:

`reward = (score_t - score_{t-1}) - step_penalty - invalid_action_penalty`

Where:

- `step_penalty = 0.01`
- `invalid_action_penalty = 0.05` for malformed or invalid actions.
- End-of-episode bonus or penalty is applied for completion quality and timeouts.

This gives dense partial-progress signals rather than sparse terminal-only rewards.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Run server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /metadata`
- `GET /tasks`

## Baseline Inference Script

Required baseline script:

- `inference.py`

It emits strict stdout lines in this order:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

Environment variables:

- `API_BASE_URL`: LLM endpoint.
- `MODEL_NAME`: model id.
- `HF_TOKEN` or `API_KEY`.
- `ENV_BASE_URL` (optional): connect to a running server.
- `LOCAL_IMAGE_NAME` (optional): default `support_ops_env-env:latest`.

Run baseline against local server:

```bash
ENV_BASE_URL=http://localhost:8000 python inference.py
```

Offline deterministic smoke test:

```bash
FORCE_HEURISTIC=1 ENV_BASE_URL=http://localhost:8000 python inference.py
```

## Docker

Build and run:

```bash
docker build -t support_ops_env-env:latest .
docker run --rm -p 8000:8000 support_ops_env-env:latest
```

## Validation

OpenEnv validator:

```bash
.venv/bin/openenv validate
```

Provided prevalidation helper:

```bash
bash preval_script.sh https://<your-space>.hf.space .
```

## Submission Notes

- Includes `openenv.yaml`.
- Includes typed action, observation, reward, and state models.
- Includes 3 graded tasks with easy, medium, hard progression.
- Includes `inference.py` in required location and format.
- Includes working Dockerfiles for Hugging Face Spaces deployment.
