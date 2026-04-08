# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the TradeEnv Environment.

This module creates an HTTP server that exposes the TradeEnvEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Multi-Role Architecture:
    Each reset() automatically cycles through 5 distinct trading roles:
    - aggressive_buyer: Fast execution, accepts slippage
    - conservative_seller: High-quality, patient execution
    - market_maker: Balanced timing, profit from volume
    - arbitrageur: Misprice exploitation, tight costs
    - volatility_trader: Swing trading with technical sophistication

    This enforces model versatility: agents must adapt strategy to role constraints.

Primary Endpoints:
    - POST /reset: Reset the environment (cycles task + role)
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Additional UX Endpoints:
    - GET /healthz: Container health check
    - GET /ui-config: Frontend hints for charts and leaderboard display

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import TradeEnvAction, TradeEnvObservation
    from .tradeenv_environment import TradeEnvEnvironment
except ImportError:
    from models import TradeEnvAction, TradeEnvObservation
    from server.tradeenv_environment import TradeEnvEnvironment


# Create the app with web interface and README integration
app = create_app(
    TradeEnvEnvironment,
    TradeEnvAction,
    TradeEnvObservation,
    env_name="TradeEnv",
    max_concurrent_envs=4,
)


@app.get("/healthz")
def healthz() -> dict:
    """Simple readiness endpoint for HF Spaces and container probes."""
    return {"status": "ok", "env": "TradeEnv"}


@app.get("/ui-config")
def ui_config() -> dict:
    """UI metadata to enrich dashboard rendering in custom frontends."""
    return {
        "title": "TradeEnv — Depth-Aware Execution Arena",
        "subtitle": "Easy → Medium → Hard tasks with role-conditional grading",
        "cards": [
            {"key": "task_score", "label": "Task Score", "range": [0.0, 1.0]},
            {"key": "slippage_bps", "label": "Estimated Slippage (bps)", "range": [0, 500]},
            {"key": "trade_efficiency", "label": "Trade Efficiency", "range": [0.0, 1.0]},
        ],
        "charts": [
            {"id": "reward_breakdown", "type": "stacked_bar"},
            {"id": "spread_and_depth", "type": "line"},
            {"id": "score_by_role", "type": "bar"},
            {"id": "sentiment_vs_reward", "type": "scatter"},
        ],
        "controls": [
            {"id": "action_type", "label": "Action", "type": "select", "options": ["buy", "sell", "hold"]},
            {"id": "quantity", "label": "Quantity", "type": "slider", "min": 0, "max": 1000},
            {"id": "urgency", "label": "Urgency", "type": "slider", "min": 0.0, "max": 1.0, "step": 0.05},
        ],
        "leaderboard": {"group_by": ["task_name", "role"], "metric": "score"},
    }


@app.get("/frontend-readme")
def frontend_readme() -> dict:
    """Frontend guidance for human-friendly interaction."""
    markdown = """
# TradeEnv Frontend Guide

## Quick Start
1. Click **Reset** to start a new episode (task + role cycle).
2. Pick **Action** (`buy`/`sell`/`hold`), set **Quantity**, and optionally adjust **Urgency**.
3. Watch score + reward breakdown in charts after each step.

## What to watch
- **Task Score (0-1)**: end-goal quality metric
- **Spread / Depth**: avoid trading large size in wide-spread thin-depth moments
- **Sentiment Signal**: optional advisory signal from Yahoo/Moneycontrol headlines

## Human Tips
- Use fewer, better-timed trades for higher efficiency.
- On hard tasks, prioritize low slippage over raw speed.
- Avoid tiny random trades; they are explicitly penalized.
""".strip()
    return {"title": "TradeEnv Frontend README", "markdown": markdown}


def main() -> None:
    """Entry-point used by OpenEnv validators and local execution."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
