---
title: TradeEnv OpenEnv Environment
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# TradeEnv: Institutional-Style Execution Environment

`TradeEnv` simulates a real execution desk task: buy/sell a large order over one trading day while minimizing market impact and maximizing fill quality.

This is a **real-world utility** benchmark for agentic decision-making in finance execution workflows.

## Motivation

Human traders split large orders over time while reacting to market signals (price, EMA, RSI, spread regimes). This environment evaluates whether an agent can:

- Minimize buy execution price
- Maximize sell execution price
- Respect hard constraints: at most `k <= 10` trades/day
- Complete inventory under time pressure

## OpenEnv Interface Compliance

The environment implements typed Pydantic models and OpenEnv primitives:

- `TradeEnvAction` (typed action)
- `TradeEnvObservation` (typed observation)
- `TradeEnvReward` (typed reward decomposition)
- `step(action)` returns transition semantics `(observation, reward, done, info)` internally, and returns `Observation` object with OpenEnv-compatible `reward/done/metadata` fields
- `reset()` returns initial observation
- `state` endpoint provides current environment state (`episode_id`, `step_count`)

Manifest: `openenv.yaml`

Validation:

- `openenv validate`
- Verified locally on 2026-04-08: `openenv validate` returned **Ready for multi-mode deployment**

## Observation and Action Space

### Action (`TradeEnvAction`)

- `action_type`: `buy | sell | hold`
- `quantity`: integer shares for current step

### Observation (`TradeEnvObservation`)

Includes:

- task metadata: `task_name`, `difficulty`
- market: `symbol`, `current_price`, `ema_fast`, `ema_slow`, `rsi_14`
- execution state: `target_side`, `target_quantity`, `remaining_quantity`, `executed_quantity`, `avg_execution_price`
- constraints: `trades_used`, `max_trades_per_day`
- OpenEnv fields: `reward`, `done`, `metadata`

## Tasks (3 levels + deterministic graders)

1. **easy_buy_quality (easy)**
   - Objective: buy near daily lows with complete fill
   - Symbols: stable liquid names
   - Max trades: 10

2. **medium_sell_quality (medium)**
   - Objective: sell near daily highs while completing inventory
   - Symbols: mixed regimes
   - Max trades: 8

3. **hard_volatile_execution (hard)**
   - Objective: buy volatile names with strict action budget and low slippage
   - Symbols: higher-volatility names
   - Max trades: 6

### Deterministic scoring (0.0–1.0)

Graders score each task with reproducible formulas using:

- execution quality vs day low/high
- fill completion ratio
- trade efficiency (hard task)

Scores are deterministic for the same episode data and actions.

## Reward Function (Dense & Meaningful)

Per-step reward has four components:

- `immediate_edge`: fill quality vs running VWAP benchmark
- `progress_bonus`: reward for advancing toward target quantity
- `constraint_penalty`: invalid side, excessive trading, late inactivity
- `terminal_adjustment`: end-of-episode completion/quality adjustment

This gives trajectory-level signal (not just terminal binary reward).

## Data Source

- Pulls last 5 months from `yfinance` for a NIFTY-50 subset (`*.NS`)
- Builds deterministic synthetic intraday path per day from OHLC anchors to simulate execution slices

## Baseline Inference Script

`inference.py`:

- Uses **OpenAI Python client**
- Reads credentials from `OPENAI_API_KEY`
- Runs all 3 tasks in deterministic order (easy → medium → hard)
- Emits `[START]`, `[STEP]`, `[END]` logs per episode
- Reports reproducible 0–1 score for each task and mean score

## Setup

```bash
uv sync
```

## Run locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

## Validate OpenEnv spec

```bash
openenv validate
```

## Build & run container

```bash
docker build -t tradeenv:latest .
docker run --rm -p 8000:8000 tradeenv:latest
```

## Hugging Face Space deployment

```bash
openenv push
```

The repo is Docker-first and tagged for OpenEnv Spaces (`tags: [openenv]`).

## Baseline run

```bash
export OPENAI_API_KEY=<your_key>
uv run python inference.py
```

If the OpenAI API is unavailable, the script falls back to a deterministic safe policy so the benchmark still completes end-to-end.

The notebook `notebooks/hackathon_project.ipynb` also contains a saved DL-policy vs random-baseline comparison for the hackathon submission workflow.

## Project structure

- `models.py`: typed Action/Observation/Reward models
- `server/tradeenv_environment.py`: environment, reward shaping, tasks, graders
- `client.py`: typed OpenEnv client
- `inference.py`: OpenAI baseline over 3 tasks
- `openenv.yaml`: environment manifest
- `Dockerfile`: root container image for local/HF deployment
- `server/Dockerfile`: mirrored build file kept for OpenEnv-oriented workflows
