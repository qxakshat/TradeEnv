---
title: TradeEnv OpenEnv Environment
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /dashboard
tags:
  - openenv
---

# TradeEnv: Institutional Execution Arena for Agent Evaluation

`TradeEnv` is an OpenEnv benchmark for a real desk workflow: execute a large buy/sell order over a trading day while balancing fill completion, market impact, slippage, and action-budget constraints.

This benchmark is built for **agent evaluation**, not just backtesting. It explicitly tests adaptation across task difficulty and trading-role objectives.

## Why judges should care (rubric-aligned)

| Rubric category | How TradeEnv addresses it |
|---|---|
| **Real-world utility (30%)** | Models execution quality under spread/depth pressure, action budgets, and completion deadlines. This mirrors buy-side/sell-side execution desks and paper-trading assistants. |
| **Task & grader quality (25%)** | 3 tasks (`easy`, `medium`, `hard`) + 5 role variants. Deterministic, clipped scoring in `[0,1]` with transparent components (`quality`, `fill`, `slippage`, `efficiency`, `role_metric`). |
| **Environment design (20%)** | `reset()` creates clean state and new episode id; typed action/observation contracts; dense reward decomposition with stepwise signal + sensible terminal logic. |
| **Code quality & spec compliance (15%)** | OpenEnv manifest in `openenv.yaml`; Docker-first runtime; FastAPI app + health endpoint; reproducible baseline runner in `inference.py`. |
| **Creativity & novelty (10%)** | Combines execution microstructure + role-conditioned grading + fundamentals + lightweight sentiment context, which is uncommon in OpenEnv finance environments. |

---

## 1) Real-world utility

Institutional execution is a practical RL/agent problem with clear operational constraints:

- **Primary objective**: buy near lower prices / sell near higher prices while completing inventory.
- **Operational constraints**: max trades per episode, finite time horizon, side-valid actions only.
- **Microstructure pressure**: spread, top-of-book depth, and depth imbalance influence realized slippage.
- **Execution trade-off**: speed vs quality (fast fills often cost more).

TradeEnv is useful for evaluating whether an agent can make **sequential, constrained decisions** under realistic market frictions.

## 2) Task and grader quality

### Tasks (difficulty ladder)

1. **`easy_buy_quality`**
   - Side: buy
   - Max trades: 10
   - Objective: complete while staying closer to day lows

2. **`medium_sell_quality`**
   - Side: sell
   - Max trades: 8
   - Objective: complete while staying closer to day highs

3. **`hard_depth_aware_execution`**
   - Side: buy
   - Max trades: 6
   - Objective: volatile symbols, thinner liquidity, stricter budget, slippage control

### Deterministic scoring (strictly 0.0–1.0)

All core metrics are clipped to `[0,1]`:

- `quality_metric`: execution quality inside daily range
- `fill_metric`: completion ratio
- `slippage_metric`: slippage-normalized quality
- `efficiency_metric`: preference for fewer trades
- `role_metric`: role-specific execution fit

Task score is deterministic for fixed episode trajectory/actions:

- Easy: `0.70*quality + 0.20*fill + 0.10*slippage`
- Medium: `0.55*quality + 0.25*fill + 0.15*slippage + 0.05*efficiency`
- Hard: `0.35*quality + 0.25*fill + 0.25*slippage + 0.15*efficiency`

### Why hard is genuinely hard

Hard mode compounds difficulty through:

- lower liquidity multiplier,
- wider spreads,
- fewer allowed actions,
- larger target size,
- role-conditioned reward pressure.

This punishes naive fixed-rate policies and rewards adaptive timing/participation control.

## 3) Environment design quality

### Clean reset semantics

`reset()`:

- creates a fresh `episode_id`,
- resets counters (`step_count`, fill/cost trackers, trade count),
- rotates task and role,
- rebuilds per-episode market/depth trajectory,
- returns initial typed observation.

### Action and observation contracts

- **Action** (`TradeEnvAction`):
  - `action_type`: `buy | sell | hold`
  - `quantity`: integer `0..10000`
  - `urgency`: float `[0,1]`

- **Observation** (`TradeEnvObservation`) includes:
  - task/role (`task_name`, `difficulty`, `role`)
  - market microstructure (`best_bid`, `best_ask`, `spread_bps`, `bid_depth_top`, `ask_depth_top`, `depth_imbalance`)
  - technical context (`ema_fast`, `ema_slow`, `rsi_14`)
  - fundamentals (`pe_ratio`, `price_to_book`, `return_on_equity`, `debt_to_equity`, `profit_margin`, `revenue_growth`, `fundamental_quality_score`)
  - sentiment (`news_sentiment_score`, `news_sentiment_confidence`)
  - execution state (`remaining_quantity`, `executed_quantity`, `trades_used`, etc.)

### Reward signal design (dense, non-sparse)

Per-step reward decomposes into:

- positive terms: `immediate_edge`, `progress_bonus`, `depth_timing_bonus`, `trade_efficiency_bonus`
- penalties: `slippage_penalty`, `random_trade_penalty`, `variance_penalty`, `constraint_penalty`
- terminal shaping: `terminal_adjustment`, `role_adjustment`

This yields informative gradients across the episode instead of only terminal success/fail.

### Episode boundaries

An episode ends when:

- target quantity is fully executed, **or**
- max trajectory length is reached.

Unfilled quantity at terminal step is force-closed with explicit penalty, preventing degenerate “do nothing” policies.

## 4) Code quality and OpenEnv compliance

### OpenEnv interface

- Manifest: `openenv.yaml`
- Runtime: FastAPI (`server.app:app`)
- Typed models: `models.py`
- Environment implementation: `server/tradeenv_environment.py`

### Validation and reproducibility protocol

Recommended checks for judges:

1. `openenv validate`
2. `docker build && docker run`
3. `GET /healthz` responds
4. Baseline run via `inference.py` (deterministic heuristic mode)

`inference.py` is deterministic by default (`POLICY_MODE=heuristic`) and prints structured logs (`[START]`, `[STEP]`, `[END]`) plus per-episode score and mean score.

### HF Space readiness

- Docker SDK frontmatter and app port are declared in this README header.
- Root route redirects to `/dashboard`.
- Health and dashboard APIs are available for smoke tests.

## 5) Creativity and novelty

TradeEnv is not a plain price-prediction environment. It introduces:

1. **Depth-aware execution microstructure** (spread/depth/imbalance-driven frictions)
2. **Role-conditional evaluation** (5 role archetypes with different objective pressure)
3. **Anti-randomness shaping** (tiny-trade and flip-flop penalties)
4. **Trajectory stability objective** (variance penalty on fill prices)
5. **Context channels beyond price** (fundamentals + advisory news sentiment)

This combination creates a richer policy-learning target than many standard finance benchmarks.

---

## Trading roles (policy adaptation challenge)

Every `reset()` rotates roles, forcing strategy adaptation:

- `aggressive_buyer`
- `conservative_seller`
- `market_maker`
- `arbitrageur`
- `volatility_trader`

No single static policy dominates across all role/task pairs.

## Data and realism notes

- Daily OHLCV source: `yfinance` (NIFTY-50 subset)
- Intraday execution path: deterministic synthetic interpolation from daily anchors
- Depth state: deterministic simulation (proxy, not exchange L2 tape)
- Paper trading endpoints are virtual and non-broker-connected

This intentionally balances realism, safety, and reproducibility.

## Setup

```bash
uv sync
```

## Run locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
open http://127.0.0.1:8000/dashboard
```

## Validate OpenEnv spec

```bash
openenv validate
```

## Build and run container

```bash
docker build -t tradeenv:latest .
docker run --rm -p 8000:8000 tradeenv:latest
```

## Hugging Face Space deployment

```bash
openenv push
```

## Baseline run

```bash
export OPENAI_API_KEY=<your_key>
uv run python inference.py
```

If remote LLM is unavailable, baseline still runs in deterministic heuristic mode.

## Project structure

- `models.py` — typed action/observation/reward/score contracts
- `server/tradeenv_environment.py` — tasks, reward shaping, deterministic grader
- `server/app.py` — OpenEnv-compatible FastAPI server + dashboard APIs
- `inference.py` — reproducible baseline runner
- `openenv.yaml` — OpenEnv manifest
- `Dockerfile` — container entrypoint for local/HF deployment
