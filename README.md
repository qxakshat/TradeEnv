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
- `urgency`: continuous value in `[0, 1]` (patient → aggressive execution)

### Observation (`TradeEnvObservation`)

Includes:

- task metadata: `task_name`, `difficulty`, `role`
- market: `symbol`, `current_price`, `best_bid`, `best_ask`, `spread_bps`, `bid_depth_top`, `ask_depth_top`, `depth_imbalance`, `estimated_slippage_bps`, `ema_fast`, `ema_slow`, `rsi_14`
- fundamentals: `pe_ratio`, `price_to_book`, `return_on_equity`, `debt_to_equity`, `profit_margin`, `revenue_growth`, `fundamental_quality_score`
- news context: `news_sentiment_score`, `news_sentiment_confidence`
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

3. **hard_depth_aware_execution (hard)**
   - Objective: buy volatile names under thin market depth with strict action budget and low slippage
   - Symbols: higher-volatility names
   - Max trades: 6

### Deterministic scoring (0.0–1.0)

Graders score each task with reproducible formulas using:

- execution quality vs day low/high
- fill completion ratio
- slippage metric (bps-normalized)
- trade efficiency (fewer trades, complete fill)

All component metrics and final score are clipped to `[0.0, 1.0]`.

Scores are deterministic for the same episode data and actions.

## Trading Roles (5 distinct strategies)

Each episode cycles through a **trading role** that determines constraints, reward weights, and grading criteria:

### 1. **Aggressive Buyer**
   - Strategy: Fast completion, accepts slippage for speed
   - Slippage factor: 1.3× (higher cost for faster execution)
   - Graded on: Speed (early execution finish) + quality of fills
   - Reward weights: Speed bonus 30%, Quality 40%

### 2. **Conservative Seller**
   - Strategy: Patient, high-quality execution; waits for best prices
   - Slippage factor: 0.7× (tight pricing)
   - Graded on: Precision (target qty accuracy) + quality
   - Reward weights: Quality 60%, Consistency bonus 8%

### 3. **Market Maker**
   - Strategy: Balanced timing; captures both sides profitably
   - Slippage factor: 0.9× (near-market)
   - Graded on: Balanced fills + quantity completion
   - Reward weights: Quality 35%, Fill balance 15%, Consistency 15%

### 4. **Arbitrageur**
   - Strategy: Exploits mispricings; highly sensitive to costs
   - Slippage factor: 0.5× (minimal slippage tolerance)
   - Graded on: Extreme precision; any overspend penalized
   - Reward weights: Quality 70% (must be excellent)

### 5. **Volatility Trader**
   - Strategy: Trades price swings; exploits technical signals
   - Slippage factor: 0.8× (balanced execution costs)
   - Graded on: Timing (when fills happen) + quality
   - Reward weights: Quality 50%, Timing 20%, Technical sophistication 12%

**Role Cycling**: Each `reset()` automatically cycles to the next role. Agents must adapt strategy to the current role's constraints and reward structure.


## Reward Function (Dense & Meaningful)

Per-step reward has eight components:

- `immediate_edge`: fill quality vs running VWAP benchmark
- `progress_bonus`: reward for advancing toward target quantity (role-adjusted)
- `slippage_penalty`: penalizes wide spread + high realized slippage pressure
- `trade_efficiency_bonus`: rewards completion with fewer trades
- `depth_timing_bonus`: rewards execution when order-book imbalance is favorable
- `random_trade_penalty`: penalizes tiny/noisy flip-flop trades that mimic random behavior
- `variance_penalty`: trajectory-level penalty for high execution-price variance
- `constraint_penalty`: invalid side, excessive trading, late inactivity
- `terminal_adjustment`: end-of-episode completion/quality adjustment
- `role_adjustment`: role-specific terminal bonus for consistency, precision, or timing

This gives trajectory-level signal (not just terminal binary reward) and role-specific incentives.

## Data Source

- Pulls last 5 months from `yfinance` for a NIFTY-50 subset (`*.NS`)
- Builds deterministic synthetic intraday path per day from OHLC anchors to simulate execution slices

## Simulated Market Depth (Novelty)

Each step includes deterministic level-1 depth state:

- `best_bid`, `best_ask`, `spread_bps`
- `bid_depth_top`, `ask_depth_top`
- `depth_imbalance` in `[-1, 1]`

Hard tasks intentionally run thinner liquidity and wider spreads, making frontier models work harder on timing, participation rate, and slippage control.

## Tiny News Sentiment Signal (Differentiator)

To differentiate from `openenv-finrl`-style pure price/technical setups, TradeEnv now includes a lightweight sentiment channel:

- Sources: Yahoo Finance RSS + Moneycontrol business RSS
- Model: tiny lexical scorer (`server/news_sentiment.py`) acting as a small-LM-style baseline
- Output in observation: `news_sentiment_score`, `news_sentiment_confidence`
- Also returned in `metadata.info.news_sentiment` with sampled headlines

Design note: sentiment is **advisory** by default and does not break deterministic grading formulas.

## Web Fundamentals Signal (Differentiator)

TradeEnv now augments each episode with website-sourced fundamentals per symbol:

- Yahoo key statistics page
- Yahoo quoteSummary endpoint (`defaultKeyStatistics`, `financialData`, `summaryDetail`)
- Moneycontrol site link in source attribution

The environment caches a weekly `data/fundamentals_snapshot.csv` and exposes values + source URLs via:

- observation fields (`pe_ratio`, `price_to_book`, `return_on_equity`, `debt_to_equity`, `profit_margin`, `revenue_growth`, `fundamental_quality_score`)
- metadata (`metadata.info.fundamentals.sources`)
- app endpoint `GET /dashboard-snapshot`

### How this differs from openenv-finrl

- **Depth-aware execution microstructure** (spread/depth/imbalance/slippage state)
- **Web fundamentals channel** (valuation + balance-sheet quality features from finance websites)
- **Role-conditional grading** (5 trading roles with different reward pressure)
- **Anti-randomness reward shaping** (micro-trade and flip-flop penalties)
- **Variance-aware execution objective** (terminal variance penalty)
- **News-aware observation channel** (tiny sentiment signal for policy adaptation)

## Broker Integration Placeholders

The environment ships with safe stubs for future broker integration:

- `server/broker_adapters.py`
   - `ZerodhaAdapter` (placeholder)
   - `PaytmMoneyAdapter` (placeholder)

No live brokerage calls are performed in hackathon mode; integration points are exposed via `metadata.info.broker_integration`.

## Baseline Inference Script

`inference.py`:

- Uses **OpenAI Python client**
- Reads credentials from `OPENAI_API_KEY`
- Runs all 3 tasks + 5 roles in deterministic order
- Default policy mode is deterministic `heuristic` for reproducible grading runs
- Optional mode: `POLICY_MODE=openai` for LLM-driven policy
- Emits `[START]`, `[STEP]`, `[END]` logs per episode
- Reports reproducible 0–1 score for each (task, role) combination and mean score

## Multi-Role Architecture

The environment enforces **model versatility**: by cycling through 5 distinct trading roles, agents must learn adaptive strategies:

- **No single optimal policy**: aggressive buyer strategy fails against conservative seller constraints
- **Emergent role-awareness**: superior agents develop role-specific execution plans
- **Meaningful complexity**: naive agents may score well on easy buys but fail on arbitrage-style precision

This mirrors real trading desks where portfolio managers shift strategies based on market regime, position type, and risk appetite.

## Setup

```bash
uv sync
```

## Run locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# New powerful dashboard UI (HF-style)
open http://127.0.0.1:8000/dashboard

# Optional custom UI metadata endpoint
curl http://127.0.0.1:8000/ui-config

# Optional frontend help page payload
curl http://127.0.0.1:8000/frontend-readme

# Optional dashboard sample with source-backed fundamentals values
curl http://127.0.0.1:8000/dashboard-snapshot

# Live market APIs used by dashboard
curl http://127.0.0.1:8000/api/market/overview?days=30
curl http://127.0.0.1:8000/api/market/history/RELIANCE.NS?days=30
curl http://127.0.0.1:8000/api/portfolio/demo
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

It now includes fundamentals-augmented feature engineering (12-dim input with valuation/profitability/growth signals) so training reflects both microstructure and company quality.

## Project structure

- `models.py`: typed Action/Observation/Reward models
- `server/tradeenv_environment.py`: environment, reward shaping, tasks, graders
- `server/broker_adapters.py`: placeholder broker connectors (Zerodha, Paytm Money)
- `client.py`: typed OpenEnv client
- `inference.py`: OpenAI baseline over 3 tasks
- `openenv.yaml`: environment manifest
- `Dockerfile`: root container image for local/HF deployment
- `server/Dockerfile`: mirrored build file kept for OpenEnv-oriented workflows
