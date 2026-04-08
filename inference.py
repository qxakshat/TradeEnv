"""Baseline inference script for TradeEnv OpenEnv benchmark.

Runs deterministic tasks and roles with reproducible
[START]/[STEP]/[END] logs for each episode.
"""

from __future__ import annotations

import os
import textwrap
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

from models import TradeEnvAction
from server.tradeenv_environment import TradeEnvEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POLICY_MODE = os.getenv("POLICY_MODE", "heuristic")  # heuristic | openai
BENCHMARK = "TradeEnv"
MAX_STEPS = 16

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a trading execution agent.
    Goal: execute inventory for the specified side with <= max_trades_per_day,
    minimize buy price / maximize sell price using current price, EMA, RSI, and progress state.

    Output strict JSON only:
    {"action_type": "buy|sell|hold", "quantity": <int>}
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _safe_json_action(model_text: str, side: str, remaining: int) -> Tuple[str, int]:
    import json

    try:
        payload = json.loads(model_text)
        action_type = str(payload.get("action_type", "hold")).lower()
        quantity = int(payload.get("quantity", 0))
    except Exception:
        action_type, quantity = "hold", 0

    if action_type not in {"buy", "sell", "hold"}:
        action_type = "hold"
    if action_type != side:
        action_type = "hold"
    quantity = max(0, min(quantity, remaining))
    return action_type, quantity


def choose_action_heuristic(obs) -> Tuple[str, int, str]:
    """Deterministic, reproducible policy baseline."""
    remaining = max(int(obs.remaining_quantity), 0)
    steps_left = max(int(obs.max_steps - obs.step_index), 1)
    side = obs.target_side

    trades_left = max(int(obs.max_trades_per_day - obs.trades_used), 0)
    if trades_left <= 0:
        return "hold", 0, "heuristic:max_trades_guard"

    spread_guard = obs.spread_bps > 26.0
    unfavorable_depth = (side == "buy" and obs.depth_imbalance < -0.15) or (
        side == "sell" and obs.depth_imbalance > 0.15
    )

    slots_left = min(steps_left, max(trades_left, 1))
    base_clip = max(1, int(np.ceil(remaining / slots_left)))

    if spread_guard and unfavorable_depth and remaining > base_clip:
        return "hold", 0, "heuristic:hold_for_depth"

    urgency = 1.0 - (remaining / max(int(obs.target_quantity), 1))
    urgency = float(np.clip(urgency, 0.0, 1.0))
    role_boost = {
        "aggressive_buyer": 1.20,
        "conservative_seller": 0.85,
        "market_maker": 1.00,
        "arbitrageur": 0.80,
        "volatility_trader": 1.05,
    }.get(obs.role, 1.0)

    qty = int(np.clip(base_clip * role_boost * (1.0 + 0.35 * urgency), 1, remaining))
    return side, qty, "heuristic:depth_aware"


def choose_action_with_model(client: OpenAI, obs) -> Tuple[str, int, str]:
    user_prompt = textwrap.dedent(
        f"""
        task={obs.task_name}
        side={obs.target_side}
        price={obs.current_price:.4f}
        ema_fast={obs.ema_fast:.4f}
        ema_slow={obs.ema_slow:.4f}
        rsi_14={obs.rsi_14:.2f}
        remaining={obs.remaining_quantity}
        trades_used={obs.trades_used}
        max_trades={obs.max_trades_per_day}
        step={obs.step_index}/{obs.max_steps}
        role={obs.role}
        best_bid={obs.best_bid:.4f}
        best_ask={obs.best_ask:.4f}
        spread_bps={obs.spread_bps:.2f}
        depth_imbalance={obs.depth_imbalance:.4f}
        est_slippage_bps={obs.estimated_slippage_bps:.2f}
        pe_ratio={obs.pe_ratio:.3f}
        price_to_book={obs.price_to_book:.3f}
        roe={obs.return_on_equity:.3f}
        debt_to_equity={obs.debt_to_equity:.3f}
        profit_margin={obs.profit_margin:.3f}
        revenue_growth={obs.revenue_growth:.3f}
        fundamental_quality={obs.fundamental_quality_score:.3f}
        sentiment_score={obs.news_sentiment_score:.3f}
        sentiment_confidence={obs.news_sentiment_confidence:.3f}
        """
    ).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        text = (completion.choices[0].message.content or "{}").strip()
        action_type, quantity = _safe_json_action(text, obs.target_side, obs.remaining_quantity)
        return action_type, quantity, text
    except Exception as exc:
        fallback_qty = max(1, obs.remaining_quantity // max(obs.max_steps - obs.step_index, 1))
        return obs.target_side, fallback_qty, f"fallback:{exc}"


def _unwrap_step_result(result):
    """Support both EnvClient StepResult and local Environment Observation returns."""
    if hasattr(result, "observation"):
        return result.observation, bool(getattr(result, "done", False)), float(getattr(result, "reward", 0.0) or 0.0)
    # Local environment returns TradeEnvObservation directly
    obs = result
    return obs, bool(getattr(obs, "done", False)), float(getattr(obs, "reward", 0.0) or 0.0)


def run_episode(env, client: Optional[OpenAI]) -> Dict:
    result = env.reset()
    obs, done, _ = _unwrap_step_result(result)
    task_name = obs.task_name
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps = 0
    final_score = 0.0
    role_name = obs.role

    for step in range(1, MAX_STEPS + 1):
        if done:
            break
        if POLICY_MODE == "openai":
            if client is None:
                raise RuntimeError("POLICY_MODE=openai requires an initialized OpenAI client")
            action_type, quantity, action_src = choose_action_with_model(client, obs)
        else:
            action_type, quantity, action_src = choose_action_heuristic(obs)
        action_txt = f"{action_type}:{quantity}"

        result = env.step(TradeEnvAction(action_type=action_type, quantity=quantity, urgency=0.6))
        obs, done, reward = _unwrap_step_result(result)

        rewards.append(reward)
        steps = step

        info = (obs.metadata or {}).get("info", {})
        score_meta = info.get("task_score")
        if score_meta:
            final_score = float(score_meta.get("score", 0.0))

        err = None if not action_src.startswith("fallback:") else action_src
        log_step(step=step, action=action_txt, reward=reward, done=done, error=err)

        if done:
            break

    success = final_score >= 0.5
    log_end(success=success, steps=steps, score=final_score, rewards=rewards)
    return {
        "task": task_name,
        "role": role_name,
        "score": final_score,
        "steps": steps,
        "rewards": rewards,
    }


def main() -> None:
    client: OpenAI | None = None
    if POLICY_MODE == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required when POLICY_MODE=openai.")
        client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)

    # Deterministic offline baseline by default: no container/network dependency.
    env = TradeEnvEnvironment()

    all_results = []
    try:
        for _ in range(6):
            all_results.append(run_episode(env, client))
    finally:
        if hasattr(env, "close"):
            env.close()

    mean_score = sum(r["score"] for r in all_results) / max(len(all_results), 1)
    print(f"\nMean score across deterministic episodes: {mean_score:.3f}", flush=True)
    for r in all_results:
        print(f"  - task={r['task']} role={r['role']} score={r['score']:.3f} steps={r['steps']}", flush=True)


if __name__ == "__main__":
    main()
