"""Baseline inference script for TradeEnv OpenEnv benchmark.

Runs three deterministic tasks (easy/medium/hard) and prints reproducible
[START]/[STEP]/[END] logs for each episode.
"""

from __future__ import annotations

import os
import textwrap
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from client import TradeEnvClient
from models import TradeEnvAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
IMAGE_NAME = os.getenv("IMAGE_NAME", "tradeenv:latest")
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


def run_episode(env: TradeEnvClient, client: OpenAI) -> Dict:
    result = env.reset()
    obs = result.observation
    task_name = obs.task_name
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps = 0
    final_score = 0.0

    for step in range(1, MAX_STEPS + 1):
        if result.done:
            break
        action_type, quantity, action_src = choose_action_with_model(client, obs)
        action_txt = f"{action_type}:{quantity}"

        result = env.step(TradeEnvAction(action_type=action_type, quantity=quantity))
        obs = result.observation
        reward = float(result.reward or 0.0)

        rewards.append(reward)
        steps = step

        info = (obs.metadata or {}).get("info", {})
        score_meta = info.get("task_score")
        if score_meta:
            final_score = float(score_meta.get("score", 0.0))

        err = None if not action_src.startswith("fallback:") else action_src
        log_step(step=step, action=action_txt, reward=reward, done=result.done, error=err)

        if result.done:
            break

    success = final_score >= 0.5
    log_end(success=success, steps=steps, score=final_score, rewards=rewards)
    return {"task": task_name, "score": final_score, "steps": steps, "rewards": rewards}


def main() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for baseline inference.")

    client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)

    # reset() cycles deterministic tasks: easy -> medium -> hard
    env = TradeEnvClient.from_docker_image(IMAGE_NAME)

    all_results = []
    try:
        for _ in range(3):
            all_results.append(run_episode(env, client))
    finally:
        env.close()

    mean_score = sum(r["score"] for r in all_results) / max(len(all_results), 1)
    print(f"\nMean score across 3 tasks: {mean_score:.3f}", flush=True)


if __name__ == "__main__":
    main()
