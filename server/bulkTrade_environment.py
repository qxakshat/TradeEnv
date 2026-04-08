"""Bulk execution environment for OpenEnv hackathon submission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import yfinance as yf
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        BulktradeAction,
        BulktradeObservation,
        BulktradeReward,
        BulktradeTaskScore,
    )
except ImportError:
    from models import BulktradeAction, BulktradeObservation, BulktradeReward, BulktradeTaskScore


NIFTY50_SYMBOLS: List[str] = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS",
    "LT.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "AXISBANK.NS", "ASIANPAINT.NS", "BAJFINANCE.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS", "POWERGRID.NS",
]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    side: Literal["buy", "sell"]
    target_quantity: int
    max_trades: int
    symbols: List[str]
    description: str


TASKS: List[TaskSpec] = [
    TaskSpec(
        name="easy_buy_quality",
        difficulty="easy",
        side="buy",
        target_quantity=1_000,
        max_trades=10,
        symbols=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ITC.NS"],
        description="Buy inventory close to intraday lows with full completion.",
    ),
    TaskSpec(
        name="medium_sell_quality",
        difficulty="medium",
        side="sell",
        target_quantity=1_000,
        max_trades=8,
        symbols=["INFY.NS", "ICICIBANK.NS", "LT.NS", "SBIN.NS"],
        description="Sell inventory close to intraday highs with limited actions.",
    ),
    TaskSpec(
        name="hard_volatile_execution",
        difficulty="hard",
        side="buy",
        target_quantity=1_200,
        max_trades=6,
        symbols=["BAJFINANCE.NS", "MARUTI.NS", "WIPRO.NS", "TITAN.NS"],
        description="Buy on volatile names with strict trade budget and minimal slippage.",
    ),
]


class BulktradeEnvironment(Environment):
    """Real-world bulk execution simulator with deterministic graders."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._task_idx = -1
        self._task: TaskSpec = TASKS[0]
        self._market = self._download_market_data(NIFTY50_SYMBOLS)
        self._episodes = self._build_episode_table(self._market)

        self._price_path: List[float] = []
        self._step_idx = 0
        self._executed_qty = 0
        self._remaining_qty = 0
        self._cum_notional = 0.0
        self._trades_used = 0
        self._fills: List[Tuple[int, float]] = []
        self._last_reward = 0.0
        self._last_score = BulktradeTaskScore(
            task_name="",
            difficulty="easy",
            score=0.0,
            rationale="not_started",
        )
        self._day_low = 0.0
        self._day_high = 0.0
        self._day_vwap = 0.0
        self._ema_fast = 0.0
        self._ema_slow = 0.0
        self._rsi = 50.0
        self._symbol = ""
        self._date = ""

    def reset(self) -> BulktradeObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._task_idx = (self._task_idx + 1) % len(TASKS)
        self._task = TASKS[self._task_idx]

        task_rows = self._episodes[self._episodes["symbol"].isin(self._task.symbols)].reset_index(drop=True)
        ep_idx = (self._reset_count // len(TASKS)) % len(task_rows)
        row = task_rows.iloc[ep_idx]

        self._symbol = str(row["symbol"])
        self._date = str(row["date"])
        self._day_low = float(row["low"])
        self._day_high = float(row["high"])
        self._day_vwap = float(row["vwap"])
        self._ema_fast = float(row["ema_fast"])
        self._ema_slow = float(row["ema_slow"])
        self._rsi = float(row["rsi_14"])

        self._price_path = self._synthetic_intraday_path(
            open_price=float(row["open"]),
            low_price=self._day_low,
            high_price=self._day_high,
            close_price=float(row["close"]),
            n_steps=12,
        )

        self._step_idx = 0
        self._executed_qty = 0
        self._remaining_qty = self._task.target_quantity
        self._cum_notional = 0.0
        self._trades_used = 0
        self._fills = []
        self._last_reward = 0.0
        self._last_score = BulktradeTaskScore(
            task_name=self._task.name,
            difficulty=self._task.difficulty,
            score=0.0,
            rationale="in_progress",
        )

        return self._build_observation(done=False, reward=0.0, info={"phase": "reset", "date": self._date})

    def step(self, action: BulktradeAction) -> BulktradeObservation:  # type: ignore[override]
        observation, reward, done, info = self.step_transition(action)
        return self._build_observation(done=done, reward=reward, info=info)

    def step_transition(self, action: BulktradeAction) -> Tuple[BulktradeObservation, float, bool, Dict]:
        self._state.step_count += 1
        self._step_idx = min(self._step_idx + 1, len(self._price_path) - 1)
        current_price = float(self._price_path[self._step_idx])

        immediate_edge = 0.0
        progress_bonus = 0.0
        constraint_penalty = 0.0
        terminal_adjustment = 0.0

        requested_qty = int(max(action.quantity, 0))

        if self._trades_used >= self._task.max_trades and action.action_type in {"buy", "sell"} and requested_qty > 0:
            constraint_penalty -= 0.20
            executed_qty = 0
            fill_price = 0.0
            invalid_reason = "max_trades_exceeded"
        elif action.action_type != self._task.side and action.action_type != "hold" and requested_qty > 0:
            constraint_penalty -= 0.15
            executed_qty = 0
            fill_price = 0.0
            invalid_reason = "wrong_side"
        else:
            invalid_reason = "none"
            executed_qty = min(requested_qty, self._remaining_qty)
            fill_price = 0.0

            if action.action_type == "hold" or executed_qty == 0:
                if self._remaining_qty > 0 and self._step_idx > int(0.75 * (len(self._price_path) - 1)):
                    constraint_penalty -= 0.02
            else:
                self._trades_used += 1
                participation = executed_qty / max(self._task.target_quantity, 1)
                slippage = 0.001 + 0.01 * participation
                if self._task.side == "buy":
                    fill_price = current_price * (1.0 + slippage)
                    immediate_edge = (self._day_vwap - fill_price) / max(self._day_vwap, 1e-6)
                else:
                    fill_price = current_price * (1.0 - slippage)
                    immediate_edge = (fill_price - self._day_vwap) / max(self._day_vwap, 1e-6)

                progress_bonus = 0.08 * participation
                self._executed_qty += executed_qty
                self._remaining_qty -= executed_qty
                self._cum_notional += executed_qty * fill_price
                self._fills.append((executed_qty, fill_price))

        done = self._remaining_qty == 0 or self._step_idx >= len(self._price_path) - 1

        if done and self._remaining_qty > 0:
            forced_qty = self._remaining_qty
            forced_slippage = 0.02
            forced_price = current_price * (1.0 + forced_slippage if self._task.side == "buy" else 1.0 - forced_slippage)
            self._executed_qty += forced_qty
            self._cum_notional += forced_qty * forced_price
            self._fills.append((forced_qty, forced_price))
            self._remaining_qty = 0
            terminal_adjustment -= 0.20

        if done:
            self._last_score = self._grade_current_task()
            terminal_adjustment += (self._last_score.score - 0.5) * 0.2

        reward_obj = BulktradeReward(
            immediate_edge=immediate_edge,
            progress_bonus=progress_bonus,
            constraint_penalty=constraint_penalty,
            terminal_adjustment=terminal_adjustment,
            total=immediate_edge + progress_bonus + constraint_penalty + terminal_adjustment,
        )
        self._last_reward = reward_obj.total

        info = {
            "task": self._task.name,
            "date": self._date,
            "symbol": self._symbol,
            "invalid_reason": invalid_reason,
            "reward_breakdown": reward_obj.model_dump(),
            "task_score": self._last_score.model_dump() if done else None,
        }

        obs = self._build_observation(done=done, reward=reward_obj.total, info=info)
        return obs, reward_obj.total, done, info

    def _build_observation(self, done: bool, reward: float, info: Dict) -> BulktradeObservation:
        avg_px = self._cum_notional / self._executed_qty if self._executed_qty > 0 else 0.0
        return BulktradeObservation(
            task_name=self._task.name,
            difficulty=self._task.difficulty,
            symbol=self._symbol,
            step_index=self._step_idx,
            max_steps=len(self._price_path),
            current_price=float(self._price_path[self._step_idx]),
            ema_fast=self._ema_fast,
            ema_slow=self._ema_slow,
            rsi_14=self._rsi,
            target_side=self._task.side,
            target_quantity=self._task.target_quantity,
            remaining_quantity=self._remaining_qty,
            executed_quantity=self._executed_qty,
            avg_execution_price=float(avg_px),
            trades_used=self._trades_used,
            max_trades_per_day=self._task.max_trades,
            done=done,
            reward=reward,
            metadata={
                "task_description": self._task.description,
                "date": self._date,
                "info": info,
            },
        )

    def _grade_current_task(self) -> BulktradeTaskScore:
        if self._executed_qty <= 0:
            return BulktradeTaskScore(
                task_name=self._task.name,
                difficulty=self._task.difficulty,
                score=0.0,
                rationale="no_execution",
            )

        avg_price = self._cum_notional / self._executed_qty
        spread = max(self._day_high - self._day_low, 1e-6)
        fill_ratio = min(self._executed_qty / self._task.target_quantity, 1.0)

        if self._task.side == "buy":
            quality = (self._day_high - avg_price) / spread
        else:
            quality = (avg_price - self._day_low) / spread

        quality = float(np.clip(quality, 0.0, 1.0))
        efficiency = float(np.clip(1.0 - (self._trades_used / max(self._task.max_trades, 1)), 0.0, 1.0))

        if self._task.difficulty == "easy":
            score = quality
        elif self._task.difficulty == "medium":
            score = 0.80 * quality + 0.20 * fill_ratio
        else:
            score = 0.65 * quality + 0.20 * fill_ratio + 0.15 * efficiency

        return BulktradeTaskScore(
            task_name=self._task.name,
            difficulty=self._task.difficulty,
            score=float(np.clip(score, 0.0, 1.0)),
            rationale=f"quality={quality:.3f}, fill={fill_ratio:.3f}, efficiency={efficiency:.3f}",
        )

    @property
    def state(self) -> State:
        return self._state

    @staticmethod
    def _synthetic_intraday_path(
        open_price: float,
        low_price: float,
        high_price: float,
        close_price: float,
        n_steps: int,
    ) -> List[float]:
        anchors = np.array([open_price, low_price, high_price, close_price], dtype=np.float64)
        anchor_x = np.array([0.0, 0.35, 0.7, 1.0], dtype=np.float64)
        grid = np.linspace(0.0, 1.0, n_steps)
        path = np.interp(grid, anchor_x, anchors)
        return [float(x) for x in path]

    @staticmethod
    def _download_market_data(symbols: List[str]) -> pd.DataFrame:
        raw = yf.download(
            tickers=symbols,
            period="5mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )

        rows: List[pd.DataFrame] = []
        for sym in symbols:
            if isinstance(raw.columns, pd.MultiIndex):
                if sym not in raw.columns.get_level_values(0):
                    continue
                sdf = raw[sym].copy()
            else:
                sdf = raw.copy()

            if sdf.empty:
                continue
            sdf = sdf.rename(columns={c: c.lower() for c in sdf.columns})
            req = ["open", "high", "low", "close", "volume"]
            if not set(req).issubset(set(sdf.columns)):
                continue

            sdf = sdf[req].dropna().copy()
            sdf["symbol"] = sym
            rows.append(sdf)

        if not rows:
            raise RuntimeError("No market data downloaded from yfinance for provided NIFTY symbols.")

        df = pd.concat(rows).reset_index().rename(columns={"Date": "date", "index": "date"})
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

        def _feat(group: pd.DataFrame) -> pd.DataFrame:
            g = group.copy()
            g["ema_fast"] = g["close"].ewm(span=5, adjust=False).mean()
            g["ema_slow"] = g["close"].ewm(span=14, adjust=False).mean()
            delta = g["close"].diff().fillna(0.0)
            up = delta.clip(lower=0.0)
            down = -delta.clip(upper=0.0)
            roll_up = up.rolling(14, min_periods=1).mean()
            roll_down = down.rolling(14, min_periods=1).mean()
            rs = roll_up / np.maximum(roll_down, 1e-9)
            g["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
            g["vwap"] = (g["close"] * g["volume"]).cumsum() / np.maximum(g["volume"].cumsum(), 1.0)
            return g

        return df.groupby("symbol", group_keys=False).apply(_feat).reset_index(drop=True)

    @staticmethod
    def _build_episode_table(df: pd.DataFrame) -> pd.DataFrame:
        # Ensure enough history for indicators and deterministic grading.
        out = df.copy()
        out["row_num"] = out.groupby("symbol").cumcount()
        out = out[out["row_num"] >= 14].reset_index(drop=True)
        return out[["symbol", "date", "open", "high", "low", "close", "vwap", "ema_fast", "ema_slow", "rsi_14"]]


if __name__ == "__main__":
    env = BulktradeEnvironment()
    obs = env.reset()
    print("Reset:", obs.task_name, obs.symbol, obs.current_price)
    for _ in range(5):
        step_obs = env.step(BulktradeAction(action_type=obs.target_side, quantity=100))
        print("Step", step_obs.step_index, "reward", step_obs.reward, "remaining", step_obs.remaining_quantity)
        if step_obs.done:
            break
