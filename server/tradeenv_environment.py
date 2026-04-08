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
        TradeEnvAction,
        TradeEnvObservation,
        TradeEnvReward,
        TradeEnvTaskScore,
        TradingRole,
    )
    from .broker_adapters import get_supported_brokers
    from .fundamentals import load_or_fetch_fundamentals
    from .news_sentiment import SentimentSnapshot, fetch_sentiment_snapshot
except ImportError:
    from models import TradeEnvAction, TradeEnvObservation, TradeEnvReward, TradeEnvTaskScore, TradingRole
    from server.broker_adapters import get_supported_brokers
    from server.fundamentals import load_or_fetch_fundamentals
    from server.news_sentiment import SentimentSnapshot, fetch_sentiment_snapshot


NIFTY50_SYMBOLS: List[str] = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS",
    "LT.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "AXISBANK.NS", "ASIANPAINT.NS", "BAJFINANCE.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS", "POWERGRID.NS",
]


@dataclass(frozen=True)
class RoleSpec:
    role: TradingRole
    description: str
    slippage_factor: float
    speed_bonus_weight: float
    quality_weight: float
    consistency_reward: float


ROLES: List[RoleSpec] = [
    RoleSpec("aggressive_buyer", "Fast completion-focused buyer", 1.30, 0.35, 0.35, 0.00),
    RoleSpec("conservative_seller", "Patient high-quality seller", 0.70, 0.05, 0.55, 0.08),
    RoleSpec("market_maker", "Balanced execution and inventory control", 0.90, 0.15, 0.40, 0.12),
    RoleSpec("arbitrageur", "Mispricing hunter with tight spread discipline", 0.55, 0.00, 0.70, 0.08),
    RoleSpec("volatility_trader", "Executes around volatility shocks", 0.85, 0.20, 0.50, 0.10),
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
        target_quantity=1_100,
        max_trades=8,
        symbols=["INFY.NS", "ICICIBANK.NS", "LT.NS", "SBIN.NS"],
        description="Sell inventory near intraday highs while respecting action budget.",
    ),
    TaskSpec(
        name="hard_depth_aware_execution",
        difficulty="hard",
        side="buy",
        target_quantity=1_250,
        max_trades=6,
        symbols=["BAJFINANCE.NS", "MARUTI.NS", "WIPRO.NS", "TITAN.NS"],
        description="Buy volatile names under thin depth and wider spreads with low slippage.",
    ),
]


class TradeEnvEnvironment(Environment):
    """Depth-aware bulk execution simulator with deterministic graders."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._task_idx = -1
        self._role_idx = -1
        self._task: TaskSpec = TASKS[0]
        self._role: RoleSpec = ROLES[0]
        self._market: pd.DataFrame | None = None
        self._episodes: pd.DataFrame | None = None
        self._fundamentals: pd.DataFrame | None = None
        self._fundamental_sources: Dict[str, List[str]] = {}

        self._price_path: List[float] = []
        self._book_path: List[Dict[str, float]] = []
        self._step_idx = 0
        self._executed_qty = 0
        self._remaining_qty = 0
        self._cum_notional = 0.0
        self._trades_used = 0
        self._fills: List[Tuple[int, float]] = []
        self._last_reward = 0.0
        self._last_slippage_bps = 0.0
        self._last_score = TradeEnvTaskScore(
            task_name="",
            difficulty="easy",
            role="aggressive_buyer",
            score=0.0,
            role_metric=0.0,
            quality_metric=0.0,
            fill_metric=0.0,
            slippage_metric=0.0,
            efficiency_metric=0.0,
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
        self._prev_action_type = "hold"
        self._prev_exec_qty = 0
        self._sentiment = SentimentSnapshot(score=0.0, confidence=0.0, headlines=[], source="neutral_fallback")
        self._current_fundamentals: Dict[str, float] = {
            "pe_ratio": 20.0,
            "price_to_book": 3.0,
            "return_on_equity": 0.12,
            "debt_to_equity": 80.0,
            "profit_margin": 0.08,
            "revenue_growth": 0.10,
            "fundamental_quality_score": 0.50,
        }

    def reset(self) -> TradeEnvObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._task_idx = (self._task_idx + 1) % len(TASKS)
        self._role_idx = (self._reset_count - 1) % len(ROLES)
        self._task = TASKS[self._task_idx]
        self._role = ROLES[self._role_idx]

        if self._market is None:
            self._market = self._download_market_data(NIFTY50_SYMBOLS)
            self._episodes = self._build_episode_table(self._market)
        if self._fundamentals is None:
            import pathlib

            snapshot = load_or_fetch_fundamentals(
                NIFTY50_SYMBOLS,
                cache_path=pathlib.Path(__file__).parent.parent / "data" / "fundamentals_snapshot.csv",
            )
            self._fundamentals = snapshot.table
            self._fundamental_sources = snapshot.source_urls

        assert self._episodes is not None
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
        self._set_current_fundamentals(self._symbol)

        self._price_path = self._synthetic_intraday_path(
            open_price=float(row["open"]),
            low_price=self._day_low,
            high_price=self._day_high,
            close_price=float(row["close"]),
            n_steps=12,
        )
        self._book_path = self._simulate_market_depth(
            symbol=self._symbol,
            date=self._date,
            prices=self._price_path,
            difficulty=self._task.difficulty,
        )

        self._step_idx = 0
        self._executed_qty = 0
        self._remaining_qty = self._task.target_quantity
        self._cum_notional = 0.0
        self._trades_used = 0
        self._fills = []
        self._last_reward = 0.0
        self._last_slippage_bps = 0.0
        self._prev_action_type = "hold"
        self._prev_exec_qty = 0
        self._last_score = TradeEnvTaskScore(
            task_name=self._task.name,
            difficulty=self._task.difficulty,
            role=self._role.role,
            score=0.0,
            role_metric=0.0,
            quality_metric=0.0,
            fill_metric=0.0,
            slippage_metric=0.0,
            efficiency_metric=0.0,
            rationale="in_progress",
        )

        # Sentiment is advisory (in observation), not part of deterministic grading.
        self._sentiment = fetch_sentiment_snapshot()

        return self._build_observation(done=False, reward=0.0, info={"phase": "reset", "date": self._date})

    def step(self, action: TradeEnvAction) -> TradeEnvObservation:  # type: ignore[override]
        _, reward, done, info = self.step_transition(action)
        return self._build_observation(done=done, reward=reward, info=info)

    def step_transition(self, action: TradeEnvAction) -> Tuple[TradeEnvObservation, float, bool, Dict]:
        self._state.step_count += 1
        self._step_idx = min(self._step_idx + 1, len(self._price_path) - 1)
        current_price = float(self._price_path[self._step_idx])
        book = self._book_path[self._step_idx]

        immediate_edge = 0.0
        progress_bonus = 0.0
        slippage_penalty = 0.0
        trade_efficiency_bonus = 0.0
        depth_timing_bonus = 0.0
        random_trade_penalty = 0.0
        variance_penalty = 0.0
        constraint_penalty = 0.0
        terminal_adjustment = 0.0
        role_adjustment = 0.0

        requested_qty = int(max(action.quantity, 0))
        urgency = float(np.clip(action.urgency, 0.0, 1.0))

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
                available_depth = max(book["bid_depth_top"] if self._task.side == "sell" else book["ask_depth_top"], 1.0)
                depth_pressure = min(executed_qty / available_depth, 3.0)

                base_slippage = 0.0008 + 0.0075 * participation + 0.0015 * depth_pressure
                urgency_slippage = 0.0025 * urgency
                slippage = (base_slippage + urgency_slippage) * self._role.slippage_factor
                self._last_slippage_bps = slippage * 10_000

                if self._task.side == "buy":
                    fill_price = current_price * (1.0 + slippage)
                    immediate_edge = (self._day_vwap - fill_price) / max(self._day_vwap, 1e-6)
                else:
                    fill_price = current_price * (1.0 - slippage)
                    immediate_edge = (fill_price - self._day_vwap) / max(self._day_vwap, 1e-6)

                progress_bonus = 0.08 * participation
                if self._role.speed_bonus_weight > 0:
                    time_progress = self._step_idx / max(len(self._price_path) - 1, 1)
                    progress_bonus += self._role.speed_bonus_weight * 0.05 * (1.0 - time_progress)

                spread_pen = min(book["spread_bps"] / 45.0, 1.0)
                slip_pen = min(self._last_slippage_bps / 120.0, 1.0)
                slippage_penalty = -0.08 * (0.6 * spread_pen + 0.4 * slip_pen)

                depth_favorability = 0.5 * (book["depth_imbalance"] + 1.0)
                if self._task.side == "sell":
                    depth_favorability = 1.0 - depth_favorability
                depth_timing_bonus = 0.05 * (depth_favorability - 0.5)

                # Penalize noisy/random micro-trading and side flip behavior.
                tiny_trade = executed_qty < max(int(0.04 * self._task.target_quantity), 1)
                flip_flop = self._prev_action_type != "hold" and self._prev_action_type != action.action_type
                if tiny_trade:
                    random_trade_penalty -= 0.025
                if flip_flop:
                    random_trade_penalty -= 0.015

                self._prev_action_type = action.action_type
                self._prev_exec_qty = executed_qty

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
            terminal_adjustment -= 0.22

        if done:
            self._last_score = self._grade_current_task()
            terminal_adjustment += (self._last_score.score - 0.5) * 0.22
            trade_efficiency_bonus = 0.10 * self._last_score.efficiency_metric
            role_adjustment = self._role.consistency_reward * self._last_score.role_metric
            if self._fills:
                fill_prices = np.array([p for _, p in self._fills], dtype=np.float64)
                if len(fill_prices) >= 2:
                    cv = float(np.std(fill_prices) / max(np.mean(fill_prices), 1e-9))
                    variance_penalty = -0.12 * float(np.clip(cv / 0.03, 0.0, 1.0))

        reward_obj = TradeEnvReward(
            immediate_edge=immediate_edge,
            progress_bonus=progress_bonus,
            slippage_penalty=slippage_penalty,
            trade_efficiency_bonus=trade_efficiency_bonus,
            depth_timing_bonus=depth_timing_bonus,
            random_trade_penalty=random_trade_penalty,
            variance_penalty=variance_penalty,
            constraint_penalty=constraint_penalty,
            terminal_adjustment=terminal_adjustment,
            role_adjustment=role_adjustment,
            total=(
                immediate_edge
                + progress_bonus
                + slippage_penalty
                + trade_efficiency_bonus
                + depth_timing_bonus
                + random_trade_penalty
                + variance_penalty
                + constraint_penalty
                + terminal_adjustment
                + role_adjustment
            ),
        )
        self._last_reward = reward_obj.total

        info = {
            "task": self._task.name,
            "role": self._role.role,
            "date": self._date,
            "symbol": self._symbol,
            "invalid_reason": invalid_reason,
            "reward_breakdown": reward_obj.model_dump(),
            "task_score": self._last_score.model_dump() if done else None,
            "broker_integration": get_supported_brokers(),
            "news_sentiment": {
                "score": self._sentiment.score,
                "confidence": self._sentiment.confidence,
                "source": self._sentiment.source,
                "headlines": self._sentiment.headlines,
            },
            "fundamentals": {
                "symbol": self._symbol,
                "values": self._current_fundamentals,
                "sources": self._fundamental_sources.get(self._symbol, []),
            },
        }

        obs = self._build_observation(done=done, reward=reward_obj.total, info=info)
        return obs, reward_obj.total, done, info

    def _build_observation(self, done: bool, reward: float, info: Dict) -> TradeEnvObservation:
        book = self._book_path[self._step_idx]
        avg_px = self._cum_notional / self._executed_qty if self._executed_qty > 0 else 0.0
        return TradeEnvObservation(
            task_name=self._task.name,
            difficulty=self._task.difficulty,
            role=self._role.role,
            symbol=self._symbol,
            step_index=self._step_idx,
            max_steps=len(self._price_path),
            current_price=float(self._price_path[self._step_idx]),
            best_bid=float(book["best_bid"]),
            best_ask=float(book["best_ask"]),
            spread_bps=float(book["spread_bps"]),
            bid_depth_top=float(book["bid_depth_top"]),
            ask_depth_top=float(book["ask_depth_top"]),
            depth_imbalance=float(book["depth_imbalance"]),
            estimated_slippage_bps=float(self._last_slippage_bps),
            news_sentiment_score=float(self._sentiment.score),
            news_sentiment_confidence=float(self._sentiment.confidence),
            pe_ratio=float(self._current_fundamentals["pe_ratio"]),
            price_to_book=float(self._current_fundamentals["price_to_book"]),
            return_on_equity=float(self._current_fundamentals["return_on_equity"]),
            debt_to_equity=float(self._current_fundamentals["debt_to_equity"]),
            profit_margin=float(self._current_fundamentals["profit_margin"]),
            revenue_growth=float(self._current_fundamentals["revenue_growth"]),
            fundamental_quality_score=float(self._current_fundamentals["fundamental_quality_score"]),
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
                "role_description": self._role.description,
                "date": self._date,
                "fundamentals": {
                    "symbol": self._symbol,
                    "values": self._current_fundamentals,
                    "sources": self._fundamental_sources.get(self._symbol, []),
                },
                "news_sentiment": {
                    "score": self._sentiment.score,
                    "confidence": self._sentiment.confidence,
                    "source": self._sentiment.source,
                    "headlines": self._sentiment.headlines,
                },
                "info": info,
                "ui": {
                    "primary_metric": "task_score",
                    "secondary_metric": "estimated_slippage_bps",
                    "charts": ["reward_breakdown", "depth_imbalance", "spread_bps", "fundamental_quality"],
                },
            },
        )

    def _set_current_fundamentals(self, symbol: str) -> None:
        if self._fundamentals is None or self._fundamentals.empty:
            return
        row = self._fundamentals[self._fundamentals["symbol"] == symbol]
        if row.empty:
            return
        rr = row.iloc[0]
        self._current_fundamentals = {
            "pe_ratio": float(rr.get("pe_ratio", 20.0)),
            "price_to_book": float(rr.get("price_to_book", 3.0)),
            "return_on_equity": float(rr.get("return_on_equity", 0.12)),
            "debt_to_equity": float(rr.get("debt_to_equity", 80.0)),
            "profit_margin": float(rr.get("profit_margin", 0.08)),
            "revenue_growth": float(rr.get("revenue_growth", 0.10)),
            "fundamental_quality_score": float(rr.get("fundamental_quality_score", 0.50)),
        }

    def _grade_current_task(self) -> TradeEnvTaskScore:
        if self._executed_qty <= 0:
            return TradeEnvTaskScore(
                task_name=self._task.name,
                difficulty=self._task.difficulty,
                role=self._role.role,
                score=0.0,
                role_metric=0.0,
                quality_metric=0.0,
                fill_metric=0.0,
                slippage_metric=0.0,
                efficiency_metric=0.0,
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

        trades_util = self._trades_used / max(self._task.max_trades, 1)
        efficiency = float(np.clip(1.0 - trades_util, 0.0, 1.0))

        realized_slippage_bps = abs(avg_price - self._day_vwap) / max(self._day_vwap, 1e-6) * 10_000
        slippage_metric = float(np.clip(1.0 - (realized_slippage_bps / 220.0), 0.0, 1.0))

        if self._task.difficulty == "easy":
            score = 0.70 * quality + 0.20 * fill_ratio + 0.10 * slippage_metric
        elif self._task.difficulty == "medium":
            score = 0.55 * quality + 0.25 * fill_ratio + 0.15 * slippage_metric + 0.05 * efficiency
        else:
            score = 0.35 * quality + 0.25 * fill_ratio + 0.25 * slippage_metric + 0.15 * efficiency

        if self._role.role == "aggressive_buyer":
            role_metric = 0.45 * quality + 0.35 * fill_ratio + 0.20 * efficiency
        elif self._role.role == "conservative_seller":
            role_metric = 0.55 * quality + 0.30 * slippage_metric + 0.15 * efficiency
        elif self._role.role == "market_maker":
            balance = float(np.clip(1.0 - abs(fill_ratio - 1.0), 0.0, 1.0))
            role_metric = 0.40 * quality + 0.30 * balance + 0.30 * slippage_metric
        elif self._role.role == "arbitrageur":
            role_metric = 0.20 * quality + 0.70 * slippage_metric + 0.10 * efficiency
        else:
            time_score = float(np.clip(self._step_idx / max(len(self._price_path) - 1, 1), 0.0, 1.0))
            role_metric = 0.45 * quality + 0.20 * fill_ratio + 0.15 * efficiency + 0.20 * (1.0 - time_score)

        return TradeEnvTaskScore(
            task_name=self._task.name,
            difficulty=self._task.difficulty,
            role=self._role.role,
            score=float(np.clip(score, 0.0, 1.0)),
            role_metric=float(np.clip(role_metric, 0.0, 1.0)),
            quality_metric=quality,
            fill_metric=float(fill_ratio),
            slippage_metric=slippage_metric,
            efficiency_metric=efficiency,
            rationale=(
                f"quality={quality:.3f}, fill={fill_ratio:.3f}, "
                f"slippage={slippage_metric:.3f}, efficiency={efficiency:.3f}, role_metric={role_metric:.3f}"
            ),
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
    def _stable_key(symbol: str, date: str, step_idx: int) -> int:
        text = f"{symbol}|{date}|{step_idx}"
        return int(sum((i + 1) * ord(ch) for i, ch in enumerate(text)))

    @classmethod
    def _simulate_market_depth(
        cls,
        symbol: str,
        date: str,
        prices: List[float],
        difficulty: Literal["easy", "medium", "hard"],
    ) -> List[Dict[str, float]]:
        if difficulty == "easy":
            liq_mult, spread_base = 1.2, 8.0
        elif difficulty == "medium":
            liq_mult, spread_base = 1.0, 14.0
        else:
            liq_mult, spread_base = 0.7, 22.0

        books: List[Dict[str, float]] = []
        for i, px in enumerate(prices):
            k = cls._stable_key(symbol, date, i)
            cyc = np.sin(0.73 * i + (k % 17) * 0.1)
            cyc2 = np.cos(0.41 * i + (k % 13) * 0.07)

            spread_bps = max(3.5, spread_base + 3.2 * cyc + 2.1 * cyc2)
            half_spread = (spread_bps / 10_000.0) * px * 0.5
            best_bid = px - half_spread
            best_ask = px + half_spread

            base_depth = 900.0 * liq_mult
            bid_depth = max(80.0, base_depth * (1.0 + 0.30 * cyc))
            ask_depth = max(80.0, base_depth * (1.0 - 0.25 * cyc2))
            imbalance = float(np.clip((bid_depth - ask_depth) / max(bid_depth + ask_depth, 1.0), -1.0, 1.0))

            books.append(
                {
                    "best_bid": float(best_bid),
                    "best_ask": float(best_ask),
                    "spread_bps": float(spread_bps),
                    "bid_depth_top": float(bid_depth),
                    "ask_depth_top": float(ask_depth),
                    "depth_imbalance": imbalance,
                }
            )
        return books

    @staticmethod
    def _download_market_data(symbols: List[str]) -> pd.DataFrame:
        import pathlib

        csv_path = pathlib.Path(__file__).parent.parent / "data" / "nifty50_market_data.csv"
        if csv_path.exists():
            print(f"Loading market data from {csv_path}")
            return pd.read_csv(csv_path)

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
            raise RuntimeError(
                "No market data downloaded from yfinance for provided NIFTY symbols. "
                "Run the data download script in notebooks/hackathon_project.ipynb "
                "to fetch and save NIFTY50 price data, then commit it to the repo."
            )

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
        out = df.copy()
        out["row_num"] = out.groupby("symbol").cumcount()
        out = out[out["row_num"] >= 14].reset_index(drop=True)
        return out[["symbol", "date", "open", "high", "low", "close", "vwap", "ema_fast", "ema_slow", "rsi_14"]]


if __name__ == "__main__":
    env = TradeEnvEnvironment()
    obs = env.reset()
    print("Reset:", obs.task_name, obs.role, obs.symbol, obs.current_price)
    for _ in range(5):
        step_obs = env.step(TradeEnvAction(action_type=obs.target_side, quantity=120, urgency=0.6))
        print("Step", step_obs.step_index, "reward", step_obs.reward, "spread_bps", step_obs.spread_bps)
        if step_obs.done:
            break
