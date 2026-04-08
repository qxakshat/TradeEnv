# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed models for the bulk execution OpenEnv environment."""

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field

# Trading role types with distinct strategies and constraints
TradingRole = Literal[
    "aggressive_buyer",     # Fast execution, accepts slippage
    "conservative_seller",  # High-quality sells, patient
    "market_maker",         # Balanced timing, profit from volume
    "arbitrageur",          # Exploits mispricings, low slippage tolerance
    "volatility_trader",    # Trades price swings, sophisticated timing
]


class TradeEnvAction(Action):
    """Agent action for bulk execution."""

    action_type: Literal["buy", "sell", "hold"] = Field(
        default="hold",
        description="Execution action. Must match the task side for a valid trade.",
    )
    quantity: int = Field(
        default=0,
        ge=0,
        le=10_000,
        description="Units to execute on this step (0 allowed for hold/no-op).",
    )
    urgency: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Execution urgency hint (0=patient, 1=aggressive).",
    )


class TradeEnvReward(BaseModel):
    """Dense reward decomposition for trajectory-level learning signal."""

    immediate_edge: float = Field(description="Price edge versus running VWAP benchmark")
    progress_bonus: float = Field(description="Reward for making required execution progress")
    slippage_penalty: float = Field(description="Penalty for high estimated slippage and wide spread")
    trade_efficiency_bonus: float = Field(description="Bonus for achieving target using fewer trades")
    depth_timing_bonus: float = Field(description="Bonus for executing when order-book depth is favorable")
    random_trade_penalty: float = Field(description="Penalty for noisy/random trade patterns (flip-flop, tiny spam trades)")
    variance_penalty: float = Field(description="Penalty for high variance in execution prices over the trajectory")
    constraint_penalty: float = Field(description="Penalty for invalid actions/over-trading")
    terminal_adjustment: float = Field(description="End-of-episode adjustment for completion quality")
    role_adjustment: float = Field(description="Role-specific reward adjustment (aggressiveness, conservatism, etc.)")
    total: float = Field(description="Final scalar reward used by the environment")


class TradeEnvTaskScore(BaseModel):
    """Deterministic task score in [0, 1]."""

    task_name: str
    difficulty: Literal["easy", "medium", "hard"]
    role: TradingRole
    score: float = Field(ge=0.0, le=1.0, description="Overall task score")
    role_metric: float = Field(ge=0.0, le=1.0, description="Role-specific performance metric")
    quality_metric: float = Field(ge=0.0, le=1.0, description="Execution quality versus day range")
    fill_metric: float = Field(ge=0.0, le=1.0, description="Target completion ratio")
    slippage_metric: float = Field(ge=0.0, le=1.0, description="Execution quality versus spread and book depth")
    efficiency_metric: float = Field(ge=0.0, le=1.0, description="Preference for lower trade count with full completion")
    rationale: str


class TradeEnvObservation(Observation):
    """Observation for the current execution state."""

    task_name: str = Field(description="Current task identifier")
    difficulty: Literal["easy", "medium", "hard"]
    role: TradingRole = Field(description="Current trading role/strategy")
    symbol: str
    step_index: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    current_price: float = Field(gt=0)
    best_bid: float = Field(gt=0)
    best_ask: float = Field(gt=0)
    spread_bps: float = Field(ge=0)
    bid_depth_top: float = Field(ge=0)
    ask_depth_top: float = Field(ge=0)
    depth_imbalance: float = Field(ge=-1.0, le=1.0)
    estimated_slippage_bps: float = Field(ge=0)
    news_sentiment_score: float = Field(ge=-1.0, le=1.0)
    news_sentiment_confidence: float = Field(ge=0.0, le=1.0)
    pe_ratio: float = Field(ge=0)
    price_to_book: float = Field(ge=0)
    return_on_equity: float = Field(ge=-1.0, le=2.0)
    debt_to_equity: float = Field(ge=0)
    profit_margin: float = Field(ge=-1.0, le=1.0)
    revenue_growth: float = Field(ge=-1.0, le=3.0)
    fundamental_quality_score: float = Field(ge=0.0, le=1.0)
    ema_fast: float = Field(gt=0)
    ema_slow: float = Field(gt=0)
    rsi_14: float = Field(ge=0, le=100)
    target_side: Literal["buy", "sell"]
    target_quantity: int = Field(gt=0)
    remaining_quantity: int = Field(ge=0)
    executed_quantity: int = Field(ge=0)
    avg_execution_price: float = Field(ge=0)
    trades_used: int = Field(ge=0)
    max_trades_per_day: int = Field(ge=1)
