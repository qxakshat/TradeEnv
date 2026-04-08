# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed models for the bulk execution OpenEnv environment."""

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class BulktradeAction(Action):
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


class BulktradeReward(BaseModel):
    """Dense reward decomposition for trajectory-level learning signal."""

    immediate_edge: float = Field(description="Price edge versus running VWAP benchmark")
    progress_bonus: float = Field(description="Reward for making required execution progress")
    constraint_penalty: float = Field(description="Penalty for invalid actions/over-trading")
    terminal_adjustment: float = Field(description="End-of-episode adjustment for completion quality")
    total: float = Field(description="Final scalar reward used by the environment")


class BulktradeTaskScore(BaseModel):
    """Deterministic task score in [0, 1]."""

    task_name: str
    difficulty: Literal["easy", "medium", "hard"]
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class BulktradeObservation(Observation):
    """Observation for the current execution state."""

    task_name: str = Field(description="Current task identifier")
    difficulty: Literal["easy", "medium", "hard"]
    symbol: str
    step_index: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    current_price: float = Field(gt=0)
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
