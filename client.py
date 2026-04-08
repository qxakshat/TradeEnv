# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TradeEnv Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import TradeEnvAction, TradeEnvObservation
except ImportError:
    from models import TradeEnvAction, TradeEnvObservation


class TradeEnvClient(
    EnvClient[TradeEnvAction, TradeEnvObservation, State]
):
    """
    Client for the TradeEnv Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with TradeEnvClient(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task_name)
        ...
        ...     result = client.step(TradeEnvAction(action_type="buy", quantity=100))
        ...     print(result.observation.remaining_quantity)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = TradeEnvClient.from_docker_image("tradeenv:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(TradeEnvAction(action_type="buy", quantity=100))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: TradeEnvAction) -> Dict:
        """
        Convert TradeEnvAction to JSON payload for step message.

        Args:
            action: TradeEnvAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "quantity": action.quantity,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TradeEnvObservation]:
        """
        Parse server response into StepResult[TradeEnvObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with TradeEnvObservation
        """
        obs_data = payload.get("observation", {})
        observation = TradeEnvObservation(
            task_name=obs_data.get("task_name", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            symbol=obs_data.get("symbol", ""),
            step_index=obs_data.get("step_index", 0),
            max_steps=obs_data.get("max_steps", 1),
            current_price=obs_data.get("current_price", 1.0),
            ema_fast=obs_data.get("ema_fast", 1.0),
            ema_slow=obs_data.get("ema_slow", 1.0),
            rsi_14=obs_data.get("rsi_14", 50.0),
            target_side=obs_data.get("target_side", "buy"),
            target_quantity=obs_data.get("target_quantity", 1),
            remaining_quantity=obs_data.get("remaining_quantity", 0),
            executed_quantity=obs_data.get("executed_quantity", 0),
            avg_execution_price=obs_data.get("avg_execution_price", 0.0),
            trades_used=obs_data.get("trades_used", 0),
            max_trades_per_day=obs_data.get("max_trades_per_day", 10),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
