"""Broker integration placeholders for future live-trading connectivity.

This module intentionally provides stubs only (no real brokerage calls) to keep
hackathon scope safe and deterministic while exposing extension points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class BrokerOrder:
    symbol: str
    side: str
    quantity: int
    order_type: str = "MARKET"
    limit_price: Optional[float] = None


class BaseBrokerAdapter:
    name: str = "base"

    def is_configured(self) -> bool:
        return False

    def place_order(self, order: BrokerOrder) -> Dict:
        return {
            "status": "placeholder",
            "broker": self.name,
            "message": "Broker integration is not enabled in hackathon mode.",
            "order": order.__dict__,
        }


class ZerodhaAdapter(BaseBrokerAdapter):
    name = "zerodha"


class PaytmMoneyAdapter(BaseBrokerAdapter):
    name = "paytm_money"


def get_supported_brokers() -> Dict[str, str]:
    return {
        "zerodha": "Placeholder adapter (future live execution via Kite Connect)",
        "paytm_money": "Placeholder adapter (future live execution via Paytm Money APIs)",
    }
