"""Web-sourced fundamentals snapshot utilities (Yahoo Finance + optional source links).

Minimal implementation for environment-level fundamental features and provenance.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass
from typing import Dict, List
from urllib.request import Request, urlopen

import pandas as pd


@dataclass(frozen=True)
class FundamentalsSnapshot:
    table: pd.DataFrame
    source_urls: Dict[str, List[str]]
    fetched_at_epoch: int


def _raw(v, default=0.0):
    if isinstance(v, dict):
        return v.get("raw", default)
    return v if v is not None else default


def _fetch_yahoo_quote_summary(symbol: str) -> Dict[str, float]:
    url = (
        f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        "?modules=defaultKeyStatistics,financialData,summaryDetail"
    )
    req = Request(url, headers={"User-Agent": "TradeEnv-Fundamentals/1.0"})
    with urlopen(req, timeout=5.0) as resp:  # nosec B310 - public finance endpoint
        payload = json.loads(resp.read().decode("utf-8"))

    q = payload.get("quoteSummary", {})
    result = (q.get("result") or [{}])[0]
    dks = result.get("defaultKeyStatistics", {})
    fin = result.get("financialData", {})
    summ = result.get("summaryDetail", {})

    pe_ratio = float(_raw(summ.get("trailingPE"), _raw(dks.get("trailingPE"), 0.0)))
    ptb = float(_raw(dks.get("priceToBook"), 0.0))
    roe = float(_raw(fin.get("returnOnEquity"), 0.0))
    dte = float(_raw(fin.get("debtToEquity"), 0.0))
    pm = float(_raw(fin.get("profitMargins"), 0.0))
    rg = float(_raw(fin.get("revenueGrowth"), 0.0))

    quality = (
        0.20 * max(0.0, min(1.0, 1.0 - pe_ratio / 45.0))
        + 0.15 * max(0.0, min(1.0, 1.0 - ptb / 8.0))
        + 0.25 * max(0.0, min(1.0, (roe + 0.1) / 0.5))
        + 0.15 * max(0.0, min(1.0, 1.0 - dte / 250.0))
        + 0.15 * max(0.0, min(1.0, (pm + 0.05) / 0.35))
        + 0.10 * max(0.0, min(1.0, (rg + 0.1) / 0.6))
    )

    return {
        "pe_ratio": max(pe_ratio, 0.0),
        "price_to_book": max(ptb, 0.0),
        "return_on_equity": float(roe),
        "debt_to_equity": max(dte, 0.0),
        "profit_margin": float(pm),
        "revenue_growth": float(rg),
        "fundamental_quality_score": float(max(0.0, min(1.0, quality))),
    }


def _default_row() -> Dict[str, float]:
    return {
        "pe_ratio": 20.0,
        "price_to_book": 3.0,
        "return_on_equity": 0.12,
        "debt_to_equity": 80.0,
        "profit_margin": 0.08,
        "revenue_growth": 0.10,
        "fundamental_quality_score": 0.50,
    }


def load_or_fetch_fundamentals(symbols: List[str], cache_path: pathlib.Path) -> FundamentalsSnapshot:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    now = int(time.time())

    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            if {"symbol", "fetched_at_epoch"}.issubset(df.columns):
                age = now - int(df["fetched_at_epoch"].max())
                if age < 7 * 24 * 3600:
                    src = {
                        s: [
                            f"https://finance.yahoo.com/quote/{s}/key-statistics/",
                            f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{s}?modules=defaultKeyStatistics,financialData,summaryDetail",
                        ]
                        for s in df["symbol"].tolist()
                    }
                    return FundamentalsSnapshot(table=df, source_urls=src, fetched_at_epoch=now)
        except Exception:
            pass

    rows = []
    source_urls: Dict[str, List[str]] = {}
    for s in symbols:
        try:
            values = _fetch_yahoo_quote_summary(s)
        except Exception:
            values = _default_row()

        row = {"symbol": s, **values, "fetched_at_epoch": now}
        rows.append(row)
        source_urls[s] = [
            f"https://finance.yahoo.com/quote/{s}/key-statistics/",
            f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{s}?modules=defaultKeyStatistics,financialData,summaryDetail",
            "https://www.moneycontrol.com/",
        ]

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return FundamentalsSnapshot(table=df, source_urls=source_urls, fetched_at_epoch=now)
