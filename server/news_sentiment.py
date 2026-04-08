"""Lightweight news sentiment utilities for market-aware observations.

Design goal: minimal dependency, deterministic scoring logic, optional live feeds.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Sequence, Tuple
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class SentimentSnapshot:
    score: float
    confidence: float
    headlines: List[str]
    source: str


POS_WORDS = {
    "gain", "gains", "up", "beat", "surge", "strong", "bull", "rally", "growth", "profit", "upgrade",
    "optimism", "record", "rebound", "rise", "improve", "outperform",
}
NEG_WORDS = {
    "fall", "falls", "down", "miss", "drop", "weak", "bear", "slump", "loss", "downgrade", "crash",
    "fear", "risk", "decline", "warning", "concern", "underperform", "cut",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def score_headlines(headlines: Sequence[str]) -> Tuple[float, float]:
    """Tiny lexical sentiment scorer (small-LM-style baseline).

    Returns:
      score in [-1, 1], confidence in [0, 1]
    """
    if not headlines:
        return 0.0, 0.0

    pos = 0
    neg = 0
    total = 0
    for h in headlines:
        toks = _tokenize(h)
        total += len(toks)
        pos += sum(1 for t in toks if t in POS_WORDS)
        neg += sum(1 for t in toks if t in NEG_WORDS)

    if pos + neg == 0:
        return 0.0, 0.1

    raw = (pos - neg) / max(pos + neg, 1)
    confidence = min(1.0, (pos + neg) / max(total, 1) * 8.0)
    score = max(-1.0, min(1.0, raw))
    return float(score), float(confidence)


def _read_rss_titles(url: str, max_items: int = 8) -> List[str]:
    req = Request(url, headers={"User-Agent": "TradeEnv-Sentiment/1.0"})
    with urlopen(req, timeout=4.0) as resp:  # nosec B310 - intentional public RSS fetch
        payload = resp.read()
    root = ET.fromstring(payload)
    titles: List[str] = []
    for item in root.findall(".//item"):
        t = item.findtext("title")
        if t:
            titles.append(t.strip())
        if len(titles) >= max_items:
            break
    return titles


def fetch_sentiment_snapshot() -> SentimentSnapshot:
    """Fetch and score tiny headline set from Yahoo + Moneycontrol RSS.

    Graceful fallback: returns neutral signal when feeds unavailable.
    """
    feeds = [
        ("yahoo_finance", "https://finance.yahoo.com/rss/topstories"),
        ("moneycontrol", "https://www.moneycontrol.com/rss/business.xml"),
    ]

    all_titles: List[str] = []
    used_sources: List[str] = []
    for src, url in feeds:
        try:
            titles = _read_rss_titles(url, max_items=6)
            if titles:
                all_titles.extend(titles)
                used_sources.append(src)
        except Exception:
            continue

    score, confidence = score_headlines(all_titles)
    source = "+".join(used_sources) if used_sources else "neutral_fallback"
    return SentimentSnapshot(score=score, confidence=confidence, headlines=all_titles[:8], source=source)
