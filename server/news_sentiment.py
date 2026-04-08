"""Lightweight news sentiment utilities for market-aware observations.

Design goal: minimal dependency, deterministic scoring logic, optional live feeds.
"""

from __future__ import annotations

import json
import pathlib
import re
import time
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


def _cache_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent / "data" / "sentiment_snapshot.json"


def _save_snapshot_cache(snapshot: SentimentSnapshot) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "score": snapshot.score,
        "confidence": snapshot.confidence,
        "headlines": snapshot.headlines,
        "source": snapshot.source,
        "cached_at_epoch": int(time.time()),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load_snapshot_cache(max_age_seconds: int = 48 * 3600) -> SentimentSnapshot | None:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cached_at = int(payload.get("cached_at_epoch", 0))
        age = int(time.time()) - cached_at
        if max_age_seconds > 0 and age > max_age_seconds:
            return None
        return SentimentSnapshot(
            score=float(payload.get("score", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            headlines=list(payload.get("headlines", []))[:8],
            source=str(payload.get("source", "cache_fallback")) + "_cache",
        )
    except Exception:
        return None


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

    if all_titles:
        score, confidence = score_headlines(all_titles)
        source = "+".join(used_sources)
        snapshot = SentimentSnapshot(score=score, confidence=confidence, headlines=all_titles[:8], source=source)
        _save_snapshot_cache(snapshot)
        return snapshot

    cached = _load_snapshot_cache(max_age_seconds=48 * 3600)
    if cached is not None:
        return cached

    return SentimentSnapshot(score=0.0, confidence=0.0, headlines=[], source="neutral_fallback")
