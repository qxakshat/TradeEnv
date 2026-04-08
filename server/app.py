# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the TradeEnv Environment.

This module creates an HTTP server that exposes the TradeEnvEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Multi-Role Architecture:
    Each reset() automatically cycles through 5 distinct trading roles:
    - aggressive_buyer: Fast execution, accepts slippage
    - conservative_seller: High-quality, patient execution
    - market_maker: Balanced timing, profit from volume
    - arbitrageur: Misprice exploitation, tight costs
    - volatility_trader: Swing trading with technical sophistication

    This enforces model versatility: agents must adapt strategy to role constraints.

Primary Endpoints:
    - POST /reset: Reset the environment (cycles task + role)
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Additional UX Endpoints:
    - GET /healthz: Container health check
    - GET /ui-config: Frontend hints for charts and leaderboard display

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import TradeEnvAction, TradeEnvObservation
    from .tradeenv_environment import TradeEnvEnvironment
except ImportError:
    from models import TradeEnvAction, TradeEnvObservation
    from server.tradeenv_environment import TradeEnvEnvironment

import time
from functools import lru_cache

import yfinance as yf
from fastapi import Query
from fastapi.responses import HTMLResponse


# Create the app with web interface and README integration
app = create_app(
    TradeEnvEnvironment,
    TradeEnvAction,
    TradeEnvObservation,
    env_name="TradeEnv",
    max_concurrent_envs=4,
)


NIFTY50_DASHBOARD = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS",
    "LT.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "AXISBANK.NS", "ASIANPAINT.NS", "BAJFINANCE.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS", "POWERGRID.NS",
]


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


@lru_cache(maxsize=16)
def _cached_market_snapshot(cache_key: int, days: int) -> dict:
    """Cached market data snapshot (cache key is minute bucket)."""
    del cache_key
    raw = yf.download(
        tickers=NIFTY50_DASHBOARD,
        period=f"{max(days, 5)}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )

    symbols_payload = []
    for sym in NIFTY50_DASHBOARD:
        try:
            if hasattr(raw.columns, "levels"):
                if sym not in raw.columns.get_level_values(0):
                    continue
                sdf = raw[sym].copy()
            else:
                sdf = raw.copy()
            if sdf.empty:
                continue
            close = sdf["Close"].dropna()
            if close.empty:
                continue
            last = _safe_float(close.iloc[-1], 0.0)
            prev = _safe_float(close.iloc[-2], last) if len(close) >= 2 else last
            chg = last - prev
            pct = (chg / max(prev, 1e-9)) * 100.0
            symbols_payload.append(
                {
                    "symbol": sym,
                    "price": round(last, 4),
                    "change": round(chg, 4),
                    "change_pct": round(pct, 4),
                    "volume": int(_safe_float(sdf["Volume"].iloc[-1], 0.0)),
                }
            )
        except Exception:
            continue

    symbols_payload.sort(key=lambda x: x["change_pct"], reverse=True)
    return {
        "asof_epoch": int(time.time()),
        "symbols": symbols_payload,
        "advancers": sum(1 for x in symbols_payload if x["change_pct"] >= 0),
        "decliners": sum(1 for x in symbols_payload if x["change_pct"] < 0),
    }


def _get_market_snapshot(days: int) -> dict:
    minute_bucket = int(time.time() // 60)
    return _cached_market_snapshot(minute_bucket, days)


def _build_demo_portfolio(snapshot: dict) -> dict:
    top = snapshot.get("symbols", [])[:8]
    if not top:
        return {"holdings": [], "total_value": 0.0, "pnl_day": 0.0, "weights": []}

    holdings = []
    total_value = 0.0
    pnl_day = 0.0
    for i, s in enumerate(top, start=1):
        qty = 8 + i * 3
        value = qty * s["price"]
        pnl = qty * s["change"]
        total_value += value
        pnl_day += pnl
        holdings.append(
            {
                "symbol": s["symbol"],
                "qty": qty,
                "ltp": s["price"],
                "value": round(value, 2),
                "day_pnl": round(pnl, 2),
            }
        )

    weights = [
        {
            "symbol": h["symbol"],
            "weight": round((h["value"] / max(total_value, 1e-9)) * 100.0, 2),
        }
        for h in holdings
    ]
    return {
        "holdings": holdings,
        "total_value": round(total_value, 2),
        "pnl_day": round(pnl_day, 2),
        "weights": weights,
    }


@app.get("/healthz")
def healthz() -> dict:
    """Simple readiness endpoint for HF Spaces and container probes."""
    return {"status": "ok", "env": "TradeEnv"}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
        """Modern HF Space dashboard for human interaction and market visualization."""
        return """
<!doctype html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
    <title>TradeEnv Pro Dashboard</title>
    <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
    <style>
        :root { --bg:#0b1020; --panel:#121a30; --muted:#97a3c6; --txt:#e6ecff; --ok:#16c47f; --bad:#ff5c7a; --acc:#4f7cff; }
        body { margin:0; font-family:Inter,system-ui,Arial,sans-serif; background:var(--bg); color:var(--txt); }
        .wrap { max-width:1360px; margin:0 auto; padding:18px; }
        .hdr { display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }
        .title { font-size:28px; font-weight:700; }
        .sub { color:var(--muted); font-size:13px; }
        .grid { display:grid; grid-template-columns:repeat(12,1fr); gap:12px; }
        .card { background:var(--panel); border-radius:14px; padding:14px; box-shadow:0 0 0 1px rgba(255,255,255,.05) inset; }
        .kpi { grid-column:span 3; }
        .kpi .v { font-size:24px; font-weight:700; margin-top:6px; }
        .kpi .l { color:var(--muted); font-size:12px; }
        .up { color:var(--ok); } .dn { color:var(--bad); }
        .chart-lg { grid-column:span 8; min-height:340px; }
        .chart-sm { grid-column:span 4; min-height:340px; }
        .tbl { grid-column:span 12; overflow:auto; }
        table { width:100%; border-collapse:collapse; font-size:13px; }
        th,td { padding:8px; border-bottom:1px solid rgba(255,255,255,.07); text-align:right; }
        th:first-child,td:first-child { text-align:left; }
        .toolbar { display:flex; gap:8px; align-items:center; }
        select,input { background:#0f1630; color:#fff; border:1px solid #2a3767; border-radius:8px; padding:7px 9px; }
        button { background:var(--acc); color:#fff; border:none; border-radius:8px; padding:8px 12px; cursor:pointer; }
    </style>
</head>
<body>
    <div class=\"wrap\">
        <div class=\"hdr\">
            <div>
                <div class=\"title\">TradeEnv HF Space — Live NIFTY50 Control Center</div>
                <div class=\"sub\">Realtime snapshot + historical charts + portfolio analytics + execution environment context</div>
            </div>
            <div class=\"toolbar\">
                <label>Lookback days</label>
                <input id=\"days\" type=\"number\" min=\"5\" max=\"180\" value=\"30\" />
                <label>Ticker</label>
                <select id=\"symbol\"></select>
                <button onclick=\"refreshAll()\">Refresh</button>
            </div>
        </div>

        <div class=\"grid\">
            <div class=\"card kpi\"><div class=\"l\">Portfolio Value</div><div id=\"kpiValue\" class=\"v\">-</div></div>
            <div class=\"card kpi\"><div class=\"l\">Day P&L</div><div id=\"kpiPnl\" class=\"v\">-</div></div>
            <div class=\"card kpi\"><div class=\"l\">Advancers / Decliners</div><div id=\"kpiBreadth\" class=\"v\">-</div></div>
            <div class=\"card kpi\"><div class=\"l\">Fundamental Quality</div><div id=\"kpiFq\" class=\"v\">-</div></div>

            <div class=\"card chart-lg\"><div id=\"priceChart\" style=\"height:320px\"></div></div>
            <div class=\"card chart-sm\"><div id=\"pieChart\" style=\"height:320px\"></div></div>

            <div class=\"card chart-lg\"><div id=\"heatChart\" style=\"height:320px\"></div></div>
            <div class=\"card chart-sm\"><div id=\"depthChart\" style=\"height:320px\"></div></div>

            <div class=\"card tbl\">
                <div style=\"font-weight:600;margin-bottom:8px\">NIFTY-50 Watchlist Snapshot</div>
                <table id=\"watchTable\"><thead><tr><th>Symbol</th><th>Price</th><th>Δ</th><th>Δ%</th><th>Volume</th></tr></thead><tbody></tbody></table>
            </div>
        </div>
    </div>

    <script>
        let lastSnapshot = null;

        function fmt(n){ return new Intl.NumberFormat('en-IN',{maximumFractionDigits:2}).format(n||0); }

        async function fetchJSON(url){ const r = await fetch(url); if(!r.ok) throw new Error('fetch failed'); return await r.json(); }

        async function loadOverview(days){
            const data = await fetchJSON(`/api/market/overview?days=${days}`);
            lastSnapshot = data;
            const sel = document.getElementById('symbol');
            const cur = sel.value;
            sel.innerHTML = data.symbols.map(s=>`<option value=\"${s.symbol}\">${s.symbol}</option>`).join('');
            if(cur && data.symbols.some(x=>x.symbol===cur)) sel.value = cur;

            document.getElementById('kpiBreadth').textContent = `${data.advancers} / ${data.decliners}`;
            const tbody = document.querySelector('#watchTable tbody');
            tbody.innerHTML = data.symbols.slice(0,20).map(s=>`<tr>
                <td>${s.symbol}</td>
                <td>${fmt(s.price)}</td>
                <td class=\"${s.change>=0?'up':'dn'}\">${fmt(s.change)}</td>
                <td class=\"${s.change_pct>=0?'up':'dn'}\">${fmt(s.change_pct)}%</td>
                <td>${fmt(s.volume)}</td>
            </tr>`).join('');

            const x = data.symbols.slice(0,20).map(s=>s.symbol.replace('.NS',''));
            const y = data.symbols.slice(0,20).map(s=>s.change_pct);
            Plotly.newPlot('heatChart', [{type:'bar', x, y, marker:{color:y.map(v=>v>=0?'#16c47f':'#ff5c7a')}}],
                {title:'Top 20 Daily % Change', paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{color:'#dfe6ff'}});
        }

        async function loadHistory(symbol, days){
            const h = await fetchJSON(`/api/market/history/${symbol}?days=${days}`);
            Plotly.newPlot('priceChart', [
                {x:h.dates, y:h.close, type:'scatter', mode:'lines', name:'Close', line:{color:'#4f7cff'}},
                {x:h.dates, y:h.sma7, type:'scatter', mode:'lines', name:'SMA7', line:{color:'#f7b801'}},
            ], {title:`${symbol} — Last ${days} Days`, paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{color:'#dfe6ff'}});
        }

        async function loadPortfolio(){
            const p = await fetchJSON('/api/portfolio/demo');
            document.getElementById('kpiValue').textContent = `₹ ${fmt(p.total_value)}`;
            const pnl = document.getElementById('kpiPnl');
            pnl.textContent = `₹ ${fmt(p.pnl_day)}`;
            pnl.className = 'v ' + (p.pnl_day>=0?'up':'dn');
            Plotly.newPlot('pieChart', [{type:'pie', labels:p.weights.map(w=>w.symbol.replace('.NS','')), values:p.weights.map(w=>w.weight), hole:0.45}],
                {title:'Portfolio Allocation', paper_bgcolor:'transparent', font:{color:'#dfe6ff'}});
        }

        async function loadSignals(symbol){
            const s = await fetchJSON('/dashboard-snapshot');
            document.getElementById('kpiFq').textContent = (s.fundamentals?.fundamental_quality_score ?? 0).toFixed(3);
            const d = await fetchJSON(`/api/market/depth/${symbol}`);
            Plotly.newPlot('depthChart', [{type:'bar', x:['BidDepth','AskDepth','SpreadBps'], y:[d.bid_depth_top,d.ask_depth_top,d.spread_bps], marker:{color:['#16c47f','#ff5c7a','#4f7cff']}}],
                {title:`Microstructure — ${symbol}`, paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{color:'#dfe6ff'}});
        }

        async function refreshAll(){
            const days = Number(document.getElementById('days').value || 30);
            await loadOverview(days);
            const symbol = document.getElementById('symbol').value || 'RELIANCE.NS';
            await Promise.all([loadHistory(symbol, days), loadPortfolio(), loadSignals(symbol)]);
        }

        document.getElementById('symbol').addEventListener('change', refreshAll);
        setInterval(refreshAll, 45000);
        refreshAll();
    </script>
</body>
</html>
"""


@app.get("/api/market/overview")
def api_market_overview(days: int = Query(default=30, ge=5, le=180)) -> dict:
        return _get_market_snapshot(days)


@app.get("/api/market/history/{symbol}")
def api_market_history(symbol: str, days: int = Query(default=30, ge=5, le=180)) -> dict:
        s = symbol.upper()
        if s not in NIFTY50_DASHBOARD:
                s = "RELIANCE.NS"

        raw = yf.download(
                tickers=s,
                period=f"{days}d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
        )
        if raw.empty:
                return {"symbol": s, "dates": [], "close": [], "sma7": []}

        close_obj = raw["Close"]
        if hasattr(close_obj, "columns"):
            # Handle MultiIndex/dataframe return shape from yfinance.
            close = close_obj.iloc[:, 0].astype(float)
        else:
            close = close_obj.astype(float)
        sma7 = close.rolling(7, min_periods=1).mean()
        dates = [d.strftime("%Y-%m-%d") for d in close.index]
        return {
                "symbol": s,
                "dates": dates,
                "close": [round(float(x), 4) for x in close.tolist()],
                "sma7": [round(float(x), 4) for x in sma7.tolist()],
        }


@app.get("/api/portfolio/demo")
def api_portfolio_demo(days: int = Query(default=30, ge=5, le=180)) -> dict:
        snap = _get_market_snapshot(days)
        return _build_demo_portfolio(snap)


@app.get("/api/market/depth/{symbol}")
def api_market_depth(symbol: str) -> dict:
        """Depth proxy from environment simulation for UI diagnostics."""
        env = TradeEnvEnvironment()
        obs = env.reset()
        # rotate until symbol match (bounded)
        for _ in range(8):
                if obs.symbol.upper() == symbol.upper():
                        break
                obs = env.reset()
        return {
                "symbol": obs.symbol,
                "best_bid": obs.best_bid,
                "best_ask": obs.best_ask,
                "spread_bps": obs.spread_bps,
                "bid_depth_top": obs.bid_depth_top,
                "ask_depth_top": obs.ask_depth_top,
                "depth_imbalance": obs.depth_imbalance,
        }


@app.get("/ui-config")
def ui_config() -> dict:
    """UI metadata to enrich dashboard rendering in custom frontends."""
    return {
        "title": "TradeEnv — Depth-Aware Execution Arena",
        "subtitle": "Easy → Medium → Hard tasks with role-conditional grading",
        "cards": [
            {"key": "task_score", "label": "Task Score", "range": [0.0, 1.0]},
            {"key": "slippage_bps", "label": "Estimated Slippage (bps)", "range": [0, 500]},
            {"key": "trade_efficiency", "label": "Trade Efficiency", "range": [0.0, 1.0]},
            {"key": "fundamental_quality_score", "label": "Fundamental Quality", "range": [0.0, 1.0]},
        ],
        "charts": [
            {"id": "reward_breakdown", "type": "stacked_bar"},
            {"id": "spread_and_depth", "type": "line"},
            {"id": "score_by_role", "type": "bar"},
            {"id": "sentiment_vs_reward", "type": "scatter"},
        ],
        "controls": [
            {"id": "action_type", "label": "Action", "type": "select", "options": ["buy", "sell", "hold"]},
            {"id": "quantity", "label": "Quantity", "type": "slider", "min": 0, "max": 1000},
            {"id": "urgency", "label": "Urgency", "type": "slider", "min": 0.0, "max": 1.0, "step": 0.05},
        ],
        "leaderboard": {"group_by": ["task_name", "role"], "metric": "score"},
        "data_sources": {
            "fundamentals": [
                "https://finance.yahoo.com/quote/{symbol}/key-statistics/",
                "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=defaultKeyStatistics,financialData,summaryDetail",
                "https://www.moneycontrol.com/",
            ],
            "sentiment": [
                "https://finance.yahoo.com/rss/topstories",
                "https://www.moneycontrol.com/rss/business.xml",
            ],
        },
    }


@app.get("/frontend-readme")
def frontend_readme() -> dict:
    """Frontend guidance for human-friendly interaction."""
    markdown = """
# TradeEnv Frontend Guide

## Quick Start
1. Click **Reset** to start a new episode (task + role cycle).
2. Pick **Action** (`buy`/`sell`/`hold`), set **Quantity**, and optionally adjust **Urgency**.
3. Watch score + reward breakdown in charts after each step.

## What to watch
- **Task Score (0-1)**: end-goal quality metric
- **Spread / Depth**: avoid trading large size in wide-spread thin-depth moments
- **Sentiment Signal**: optional advisory signal from Yahoo/Moneycontrol headlines

## Human Tips
- Use fewer, better-timed trades for higher efficiency.
- On hard tasks, prioritize low slippage over raw speed.
- Avoid tiny random trades; they are explicitly penalized.
""".strip()
    return {"title": "TradeEnv Frontend README", "markdown": markdown}


@app.get("/dashboard-snapshot")
def dashboard_snapshot() -> dict:
    """One-step sample with source-backed values for dashboard panels."""
    env = TradeEnvEnvironment()
    obs = env.reset()
    meta = obs.metadata or {}
    info = meta.get("info", {})
    fundamentals_meta = meta.get("fundamentals", {})
    sentiment_meta = meta.get("news_sentiment", {})
    return {
        "task": obs.task_name,
        "role": obs.role,
        "symbol": obs.symbol,
        "fundamentals": {
            "pe_ratio": obs.pe_ratio,
            "price_to_book": obs.price_to_book,
            "return_on_equity": obs.return_on_equity,
            "debt_to_equity": obs.debt_to_equity,
            "profit_margin": obs.profit_margin,
            "revenue_growth": obs.revenue_growth,
            "fundamental_quality_score": obs.fundamental_quality_score,
            "sources": (fundamentals_meta or {}).get("sources", (info.get("fundamentals") or {}).get("sources", [])),
        },
        "sentiment": {
            "score": obs.news_sentiment_score,
            "confidence": obs.news_sentiment_confidence,
            "source": (sentiment_meta or {}).get("source", (info.get("news_sentiment") or {}).get("source", "neutral_fallback")),
        },
    }


def main() -> None:
    """Entry-point used by OpenEnv validators and local execution."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
