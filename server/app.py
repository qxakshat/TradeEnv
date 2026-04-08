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
    from .tradeenv_environment import TASKS, TradeEnvEnvironment
    from .fundamentals import load_or_fetch_fundamentals
except ImportError:
    from models import TradeEnvAction, TradeEnvObservation
    from server.tradeenv_environment import TASKS, TradeEnvEnvironment
    from server.fundamentals import load_or_fetch_fundamentals

import time
import threading
import pathlib
from uuid import uuid4
from functools import lru_cache

import yfinance as yf
import pandas as pd
from fastapi import Body, Query
from fastapi.responses import RedirectResponse
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

PAPER_LOCK = threading.Lock()
PAPER_ACCOUNT = {
    "cash": 2_000_000.0,
    "start_cash": 2_000_000.0,
    "positions": {},  # symbol -> quantity
    "trades": [],
}

HUMAN_LOCK = threading.Lock()
HUMAN_SESSIONS: dict[str, TradeEnvEnvironment] = {}
HUMAN_SYMBOLS: list[str] = sorted({s for task in TASKS for s in task.symbols})


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
    raw = pd.DataFrame()
    try:
        raw = yf.download(
            tickers=NIFTY50_DASHBOARD,
            period=f"{max(days, 5)}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )
    except Exception:
        raw = pd.DataFrame()

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

    if not symbols_payload:
        csv_path = pathlib.Path(__file__).parent.parent / "data" / "nifty50_market_data.csv"
        if csv_path.exists():
            try:
                local = pd.read_csv(csv_path)
                if {"symbol", "date", "close"}.issubset(local.columns):
                    local = local.copy()
                    local["date"] = pd.to_datetime(local["date"], errors="coerce")
                    local = local.dropna(subset=["date"]).sort_values(["symbol", "date"])
                    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=max(days, 5))
                    recent = local[local["date"] >= cutoff]
                    if recent.empty:
                        recent = local

                    for sym in NIFTY50_DASHBOARD:
                        sdf = recent[recent["symbol"] == sym]
                        if sdf.empty:
                            continue
                        close = sdf["close"].dropna()
                        if close.empty:
                            continue
                        last = _safe_float(close.iloc[-1], 0.0)
                        prev = _safe_float(close.iloc[-2], last) if len(close) >= 2 else last
                        chg = last - prev
                        pct = (chg / max(prev, 1e-9)) * 100.0
                        volume_col = "volume" if "volume" in sdf.columns else None
                        volume = int(_safe_float(sdf[volume_col].iloc[-1], 0.0)) if volume_col else 0
                        symbols_payload.append(
                            {
                                "symbol": sym,
                                "price": round(last, 4),
                                "change": round(chg, 4),
                                "change_pct": round(pct, 4),
                                "volume": volume,
                            }
                        )
            except Exception:
                pass

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


def _latest_price(symbol: str) -> float:
    snap = _get_market_snapshot(30)
    for row in snap.get("symbols", []):
        if row["symbol"].upper() == symbol.upper():
            return float(row["price"])
    return 0.0


def _serialize_observation(obs: TradeEnvObservation) -> dict:
    to_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
    return {
        "task_name": to_dict.get("task_name"),
        "role": to_dict.get("role"),
        "symbol": to_dict.get("symbol"),
        "step_index": to_dict.get("step_index"),
        "max_steps": to_dict.get("max_steps"),
        "current_price": to_dict.get("current_price"),
        "target_side": to_dict.get("target_side"),
        "target_quantity": to_dict.get("target_quantity"),
        "remaining_quantity": to_dict.get("remaining_quantity"),
        "executed_quantity": to_dict.get("executed_quantity"),
        "trades_used": to_dict.get("trades_used"),
        "max_trades_per_day": to_dict.get("max_trades_per_day"),
        "reward": to_dict.get("reward", 0.0),
        "done": to_dict.get("done", False),
    }


def _get_or_create_human_env(session_id: str | None = None) -> tuple[str, TradeEnvEnvironment]:
    with HUMAN_LOCK:
        if session_id and session_id in HUMAN_SESSIONS:
            return session_id, HUMAN_SESSIONS[session_id]
        new_id = session_id or str(uuid4())
        env = TradeEnvEnvironment()
        HUMAN_SESSIONS[new_id] = env
        return new_id, env


def _reset_human_env_to_symbol(env: TradeEnvEnvironment, symbol: str | None = None) -> TradeEnvObservation:
    target = (symbol or "").upper().strip()
    if target and target not in HUMAN_SYMBOLS:
        target = ""

    obs = env.reset()
    if not target:
        return obs

    max_tries = max(1, len(TASKS) * len(HUMAN_SYMBOLS))
    for _ in range(max_tries):
        if (obs.symbol or "").upper() == target:
            return obs
        obs = env.reset()
    return obs


@lru_cache(maxsize=64)
def _cached_symbol_fundamentals(cache_day: int, symbol: str) -> dict:
    del cache_day
    normalized = symbol.upper().strip()
    if normalized not in NIFTY50_DASHBOARD:
        normalized = "RELIANCE.NS"

    snapshot = load_or_fetch_fundamentals(
        NIFTY50_DASHBOARD,
        cache_path=pathlib.Path(__file__).parent.parent / "data" / "fundamentals_snapshot.csv",
    )
    table = snapshot.table
    row_df = table[table["symbol"] == normalized]

    if row_df.empty:
        return {
            "symbol": normalized,
            "pe_ratio": 0.0,
            "price_to_book": 0.0,
            "return_on_equity": 0.0,
            "debt_to_equity": 0.0,
            "profit_margin": 0.0,
            "revenue_growth": 0.0,
            "fundamental_quality_score": 0.0,
            "sources": snapshot.source_urls.get(normalized, []),
        }

    row = row_df.iloc[0]
    return {
        "symbol": normalized,
        "pe_ratio": round(_safe_float(row.get("pe_ratio", 0.0), 0.0), 4),
        "price_to_book": round(_safe_float(row.get("price_to_book", 0.0), 0.0), 4),
        "return_on_equity": round(_safe_float(row.get("return_on_equity", 0.0), 0.0), 4),
        "debt_to_equity": round(_safe_float(row.get("debt_to_equity", 0.0), 0.0), 4),
        "profit_margin": round(_safe_float(row.get("profit_margin", 0.0), 0.0), 4),
        "revenue_growth": round(_safe_float(row.get("revenue_growth", 0.0), 0.0), 4),
        "fundamental_quality_score": round(_safe_float(row.get("fundamental_quality_score", 0.0), 0.0), 4),
        "sources": snapshot.source_urls.get(normalized, []),
    }


def _get_symbol_fundamentals(symbol: str) -> dict:
    day_bucket = int(time.time() // (24 * 3600))
    return _cached_symbol_fundamentals(day_bucket, symbol)


@app.get("/healthz")
def healthz() -> dict:
    """Simple readiness endpoint for HF Spaces and container probes."""
    return {"status": "ok", "env": "TradeEnv"}


@app.get("/")
def root_redirect() -> RedirectResponse:
    """Ensure root always lands on the custom dashboard UI."""
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    """Light professional dashboard with market intelligence panels."""
    return """
<!doctype html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
    <title>TradeEnv Professional Dashboard</title>
    <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
    <style>
        :root {
            --bg:#f6f8fb; --panel:#ffffff; --line:#dfe5ef; --text:#1f2a44; --muted:#6d7b97;
            --pos:#148a57; --neg:#c7344d; --primary:#2f66d0; --warning:#9a6d00;
        }
        * { box-sizing:border-box; }
        body { margin:0; font-family:Inter,system-ui,Arial,sans-serif; background:var(--bg); color:var(--text); }
        .wrap { max-width:1380px; margin:0 auto; padding:12px; }
        .header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px; gap:10px; }
        .title { font-size:30px; font-weight:700; margin:0; }
        .subtitle { color:var(--muted); margin-top:4px; }
        .badge { display:inline-block; background:#fff7df; color:var(--warning); border:1px solid #f0ddb2; border-radius:999px; padding:5px 10px; font-size:12px; margin-right:6px; }
        .toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
        input,select,button { border-radius:10px; border:1px solid var(--line); padding:8px 10px; font-size:13px; background:#fff; color:var(--text); }
        button { background:var(--primary); color:#fff; border-color:var(--primary); cursor:pointer; }
        .grid { display:grid; grid-template-columns:repeat(12,1fr); gap:8px; }
        .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:10px; display:flex; flex-direction:column; gap:6px; height:100%; min-height:96px; }
        .kpi { grid-column:span 3; }
        .kpi .label { color:var(--muted); font-size:12px; }
        .kpi .value { margin-top:4px; font-size:21px; font-weight:700; }
        .pos{ color:var(--pos);} .neg{ color:var(--neg);} .muted{ color:var(--muted);}
        .wide{ grid-column:span 8;} .narrow{ grid-column:span 4;} .full{ grid-column:span 12; }
        .panel-title { font-weight:600; margin-bottom:2px; line-height:1.25; }
        .small { font-size:12px; color:var(--muted); line-height:1.4; }
        table { width:100%; border-collapse:collapse; font-size:13px; }
        th,td { padding:8px; border-bottom:1px solid var(--line); text-align:right; }
        th:first-child,td:first-child { text-align:left; }
        .toolbar input,.toolbar select,.toolbar button { width:100%; min-height:34px; }
        .news-list { margin:0; padding-left:16px; display:grid; gap:6px; }
        .news-list li { color:var(--text); line-height:1.35; }
        .readme-list { margin:0; padding-left:18px; display:grid; gap:6px; }
        .readme-list code { background:#f1f5fb; padding:1px 5px; border-radius:6px; }
        @media (max-width: 1100px){
            .kpi,.wide,.narrow,.full{ grid-column:span 12; }
            .toolbar{ width:100%; }
        }
        @media (max-width: 700px){
            .wrap { padding:10px; }
            .title { font-size:22px; }
            .subtitle { font-size:13px; }
            .kpi .value { font-size:18px; }
            input,select,button { padding:7px 8px; font-size:12px; }
        }
    </style>
</head>
<body>
    <div class=\"wrap\">
        <div class=\"header\">
            <div>
                <h1 class=\"title\">TradeEnv Professional NIFTY-50 Dashboard</h1>
                <div class=\"subtitle\">Live market data + OpenEnv controls in one compact workspace.</div>
                <div style=\"margin-top:8px\">
                    <span class=\"badge\">Depth metrics are simulated in environment (not exchange L2 feed)</span>
                    <span class=\"badge\">Sentiment snippets are advisory and may be delayed</span>
                </div>
            </div>
            <div class=\"toolbar\">
                <label>Lookback</label>
                <input id=\"days\" type=\"number\" min=\"5\" max=\"180\" value=\"45\" />
                <label>Ticker</label>
                <select id=\"symbol\"></select>
                <button onclick=\"refreshAll()\">Refresh</button>
            </div>
        </div>

        <div class=\"grid\">
            <div class=\"card kpi\"><div class=\"label\">Selected Ticker</div><div id=\"kpiSymbol\" class=\"value\">-</div></div>
            <div class=\"card kpi\"><div class=\"label\">Last Price</div><div id=\"kpiPrice\" class=\"value\">-</div></div>
            <div class=\"card kpi\"><div class=\"label\">Market Breadth (A/D)</div><div id=\"kpiBreadth\" class=\"value\">-</div></div>
            <div class=\"card kpi\"><div class=\"label\">Sentiment Score</div><div id=\"kpiSent\" class=\"value\">-</div></div>

            <div class=\"card full\"><div class=\"panel-title\">Candlestick + SMA (Real Yahoo daily candles)</div><div id=\"priceChart\" style=\"height:320px\"></div></div>

            <div class=\"card wide\"><div class=\"panel-title\">Top Movers</div><div id=\"heatChart\" style=\"height:300px\"></div></div>
            <div class=\"card narrow\"><div class=\"panel-title\">Depth Proxy (Simulated)</div><div id=\"depthChart\" style=\"height:300px\"></div></div>

            <div class=\"card full\">
                <div class=\"panel-title\">Sentiment News Snippets</div>
                <div class=\"small\" id=\"newsMeta\">Loading headlines…</div>
                <ul class=\"news-list\" id=\"newsList\"></ul>
            </div>

            <div class=\"card full\">
                <div class=\"panel-title\">Agent Interactions README</div>
                <div class=\"small\">How to use agent APIs programmatically from this dashboard backend.</div>
                <ul class=\"readme-list\">
                    <li><b>Discover contract:</b> call <code>GET /api/agent/specs</code> for endpoint and strategy metadata.</li>
                    <li><b>Get suggestion:</b> call <code>POST /api/agent/suggest</code> with payload <code>{\"symbol\": \"RELIANCE.NS\"}</code>.</li>
                    <li><b>Use market context:</b> combine <code>/api/market/overview</code>, <code>/api/market/history/{symbol}</code>, and <code>/api/market/depth/{symbol}</code>.</li>
                    <li><b>Execution loop:</b> map agent output to your own executor or environment runner (no UI-side trade execution required).</li>
                </ul>
            </div>

            <div class=\"card full\">
                <div class=\"panel-title\">NIFTY-50 Watchlist Snapshot (Real)</div>
                <table id=\"watchTable\"><thead><tr><th>Symbol</th><th>Price</th><th>Δ</th><th>Δ%</th><th>Volume</th></tr></thead><tbody></tbody></table>
            </div>

        </div>
    </div>

    <script>
        function fmt(n){ return new Intl.NumberFormat('en-IN',{maximumFractionDigits:2}).format(n||0); }
        async function fetchJSON(url, opts){ const r=await fetch(url, opts); if(!r.ok) throw new Error(await r.text()); return await r.json(); }
        function axisStyle(){ return {gridcolor:'#edf1f7', linecolor:'#dbe3f0'}; }

        async function loadOverview(days){
            const data = await fetchJSON(`/api/market/overview?days=${days}`);
            const sel = document.getElementById('symbol');
            const cur = sel.value;
            sel.innerHTML = data.symbols.map(s=>`<option value=\"${s.symbol}\">${s.symbol}</option>`).join('');
            if(cur && data.symbols.some(x=>x.symbol===cur)) sel.value = cur;
            document.getElementById('kpiBreadth').textContent = `${data.advancers} / ${data.decliners}`;

            const activeSym = sel.value || 'RELIANCE.NS';
            const activeRow = data.symbols.find(x => x.symbol === activeSym);
            document.getElementById('kpiSymbol').textContent = activeSym.replace('.NS','');
            const priceEl = document.getElementById('kpiPrice');
            priceEl.textContent = activeRow ? `₹ ${fmt(activeRow.price)}` : '-';
            priceEl.className = 'value ' + ((activeRow?.change_pct || 0) >= 0 ? 'pos' : 'neg');

            const body = document.querySelector('#watchTable tbody');
            body.innerHTML = data.symbols.slice(0,20).map(s=>`<tr>
                <td>${s.symbol}</td><td>${fmt(s.price)}</td>
                <td class=\"${s.change>=0?'pos':'neg'}\">${fmt(s.change)}</td>
                <td class=\"${s.change_pct>=0?'pos':'neg'}\">${fmt(s.change_pct)}%</td>
                <td>${fmt(s.volume)}</td></tr>`).join('');

            const x = data.symbols.slice(0,20).map(s=>s.symbol.replace('.NS',''));
            const y = data.symbols.slice(0,20).map(s=>s.change_pct);
            Plotly.newPlot('heatChart', [{type:'bar', x, y, marker:{color:y.map(v=>v>=0?'#1b9e63':'#d6455d')}}], {
                margin:{t:40,l:40,r:10,b:50}, paper_bgcolor:'#fff', plot_bgcolor:'#fff', xaxis:axisStyle(), yaxis:axisStyle()
            }, {displayModeBar:false});
        }

        async function loadHistory(symbol, days){
            const h = await fetchJSON(`/api/market/history/${symbol}?days=${days}`);
            Plotly.newPlot('priceChart', [
                {type:'candlestick', x:h.dates, open:h.open, high:h.high, low:h.low, close:h.close, name:'OHLC'},
                {type:'scatter', mode:'lines', x:h.dates, y:h.sma7, name:'SMA7', line:{color:'#2f66d0', width:1.8}},
            ], {
                margin:{t:36,l:45,r:10,b:40}, paper_bgcolor:'#fff', plot_bgcolor:'#fff',
                xaxis:axisStyle(), yaxis:axisStyle(), showlegend:true, legend:{orientation:'h'}
            }, {displayModeBar:false});
        }

        async function loadSignals(symbol){
            const d = await fetchJSON(`/api/market/depth/${symbol}`);
            Plotly.newPlot('depthChart', [{type:'bar', x:['BidDepth','AskDepth','Spread(bps)'], y:[d.bid_depth_top,d.ask_depth_top,d.spread_bps], marker:{color:['#1b9e63','#d6455d','#2f66d0']}}], {
                margin:{t:36,l:40,r:10,b:40}, paper_bgcolor:'#fff', plot_bgcolor:'#fff', xaxis:axisStyle(), yaxis:axisStyle()
            }, {displayModeBar:false});
        }

        async function loadNewsSnippets(){
            const snap = await fetchJSON('/dashboard-snapshot');
            const s = snap.sentiment || {};
            const headlines = Array.isArray(s.headlines) ? s.headlines : [];
            const source = s.source || 'neutral_fallback';
            const score = Number(s.score || 0);
            const conf = Number(s.confidence || 0);
            document.getElementById('newsMeta').textContent = `source=${source} | score=${score.toFixed(3)} | confidence=${conf.toFixed(3)}`;
            const sentEl = document.getElementById('kpiSent');
            sentEl.textContent = score.toFixed(3);
            sentEl.className = 'value ' + (score >= 0 ? 'pos' : 'neg');

            const list = document.getElementById('newsList');
            if(!headlines.length){
                list.innerHTML = '<li>No recent headlines available from sentiment feeds.</li>';
                return;
            }
            list.innerHTML = headlines.slice(0,6).map(h => `<li>${h}</li>`).join('');
        }

        async function refreshAll(){
            const days = Number(document.getElementById('days').value || 45);
            await loadOverview(days);
            const symbol = document.getElementById('symbol').value || 'RELIANCE.NS';
            await Promise.all([loadHistory(symbol, days), loadSignals(symbol), loadNewsSnippets()]);
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
        return {"symbol": s, "dates": [], "open": [], "high": [], "low": [], "close": [], "sma7": []}

    def pick_col(name: str):
        obj = raw[name]
        if hasattr(obj, "columns"):
            return obj.iloc[:, 0].astype(float)
        return obj.astype(float)

    o = pick_col("Open")
    h = pick_col("High")
    l = pick_col("Low")
    c = pick_col("Close")
    sma7 = c.rolling(7, min_periods=1).mean()
    dates = [d.strftime("%Y-%m-%d") for d in c.index]
    return {
        "symbol": s,
        "dates": dates,
        "open": [round(float(x), 4) for x in o.tolist()],
        "high": [round(float(x), 4) for x in h.tolist()],
        "low": [round(float(x), 4) for x in l.tolist()],
        "close": [round(float(x), 4) for x in c.tolist()],
        "sma7": [round(float(x), 4) for x in sma7.tolist()],
    }


@app.get("/api/market/fundamentals/{symbol}")
def api_market_fundamentals(symbol: str) -> dict:
    return _get_symbol_fundamentals(symbol)


@app.get("/api/portfolio/demo")
def api_portfolio_demo(days: int = Query(default=30, ge=5, le=180)) -> dict:
    snap = _get_market_snapshot(days)
    return _build_demo_portfolio(snap)


@app.get("/api/market/depth/{symbol}")
def api_market_depth(symbol: str) -> dict:
    """Depth proxy from environment simulation for UI diagnostics."""
    env = TradeEnvEnvironment()
    obs = env.reset()
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


@app.post("/api/human/session/new")
def api_human_session_new() -> dict:
    session_id, env = _get_or_create_human_env()
    obs = env.reset()
    return {
        "session_id": session_id,
        "observation": _serialize_observation(obs),
        "reward": float(obs.reward or 0.0),
        "done": bool(obs.done),
    }


@app.post("/api/human/reset")
def api_human_reset(payload: dict = Body(...)) -> dict:
    session_id = str(payload.get("session_id", "")).strip()
    symbol = str(payload.get("symbol", "")).strip().upper()
    if not session_id:
        raise ValueError("session_id is required")
    _, env = _get_or_create_human_env(session_id)
    obs = _reset_human_env_to_symbol(env, symbol)
    return {
        "session_id": session_id,
        "observation": _serialize_observation(obs),
        "reward": float(obs.reward or 0.0),
        "done": bool(obs.done),
    }


@app.post("/api/human/step")
def api_human_step(payload: dict = Body(...)) -> dict:
    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        raise ValueError("session_id is required")
    _, env = _get_or_create_human_env(session_id)
    action = TradeEnvAction(
        action_type=str(payload.get("action_type", "hold")),
        quantity=int(payload.get("quantity", 0)),
        urgency=float(payload.get("urgency", 0.5)),
    )
    obs = env.step(action)
    return {
        "session_id": session_id,
        "observation": _serialize_observation(obs),
        "reward": float(obs.reward or 0.0),
        "done": bool(obs.done),
    }


@app.get("/api/human/state")
def api_human_state(session_id: str = Query(...)) -> dict:
    sid = session_id.strip()
    if not sid:
        raise ValueError("session_id is required")
    _, env = _get_or_create_human_env(sid)
    return {
        "session_id": sid,
        "episode_id": env.state.episode_id,
        "step_count": env.state.step_count,
    }


@app.post("/api/paper/reset")
def api_paper_reset() -> dict:
    with PAPER_LOCK:
        PAPER_ACCOUNT["cash"] = PAPER_ACCOUNT["start_cash"]
        PAPER_ACCOUNT["positions"] = {}
        PAPER_ACCOUNT["trades"] = []
    return {"status": "ok"}


@app.get("/api/paper/state")
def api_paper_state() -> dict:
    with PAPER_LOCK:
        positions = dict(PAPER_ACCOUNT["positions"])
        trades = list(PAPER_ACCOUNT["trades"])
        cash = float(PAPER_ACCOUNT["cash"])
        start_cash = float(PAPER_ACCOUNT["start_cash"])

    holdings = []
    holdings_value = 0.0
    for sym, qty in positions.items():
        if qty <= 0:
            continue
        px = _latest_price(sym)
        val = qty * px
        holdings_value += val
        holdings.append({"symbol": sym, "qty": qty, "ltp": round(px, 4), "value": round(val, 2)})

    portfolio_value = cash + holdings_value
    unrealized_pnl = portfolio_value - start_cash
    weights = [
        {"symbol": h["symbol"], "weight": round((h["value"] / max(holdings_value, 1e-9)) * 100.0, 2)}
        for h in holdings
    ]
    return {
        "cash": round(cash, 2),
        "portfolio_value": round(portfolio_value, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "holdings": holdings,
        "weights": weights,
        "trades": trades,
        "is_simulated": True,
        "notice": "Paper trades are virtual and use last snapshot prices (non-exchange execution).",
    }


@app.post("/api/paper/trade")
def api_paper_trade(payload: dict = Body(...)) -> dict:
    symbol = str(payload.get("symbol", "RELIANCE.NS")).upper()
    side = str(payload.get("side", "buy")).lower()
    qty = int(max(1, int(payload.get("qty", 1))))
    if symbol not in NIFTY50_DASHBOARD:
        symbol = "RELIANCE.NS"
    if side not in {"buy", "sell"}:
        side = "buy"

    price = _latest_price(symbol)
    if price <= 0:
        raise ValueError("Price unavailable for selected symbol")

    with PAPER_LOCK:
        positions = PAPER_ACCOUNT["positions"]
        cash = float(PAPER_ACCOUNT["cash"])
        notional = qty * price
        if side == "buy":
            if notional > cash:
                qty = int(cash // max(price, 1e-9))
                notional = qty * price
            if qty <= 0:
                return {"status": "rejected", "reason": "insufficient_cash"}
            PAPER_ACCOUNT["cash"] = cash - notional
            positions[symbol] = int(positions.get(symbol, 0)) + qty
        else:
            held = int(positions.get(symbol, 0))
            if held <= 0:
                return {"status": "rejected", "reason": "no_position"}
            qty = min(qty, held)
            notional = qty * price
            PAPER_ACCOUNT["cash"] = cash + notional
            positions[symbol] = held - qty
            if positions[symbol] <= 0:
                positions.pop(symbol, None)

        trade = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": round(price, 4),
            "notional": round(notional, 2),
        }
        PAPER_ACCOUNT["trades"].append(trade)
    return {"status": "ok", "trade": trade}


@app.get("/api/agent/specs")
def api_agent_specs() -> dict:
    return {
        "name": "TradeEnvAgentAssist",
        "contract": "POST /api/agent/suggest with {symbol} -> returns side/qty/reason for paper trading",
        "strategies": ["trend_follow", "mean_reversion", "depth_aware"],
        "endpoints": [
            "GET /api/agent/specs",
            "POST /api/agent/suggest",
            "GET /api/paper/state",
            "POST /api/paper/trade",
            "POST /api/paper/reset",
        ],
    }


@app.post("/api/agent/suggest")
def api_agent_suggest(payload: dict = Body(...)) -> dict:
    symbol = str(payload.get("symbol", "RELIANCE.NS")).upper()
    if symbol not in NIFTY50_DASHBOARD:
        symbol = "RELIANCE.NS"
    hist = api_market_history(symbol=symbol, days=30)
    if not hist.get("close"):
        return {"symbol": symbol, "action": {"side": "buy", "qty": 1}, "reason": "fallback_no_history"}

    close = hist["close"]
    sma7 = hist["sma7"]
    last = close[-1]
    ma = sma7[-1]
    if last >= ma:
        side, qty, reason = "buy", 12, "trend_follow: close above SMA7"
    else:
        side, qty, reason = "sell", 10, "mean_reversion: close below SMA7"
    return {"symbol": symbol, "action": {"side": side, "qty": qty}, "reason": reason}


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
            "headlines": (sentiment_meta or {}).get("headlines", (info.get("news_sentiment") or {}).get("headlines", [])),
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
