"""Local dashboard: fund the wallet, watch the bot, and pull the brakes.

Served on 127.0.0.1 by default. It shows the trading wallet's PUBLIC address,
a QR code and a "Deposit with Phantom" button that builds a plain SOL transfer
for the user's own wallet to sign. The private key never leaves the trader
process and is never sent to the page. Withdrawals are a CLI command
(`ai_trader withdraw`), not a button, so a stray click or a hostile page can't
move funds out.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from ai_trader.config import RiskSettings
from utils.logger import get_logger

if TYPE_CHECKING:
    from ai_trader.config import AITraderSettings
    from ai_trader.engine import Engine
    from ai_trader.feed import LiveFeed

logger = get_logger(__name__)

_PAGE = Path(__file__).with_name("dashboard.html")
#: Risk settings the dashboard may change while the bot runs.
_EDITABLE = (
    "max_position_sol",
    "max_position_pct",
    "max_open_positions",
    "max_total_exposure_sol",
    "max_daily_loss_sol",
    "max_trades_per_day",
    "max_consecutive_losses",
    "reserve_sol",
)


def wallet_address() -> str | None:
    """The public address of SOLANA_PRIVATE_KEY, if set. The key itself stays here."""
    key = os.environ.get("SOLANA_PRIVATE_KEY")
    if not key:
        return None
    try:
        from core.wallet import Wallet  # noqa: PLC0415

        return str(Wallet(key).pubkey)
    except Exception:  # noqa: BLE001 - a bad key just means no address to show
        return None


class SolPrice:
    """SOL/USD from public exchange tickers via ccxt, cached for a minute."""

    def __init__(self) -> None:
        self.value: float | None = None
        self.at = 0.0

    async def get(self) -> float | None:
        if time.time() - self.at < 60:
            return self.value
        self.at = time.time()
        try:
            import ccxt.async_support as ccxt  # noqa: PLC0415
        except ImportError:
            return None
        for name, symbol in (
            ("kraken", "SOL/USD"),
            ("coinbase", "SOL/USD"),
            ("binance", "SOL/USDT"),
        ):
            ex = getattr(ccxt, name)()
            try:
                ticker = await asyncio.wait_for(ex.fetch_ticker(symbol), 8)
                self.value = float(ticker["last"])
                break
            except Exception:  # noqa: BLE001,S112 - try the next exchange
                continue
            finally:
                await ex.close()
        return self.value


def build_app(
    settings: AITraderSettings, engine: Engine, feed: LiveFeed | None
) -> web.Application:
    sol_price = SolPrice()
    address = getattr(engine.broker, "address", None) or wallet_address()
    rpc_client = None

    @web.middleware
    async def auth(request: web.Request, handler: Any) -> web.StreamResponse:
        token = settings.gui.access_token
        if (
            token
            and request.query.get("token") != token
            and request.headers.get("X-Token") != token
        ):
            raise web.HTTPUnauthorized(text="missing or wrong ?token=")
        if request.method == "POST":
            # Same-origin only: another site open in the browser can't press buttons here.
            origin = request.headers.get("Origin")
            if origin and origin.split("://", 1)[-1] != request.host:
                raise web.HTTPForbidden(text="cross-origin request refused")
        return await handler(request)

    async def get_rpc() -> Any:
        nonlocal rpc_client
        if rpc_client is None and settings.feed.rpc_endpoint.startswith("http"):
            from solana.rpc.async_api import AsyncClient  # noqa: PLC0415

            rpc_client = AsyncClient(settings.feed.rpc_endpoint)
        return rpc_client

    async def wallet_balance() -> float | None:
        if not address:
            return None
        client = await get_rpc()
        if client is None:
            return None
        try:
            from solders.pubkey import Pubkey  # noqa: PLC0415

            resp = await asyncio.wait_for(
                client.get_balance(Pubkey.from_string(address)), 8
            )
            return resp.value / 1e9
        except Exception:  # noqa: BLE001 - shown as unknown
            return None

    async def page(_: web.Request) -> web.Response:
        return web.Response(
            text=_PAGE.read_text(encoding="utf-8"), content_type="text/html"
        )

    async def status(_: web.Request) -> web.Response:
        now = time.time()
        r = engine.risk
        bal, usd = await asyncio.gather(wallet_balance(), sol_price.get())
        body = {
            "name": settings.name,
            "mode": engine.mode,
            "run_id": engine.run_id,
            "uptime_s": now - engine.started_at,
            "address": address,
            "wallet_balance_sol": bal,
            "sol_usd": usd,
            "equity": engine.equity(),
            "start_bankroll_sol": settings.risk.paper_bankroll_sol
            if engine.mode == "paper"
            else None,
            "feed": {
                "demo": feed is None,
                "connected": bool(feed and feed.connected),
                "last_event_age_s": now - feed.last_event_at
                if feed and feed.last_event_at
                else None,
            },
            "stats": engine.stats,
            "social": None
            if engine.social is None
            else {
                **engine.social.stats,
                "dex_ok": engine.social.dex_ok,
                "dex_trending_words": sorted(
                    engine.social.dex_words, key=engine.social.dex_words.get, reverse=True
                )[:15],
                "x_enabled": bool(settings.social.x_bearer_token),
            },
            "risk": {
                "paused": r.state.paused,
                "killed": r.kill_requested(),
                "realized_today_sol": r.state.realized_today,
                "trades_today": r.state.trades_today,
                "cooldown_left_s": max(0.0, r.state.cooldown_until - now),
                "halted_reason": r.state.halted_reason,
                "limits": {k: getattr(r.s, k) for k in _EDITABLE},
            },
            "strategy": settings.strategy.model_dump(),
            "llm": {
                "enabled": engine.scorer.claude is not None,
                "model": settings.llm.model,
            },
            "positions": engine.position_rows(),
            "decisions": engine.recent_decisions[:25],
        }
        return web.json_response(
            body, dumps=lambda o: __import__("json").dumps(o, default=str)
        )

    async def equity(_: web.Request) -> web.Response:
        df = engine.store.frame(
            "SELECT ts, equity_sol FROM equity WHERE run_id = ? ORDER BY ts",
            [engine.run_id],
        )
        pts: dict[int, float] = {int(r["ts"]): r["equity_sol"] for r in df.to_dicts()}
        return web.json_response(
            [{"time": t, "value": v} for t, v in sorted(pts.items())]
        )

    async def trades(_: web.Request) -> web.Response:
        df = engine.store.frame(
            "SELECT ts, symbol, mint, side, sol, price, pnl_sol, reason, signature FROM trades "
            "WHERE run_id = ? ORDER BY ts DESC LIMIT 200",
            [engine.run_id],
        )
        return web.json_response(df.to_dicts())

    async def blockhash(_: web.Request) -> web.Response:
        client = await get_rpc()
        if client is None:
            raise web.HTTPServiceUnavailable(text="no RPC endpoint configured")
        resp = await client.get_latest_blockhash()
        return web.json_response({"blockhash": str(resp.value.blockhash)})

    async def pause(_: web.Request) -> web.Response:
        engine.risk.state.paused = True
        return web.json_response({"ok": True})

    async def resume(_: web.Request) -> web.Response:
        engine.risk.state.paused = False
        return web.json_response({"ok": True})

    async def kill(_: web.Request) -> web.Response:
        settings.kill_file.parent.mkdir(parents=True, exist_ok=True)
        settings.kill_file.write_text(f"killed from dashboard at {time.ctime()}\n")
        engine.risk.state.killed = True
        await engine.on_tick(time.time())  # starts selling everything now
        return web.json_response({"ok": True})

    async def unkill(_: web.Request) -> web.Response:
        settings.kill_file.unlink(missing_ok=True)
        engine.risk.state.killed = False
        return web.json_response({"ok": True})

    async def limits(request: web.Request) -> web.Response:
        body = await request.json()
        current = engine.risk.s.model_dump()
        for k, v in body.items():
            if k in _EDITABLE:
                current[k] = v
        try:
            new = RiskSettings.model_validate(current)
        except Exception as e:
            raise web.HTTPBadRequest(text=str(e)) from e
        engine.risk.s = new
        logger.warning(
            f"Risk limits changed from the dashboard: { {k: body[k] for k in body if k in _EDITABLE} }"
        )
        return web.json_response({"ok": True})

    app = web.Application(middlewares=[auth])
    app.add_routes(
        [
            web.get("/", page),
            web.get("/api/status", status),
            web.get("/api/equity", equity),
            web.get("/api/trades", trades),
            web.get("/api/blockhash", blockhash),
            web.post("/api/pause", pause),
            web.post("/api/resume", resume),
            web.post("/api/kill", kill),
            web.post("/api/unkill", unkill),
            web.post("/api/limits", limits),
        ]
    )
    return app


async def start_gui(
    settings: AITraderSettings, engine: Engine, feed: LiveFeed | None
) -> web.AppRunner:
    app = build_app(settings, engine, feed)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, settings.gui.host, settings.gui.port)
    await site.start()
    url = f"http://{settings.gui.host}:{settings.gui.port}/"
    if settings.gui.access_token:
        url += "?token=<your gui.access_token>"
    logger.info(f"Dashboard: {url}")
    if (
        settings.gui.host not in ("127.0.0.1", "localhost")
        and not settings.gui.access_token
    ):
        logger.warning(
            "Dashboard is reachable from the network with no access_token set"
        )
    return runner
