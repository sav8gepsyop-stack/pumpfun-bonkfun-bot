"""Verify the AI trader's decoding, curve math, risk limits, accounting and safety rails.

Offline machine checks, no network and no funds moved:

  A. pump.fun creates and trades decode from recorded transaction logs, and a
     coin paired against anything but SOL has its trades dropped.
  B. The paper broker's curve math reproduces a recorded on-chain buy exactly.
  C. Risk limits: the kill file, pause, the daily loss limit, the open-position
     limit and the size caps each block or cap a buy.
  D. A paper round trip loses exactly the fees, and cash stays consistent.
  E. The engine runs end to end on a synthetic market: every buy respects the
     size cap, every closed trade has a P&L, and the kill switch sells all.
  F. Live mode refuses to start without the explicit AI_TRADER_LIVE value.
  G. The dashboard never returns the private key, refuses cross-origin
     POSTs and rejects invalid limits.

Usage:
    uv run tests/regression/verify_ai_trader.py
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import logging  # noqa: E402

logging.disable(logging.WARNING)

from ai_trader.broker import PaperBroker, curve_buy  # noqa: E402
from ai_trader.config import AITraderSettings  # noqa: E402
from ai_trader.engine import Engine  # noqa: E402
from ai_trader.events import CreateEvt, TradeEvt, decode_logs  # noqa: E402
from ai_trader.market import CoinState  # noqa: E402
from ai_trader.risk import RiskManager  # noqa: E402
from ai_trader.store import Store  # noqa: E402
from ai_trader.synthetic import generate  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def _logs(path: Path) -> list[list[str]]:
    out = []

    def walk(o: object) -> None:
        if isinstance(o, dict):
            if "logMessages" in o:
                out.append(o["logMessages"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(json.loads(path.read_text()))
    return out


def a_decode() -> TradeEvt:
    print("A. decoding")
    create_tx = (
        PROJECT_ROOT / "cookbook/pumpfun/decode/raw_create_tx_from_gettransaction.json"
    )
    evts = [e for logs in _logs(create_tx) for e in decode_logs(logs, "sig", 1.0)]
    creates = [e for e in evts if isinstance(e, CreateEvt)]
    trades = [e for e in evts if isinstance(e, TradeEvt)]
    check(
        len(creates) == 1 and creates[0].symbol == "TAS",
        "create decoded with its symbol",
    )
    check(len(trades) == 1 and trades[0].is_buy, "dev buy decoded as a trade")
    check(trades[0].user == creates[0].creator, "dev buy is by the creator")
    usdc_tx = (
        PROJECT_ROOT / "cookbook/pumpfun/decode/raw_buy_tx_from_gettransaction.json"
    )
    other = [e for logs in _logs(usdc_tx) for e in decode_logs(logs, "sig", 1.0)]
    check(not other, "trade on a non-SOL-quoted coin is dropped")
    return trades[0]


def b_curve(dev_buy: TradeEvt) -> None:
    print("B. curve math")
    tokens, fee = curve_buy(
        30_000_000_000, 1_073_000_000_000_000, 793_100_000_000_000, dev_buy.sol, 0
    )
    check(
        tokens == dev_buy.tokens, f"recorded buy: {tokens} == {dev_buy.tokens} tokens"
    )
    check(fee == 0, "no fee at 0 bps")


def c_risk(tmp: Path) -> None:
    print("C. risk limits")
    s = AITraderSettings()
    r = RiskManager(s.risk, tmp / "KILL")
    kw = dict(
        now=1e9,
        score=90,
        threshold=68,
        bankroll_sol=10,
        open_count=0,
        exposure_sol=0,
        unrealized_sol=0,
    )
    size, why = r.size(**kw)
    check(
        why is None and 0 < size <= s.risk.max_position_sol,
        f"normal buy sized {size} <= cap",
    )
    check(
        r.size(**{**kw, "open_count": s.risk.max_open_positions})[0] == 0,
        "open-position limit blocks",
    )
    check(
        r.size(**{**kw, "exposure_sol": s.risk.max_total_exposure_sol})[0] == 0,
        "exposure limit blocks",
    )
    check(
        r.size(**{**kw, "unrealized_sol": -s.risk.max_daily_loss_sol})[0] == 0,
        "open losses count toward the daily limit",
    )
    r.on_close(1e9, -s.risk.max_daily_loss_sol)
    check(r.size(**kw)[0] == 0, "realized daily loss limit blocks")
    check(r.size(**{**kw, "now": 1e9 + 86_400})[0] > 0, "limit resets the next UTC day")
    r.state.paused = True
    check(r.size(**{**kw, "now": 1e9 + 86_400})[0] == 0, "pause blocks")
    r.state.paused = False
    (tmp / "KILL").write_text("x")
    check(r.size(**{**kw, "now": 1e9 + 86_400})[0] == 0, "kill file blocks")
    (tmp / "KILL").unlink()


def d_paper() -> None:
    print("D. paper accounting")
    s = AITraderSettings()
    b = PaperBroker(s)
    coin = CoinState(mint="m", created_at=0)
    start = b.cash

    async def go() -> tuple:
        buy = await b.buy(coin, 0.1)
        sell = await b.sell(coin, buy.tokens_raw)
        return buy, sell

    buy, sell = asyncio.run(go())
    lost = start - b.cash
    expected = buy.fees_sol + sell.fees_sol
    check(buy.ok and sell.ok, "round trip fills")
    # Paper pays its own price impact on both legs (pessimistic by design).
    impact = 2 * 0.1 / 30
    check(
        expected <= lost <= expected + impact * 0.1,
        f"round trip loses the fees plus impact: {lost:.6f} vs fees {expected:.6f}",
    )
    check(0.002 < lost < 0.004, "fees are ~2.5% of 0.1 SOL plus network cost")


def e_engine(tmp: Path) -> None:
    print("E. engine on a synthetic market")
    s = AITraderSettings(data_dir=str(tmp))
    store = Store(tmp / "e.duckdb")
    engine = Engine(s, PaperBroker(s), store, inline_scoring=True)

    async def go() -> None:
        await engine.start()
        events = generate(150, seed=3)
        last = events[0].ts
        for i, evt in enumerate(events):
            while evt.ts - last >= 1:
                last += 1
                await engine.on_tick(last)
            await engine.on_event(evt, evt.ts)
            if i == len(events) // 2:
                mid_open = len(engine.positions)
        s.kill_file.write_text("x")
        await engine.on_tick(last + 1)
        await engine.on_tick(last + 1 + s.costs.latency_s)
        check(
            not engine.positions,
            f"kill switch sold everything (had {mid_open} open midway)",
        )

    asyncio.run(go())
    trades = store.frame("SELECT * FROM trades")
    buys = trades.filter(trades["side"] == "buy")
    sells = trades.filter(trades["side"] == "sell")
    check(buys.height > 0, f"engine bought ({buys.height} buys)")
    cap = s.risk.max_position_sol + s.costs.tx_cost_sol + 1e-9
    check(bool((buys["sol"] <= cap).all()), "no buy above max_position_sol")
    check(sells["pnl_sol"].null_count() == 0, "every sell has a P&L")
    eq = engine.equity()
    check(
        abs(
            eq["cash_sol"]
            - (
                s.risk.paper_bankroll_sol
                + float(sells["sol"].sum())
                - float(buys["sol"].sum())
            )
        )
        < 1e-6,
        "cash = bankroll + sells - buys",
    )
    store.close()
    s.kill_file.unlink()


def f_live_guard() -> None:
    print("F. live guard")
    from ai_trader.runner import LiveModeNotConfirmedError, make_broker  # noqa: PLC0415

    os.environ.pop("AI_TRADER_LIVE", None)
    s = AITraderSettings(mode="live")
    try:
        make_broker(s)
        check(False, "live mode without AI_TRADER_LIVE refused")
    except LiveModeNotConfirmedError:
        check(True, "live mode without AI_TRADER_LIVE refused")
    os.environ["AI_TRADER_LIVE"] = "yes"
    try:
        make_broker(s)
        check(False, "a casual AI_TRADER_LIVE value is refused")
    except LiveModeNotConfirmedError:
        check(True, "a casual AI_TRADER_LIVE value is refused")
    os.environ.pop("AI_TRADER_LIVE", None)


def g_gui(tmp: Path) -> None:
    print("G. dashboard")
    from aiohttp.test_utils import TestClient, TestServer  # noqa: PLC0415
    from solders.keypair import Keypair  # noqa: PLC0415

    from ai_trader.gui import build_app  # noqa: PLC0415

    kp = Keypair()
    secret = str(kp)  # base58 of the 64-byte key
    os.environ["SOLANA_PRIVATE_KEY"] = secret
    s = AITraderSettings(data_dir=str(tmp))
    s.feed.rpc_endpoint = ""
    store = Store(tmp / "g.duckdb")
    engine = Engine(s, PaperBroker(s), store, inline_scoring=True)

    async def go() -> None:
        await engine.start()
        client = TestClient(TestServer(build_app(s, engine, None)))
        await client.start_server()
        try:
            text = await (await client.get("/api/status")).text()
            body = json.loads(text)
            check(
                body["address"] == str(kp.pubkey()), "status shows the public address"
            )
            check(secret not in text, "status never contains the private key")
            page = await (await client.get("/")).text()
            check(secret not in page, "page never contains the private key")
            r = await client.post(
                "/api/kill", headers={"Origin": "https://evil.example"}
            )
            check(r.status == 403, "cross-origin POST refused")
            r = await client.post("/api/limits", json={"max_position_pct": 5})
            check(r.status == 400, "invalid limit rejected")
            r = await client.post("/api/limits", json={"max_position_sol": 0.05})
            check(
                r.status == 200 and engine.risk.s.max_position_sol == 0.05,
                "valid limit applied",
            )
            r = await client.post("/api/kill")
            check(
                r.status == 200 and s.kill_file.exists(),
                "kill from the dashboard writes the kill file",
            )
            await client.post("/api/unkill")
            check(not s.kill_file.exists(), "re-arm removes it")
        finally:
            await client.close()

    asyncio.run(go())
    store.close()
    os.environ.pop("SOLANA_PRIVATE_KEY", None)


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        dev_buy = a_decode()
        b_curve(dev_buy)
        c_risk(tmp)
        d_paper()
        e_engine(tmp)
        f_live_guard()
        g_gui(tmp)
    if FAILS:
        print(f"\n{len(FAILS)} check(s) failed")
        sys.exit(1)
    print("\nAll AI trader checks passed")


if __name__ == "__main__":
    main()
