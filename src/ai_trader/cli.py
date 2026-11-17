"""`ai_trader` command line: run, backtest, report, demo, kill, address, withdraw."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from ai_trader.config import AITraderSettings, load_settings
from utils.logger import (
    get_logger,
    install_secret_redaction,
    register_endpoint,
    register_secret,
    setup_file_logging,
)

logger = get_logger("ai_trader")
DEFAULT_CONFIG = "config/ai-trader.yaml"


def _settings(args: argparse.Namespace) -> AITraderSettings:
    path = args.config if args.config and Path(args.config).exists() else None
    if args.config and path is None and args.config != DEFAULT_CONFIG:
        sys.exit(f"config {args.config} not found")
    s = load_settings(path)
    for endpoint in (s.feed.rpc_endpoint, s.feed.wss_endpoint):
        if endpoint:
            register_endpoint(endpoint)
    for name in ("ANTHROPIC_API_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY"):
        if os.environ.get(name):
            register_secret(os.environ[name])
    install_secret_redaction()
    return s


def cmd_run(args: argparse.Namespace) -> int:
    from ai_trader.runner import LiveModeNotConfirmedError, run  # noqa: PLC0415

    s = _settings(args)
    if args.live:
        s.mode = "live"
        s.private_key = os.environ.get("SOLANA_PRIVATE_KEY") or None
    Path("logs").mkdir(exist_ok=True)
    setup_file_logging(f"logs/{s.name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    if not s.feed.wss_endpoint:
        sys.exit(
            "Set SOLANA_NODE_WSS_ENDPOINT in .env (a websocket RPC, e.g. from Chainstack)."
        )
    try:
        asyncio.run(run(s, gui=not args.no_gui))
    except LiveModeNotConfirmedError as e:
        sys.exit(str(e))
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from ai_trader.backtest import run_backtest  # noqa: PLC0415
    from ai_trader.feed import replay  # noqa: PLC0415

    s = _settings(args)
    s.llm.enabled = s.llm.enabled and args.llm
    if args.synthetic:
        from ai_trader.synthetic import generate  # noqa: PLC0415

        events = generate(args.synthetic, seed=args.seed)
        label = f"synthetic-{args.synthetic}"
        print(
            "NOTE: synthetic data is an invented market. It tests the plumbing; it says nothing about real profitability."
        )
    else:
        paths = (
            [Path(p) for p in args.data]
            if args.data
            else sorted((s.data_path / "feed").glob("*.jsonl*"))
        )
        if not paths:
            sys.exit(
                "No recordings found. Run `ai_trader run` (paper mode records the live feed to data/feed/) first."
            )
        events = replay(paths)
        label = "recorded"
    report, out = asyncio.run(run_backtest(s, events, label=label))
    _print_report(report)
    print(f"\nReport: {out['html']}\n        {out['md']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from ai_trader import report as rep  # noqa: PLC0415
    from ai_trader.store import Store  # noqa: PLC0415

    s = _settings(args)
    db = s.data_path / f"{s.name}.duckdb"
    if not db.exists():
        sys.exit(f"No data yet at {db}; run the trader first.")
    store = Store(db)
    run_id = args.run
    if not run_id:
        runs = store.frame(
            "SELECT run_id, min(ts) AS t FROM trades GROUP BY run_id ORDER BY t DESC LIMIT 1"
        )
        if runs.is_empty():
            sys.exit("No trades recorded yet.")
        run_id = runs["run_id"][0]
    start = s.risk.paper_bankroll_sol if run_id.startswith("paper") else args.start_sol
    result = rep.compute(store, run_id, start)
    out = rep.write(result, s.data_path / "reports" / run_id, f"{s.name} {run_id}")
    store.close()
    _print_report(result)
    print(f"\nReport: {out['html']}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """The dashboard on an invented market, sped up; no RPC, no wallet needed."""
    from ai_trader.broker import PaperBroker  # noqa: PLC0415
    from ai_trader.engine import Engine  # noqa: PLC0415
    from ai_trader.gui import start_gui  # noqa: PLC0415
    from ai_trader.store import Store  # noqa: PLC0415
    from ai_trader.synthetic import generate  # noqa: PLC0415

    s = _settings(args)
    s.mode = "paper"
    s.llm.enabled = False
    s.name = "demo"

    async def main() -> None:
        events = generate(args.coins, seed=args.seed, start=time.time())
        store = Store(s.data_path / "demo.duckdb")
        engine = Engine(s, PaperBroker(s), store, inline_scoring=True)
        engine.risk.kill_file = s.data_path / "DEMO_KILL"
        await engine.start()
        runner = await start_gui(s, engine, None)
        print(
            f"Demo running on an invented market at {args.speed}x speed; open the "
            "dashboard link logged above. Ctrl+C to stop."
        )
        t0, e0 = time.time(), events[0].ts
        last_tick = e0
        try:
            for evt in events:
                virtual = e0 + (time.time() - t0) * args.speed
                if evt.ts > virtual:
                    await asyncio.sleep((evt.ts - virtual) / args.speed)
                while evt.ts - last_tick >= 1:
                    last_tick += 1
                    await engine.on_tick(last_tick)
                await engine.on_event(evt, evt.ts)
            print("Demo data finished; dashboard stays up. Ctrl+C to stop.")
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()
            store.close()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    s = _settings(args)
    s.kill_file.parent.mkdir(parents=True, exist_ok=True)
    if args.off:
        s.kill_file.unlink(missing_ok=True)
        print("Kill switch off: the bot may buy again.")
    else:
        s.kill_file.write_text(f"killed from CLI at {time.ctime()}\n")
        print(
            f"Kill switch on ({s.kill_file}): the running bot sells everything and stops buying."
        )
    return 0


def cmd_address(args: argparse.Namespace) -> int:
    from ai_trader.gui import wallet_address  # noqa: PLC0415

    _settings(args)
    addr = wallet_address()
    if not addr:
        sys.exit(
            "SOLANA_PRIVATE_KEY is not set (or not a valid key). Create a wallet with `pump_bot_wallet`."
        )
    print(addr)
    return 0


def cmd_withdraw(args: argparse.Namespace) -> int:
    from solana.rpc.async_api import AsyncClient  # noqa: PLC0415
    from solders.message import MessageV0  # noqa: PLC0415
    from solders.pubkey import Pubkey  # noqa: PLC0415
    from solders.system_program import TransferParams, transfer  # noqa: PLC0415
    from solders.transaction import VersionedTransaction  # noqa: PLC0415

    from core.wallet import Wallet  # noqa: PLC0415

    s = _settings(args)
    key = os.environ.get("SOLANA_PRIVATE_KEY")
    if not key:
        sys.exit("SOLANA_PRIVATE_KEY is not set.")
    wallet = Wallet(key)
    to = Pubkey.from_string(args.to)
    fee = 5000

    async def main() -> None:
        client = AsyncClient(s.feed.rpc_endpoint)
        try:
            balance = (await client.get_balance(wallet.pubkey)).value
            lamports = (
                balance - fee if args.amount == "all" else int(float(args.amount) * 1e9)
            )
            if lamports <= 0 or lamports + fee > balance:
                sys.exit(f"Balance is {balance / 1e9:.6f} SOL; can't send that.")
            print(f"Send {lamports / 1e9:.6f} SOL from {wallet.pubkey} to {to}?")
            if not args.yes and input("Type 'yes' to send: ").strip().lower() != "yes":
                sys.exit("Cancelled.")
            blockhash = (await client.get_latest_blockhash()).value.blockhash
            ix = transfer(
                TransferParams(
                    from_pubkey=wallet.pubkey, to_pubkey=to, lamports=lamports
                )
            )
            msg = MessageV0.try_compile(wallet.pubkey, [ix], [], blockhash)
            tx = VersionedTransaction(msg, [wallet.keypair])
            sig = (await client.send_transaction(tx)).value
            print(f"Sent: https://solscan.io/tx/{sig}")
        finally:
            await client.close()

    asyncio.run(main())
    return 0


def _print_report(r: dict) -> None:
    print(f"\nVerdict: {r['verdict']}")
    for k in (
        "round_trips",
        "net_pnl_sol",
        "return_pct",
        "win_rate",
        "profit_factor",
        "expectancy_sol",
        "max_drawdown_pct",
        "fees_sol",
    ):
        if k in r:
            v = r[k]
            print(f"  {k:<20} {v:.4f}" if isinstance(v, float) else f"  {k:<20} {v}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ai_trader", description="AI trader for pump.fun (paper by default)"
    )
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"YAML settings (default {DEFAULT_CONFIG})",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser(
        "run", help="trade the live feed (paper unless the config says live)"
    )
    r.add_argument("--no-gui", action="store_true")
    r.add_argument(
        "--live",
        action="store_true",
        help="same as mode: live; still needs AI_TRADER_LIVE",
    )
    r.set_defaults(fn=cmd_run)
    b = sub.add_parser(
        "backtest", help="replay recorded feed (or synthetic data) with paper fills"
    )
    b.add_argument("--data", nargs="*", help="recording files (default data/feed/*)")
    b.add_argument(
        "--synthetic",
        type=int,
        metavar="COINS",
        help="use an invented market of COINS coins",
    )
    b.add_argument("--seed", type=int, default=7)
    b.add_argument(
        "--llm", action="store_true", help="also call Claude (costs API credits)"
    )
    b.set_defaults(fn=cmd_backtest)
    rp = sub.add_parser(
        "report", help="performance report of the latest (or --run) paper/live run"
    )
    rp.add_argument("--run")
    rp.add_argument(
        "--start-sol", type=float, default=0.0, help="starting balance, for live runs"
    )
    rp.set_defaults(fn=cmd_report)
    d = sub.add_parser(
        "demo", help="see the dashboard on an invented market, no setup needed"
    )
    d.add_argument("--coins", type=int, default=600)
    d.add_argument("--speed", type=float, default=20)
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(fn=cmd_demo)
    k = sub.add_parser("kill", help="stop buying and sell everything (--off to re-arm)")
    k.add_argument("--off", action="store_true")
    k.set_defaults(fn=cmd_kill)
    a = sub.add_parser("address", help="print the trading wallet's public address")
    a.set_defaults(fn=cmd_address)
    w = sub.add_parser(
        "withdraw", help="send SOL from the trading wallet to your own wallet"
    )
    w.add_argument("--to", required=True)
    w.add_argument("--amount", required=True, help="SOL, or 'all'")
    w.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    w.set_defaults(fn=cmd_withdraw)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
