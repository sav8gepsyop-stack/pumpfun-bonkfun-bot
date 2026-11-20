"""Replay recorded (or synthetic) events through the real engine with paper fills."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from ai_trader import report
from ai_trader.broker import PaperBroker
from ai_trader.engine import Engine
from ai_trader.store import Store

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ai_trader.config import AITraderSettings
    from ai_trader.events import Event


async def run_backtest(
    settings: AITraderSettings,
    events: Iterable[Event],
    *,
    label: str,
    out_root: Path | None = None,
) -> tuple[dict, dict[str, Path]]:
    """Run the engine over `events`; returns the report and where it was written.

    Each backtest gets its own store, so it never mixes with paper or live data.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = (out_root or settings.data_path / "backtests") / f"{stamp}-{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    store = Store(out_dir / "backtest.duckdb")
    broker = PaperBroker(settings)
    engine = Engine(settings, broker, store, inline_scoring=True, run_id=f"bt-{stamp}")
    # A backtest must not be stopped by a kill file left over from a live run.
    engine.risk.kill_file = out_dir / "KILL"
    await engine.start()
    last_tick = None
    now = 0.0
    for evt in events:
        now = evt.ts
        if last_tick is None:
            last_tick = now
        while now - last_tick >= 1.0:
            last_tick += 1.0
            await engine.on_tick(last_tick)
        await engine.on_event(evt, now)
    # Close what is still open at the last price, so the report counts it.
    await engine.liquidate_all(now, "end of backtest")
    engine.snapshot_now(now + 1)
    result = report.compute(store, engine.run_id, settings.risk.paper_bankroll_sol)
    result["engine_stats"] = engine.stats
    paths = report.write(result, out_dir, f"Backtest {label}")
    store.close()
    return result, paths
