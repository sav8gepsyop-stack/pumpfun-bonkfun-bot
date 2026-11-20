"""Run the trader on the live feed, paper or live, with the dashboard beside it."""

from __future__ import annotations

import asyncio
import contextlib
import signal
import time
from typing import TYPE_CHECKING

from ai_trader.broker import LiveBroker, PaperBroker
from ai_trader.config import LIVE_CONFIRM_ENV, LIVE_CONFIRM_VALUE
from ai_trader.engine import Engine
from ai_trader.feed import LiveFeed, Recorder
from ai_trader.store import Store
from utils.logger import get_logger

if TYPE_CHECKING:
    from ai_trader.config import AITraderSettings

logger = get_logger(__name__)


class LiveModeNotConfirmedError(RuntimeError):
    pass


def make_broker(settings: AITraderSettings) -> PaperBroker | LiveBroker:
    if settings.mode == "paper":
        return PaperBroker(settings)
    if not settings.live_confirmed():
        raise LiveModeNotConfirmedError(
            f"mode is live, but {LIVE_CONFIRM_ENV} is not set to {LIVE_CONFIRM_VALUE}. "
            "Live mode spends real SOL; set it only after reviewing a paper run."
        )
    return LiveBroker(settings)


async def run(settings: AITraderSettings, *, gui: bool = True) -> None:
    broker = make_broker(settings)
    settings.data_path.mkdir(parents=True, exist_ok=True)
    store = Store(settings.data_path / f"{settings.name}.duckdb")
    engine = Engine(settings, broker, store)
    social = None
    if settings.social.enabled:
        from ai_trader.social import SocialTracker  # noqa: PLC0415

        social = SocialTracker(settings.social)
        await social.start()
        engine.social = social
    await engine.start()
    token_infos = getattr(broker, "token_infos", None)
    feed = LiveFeed(
        settings.feed.wss_endpoint,
        recorder=Recorder(settings.data_path / "feed")
        if settings.feed.record
        else None,
        token_infos=token_infos,
        nats_url=settings.feed.nats_url,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    logger.info(
        f"AI trader '{settings.name}' starting in {settings.mode.upper()} mode, run {engine.run_id}"
    )
    if settings.mode == "live":
        logger.warning(f"LIVE: trading real SOL from wallet {broker.address}")

    async def consume() -> None:
        async for evt in feed.events():
            await engine.on_event(evt, time.time())

    async def tick() -> None:
        while True:
            await asyncio.sleep(1)
            await engine.on_tick(time.time())
            if token_infos is not None and len(token_infos) > 6000:
                keep = set(engine.tracker.coins)
                for mint in [m for m in token_infos if m not in keep]:
                    token_infos.pop(mint, None)

    tasks = [asyncio.create_task(consume()), asyncio.create_task(tick())]
    runner = None
    if gui:
        from ai_trader.gui import start_gui  # noqa: PLC0415

        runner = await start_gui(settings, engine, feed)
    stopper = asyncio.create_task(stop.wait())
    done, _ = await asyncio.wait([*tasks, stopper], return_when=asyncio.FIRST_COMPLETED)
    for t in done:
        if t is not stopper and t.exception():
            logger.error(f"Trader task failed: {t.exception()!r}")
    logger.info("Stopping...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await engine.drain()
    engine.snapshot_now(time.time())
    if engine.positions:
        logger.warning(
            f"{len(engine.positions)} position(s) still open. "
            + (
                "They resume on the next start; use the kill switch to sell everything."
                if settings.mode == "live"
                else "Paper positions are dropped."
            )
        )
    store.save_creators(engine.tracker.creators.to_rows())
    if feed.recorder:
        feed.recorder.close()
    if runner is not None:
        await runner.cleanup()
    if social is not None:
        await social.close()
    await broker.close()
    store.close()
