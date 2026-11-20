"""Market data in: a live pump.fun log stream, or a recording of one.

`LiveFeed` subscribes to every transaction that mentions the pump.fun program
(one `logsSubscribe`, which every RPC provider supports), decodes creates,
trades and graduations, and optionally records them to gzipped JSONL so the
same session can be replayed later by the backtester. It reconnects with
backoff (tenacity) and can mirror events to NATS for other consumers.
"""

from __future__ import annotations

import asyncio
import gzip
import zlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import websockets
import websockets.exceptions
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_never,
    wait_exponential_jitter,
)

from ai_trader.events import Event, decode_logs, from_dict, to_dict
from utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from interfaces.core import TokenInfo

logger = get_logger(__name__)

PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"


class Recorder:
    """Writes events to data/feed/<UTC date>_<start time>.jsonl.gz.

    One file per day per run: a run that is killed leaves a truncated gzip,
    and appending to it would make the rest of that file unreadable.
    """

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self._day = ""
        self._started = time.strftime("%H%M%S", time.gmtime())
        self._fh: gzip.GzipFile | None = None
        self._n = 0

    def write(self, evt: Event) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime(evt.ts))
        if day != self._day:
            self.close()
            self._day = day
            self._fh = gzip.open(self.dir / f"{day}_{self._started}.jsonl.gz", "at")
        assert self._fh is not None
        self._fh.write(json.dumps(to_dict(evt), separators=(",", ":")) + "\n")
        self._n += 1
        if self._n % 500 == 0:
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def replay(paths: list[Path]) -> Iterator[Event]:
    """Events from recordings, in file order (each file is already in time order)."""
    for path in sorted(paths):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as fh:
            try:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            yield from_dict(json.loads(line))
                        except ValueError:
                            break  # a half-written last line
            except (EOFError, OSError, zlib.error):
                # A run that was killed (or is still writing) leaves the gzip
                # unfinished; use what was readable.
                logger.warning(f"{path.name} ends early (unfinished recording)")


class _TokenInfoParser:
    """Builds the repo's TokenInfo from a create's logs, so a live buy needs no RPC."""

    def __init__(self) -> None:
        from interfaces.core import Platform  # noqa: PLC0415
        from monitoring.universal_logs_listener import (
            UniversalLogsListener,
        )

        listener = UniversalLogsListener("wss://unused", platforms=[Platform.PUMP_FUN])
        self.parser = listener.platform_parsers[Platform.PUMP_FUN]

    def parse(self, logs: list[str], signature: str) -> TokenInfo | None:
        try:
            return self.parser.parse_token_creation_from_logs(logs, signature)
        except Exception:  # noqa: BLE001 - one bad tx must not stop the feed
            return None


class LiveFeed:
    def __init__(
        self,
        wss_endpoint: str,
        *,
        recorder: Recorder | None = None,
        token_infos: dict[str, TokenInfo] | None = None,
        nats_url: str | None = None,
    ) -> None:
        if not wss_endpoint.startswith(("ws://", "wss://")):
            raise ValueError(
                "feed.wss_endpoint must be a ws:// or wss:// URL (set SOLANA_NODE_WSS_ENDPOINT)"
            )  # noqa: TRY003
        self.wss = wss_endpoint
        self.recorder = recorder
        self.token_infos = token_infos
        self._ti_parser = _TokenInfoParser() if token_infos is not None else None
        self.nats_url = nats_url
        self._nats = None
        self.connected = False
        self.last_event_at = 0.0

    async def _connect_nats(self) -> None:
        if not self.nats_url:
            return
        try:
            import nats  # noqa: PLC0415

            self._nats = await nats.connect(self.nats_url)
            logger.info("Publishing events to NATS")
        except Exception as e:  # noqa: BLE001 - NATS is optional
            logger.warning(
                f"NATS unavailable ({type(e).__name__}); continuing without it"
            )

    async def events(self) -> AsyncIterator[Event]:
        await self._connect_nats()
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(
                (OSError, websockets.exceptions.WebSocketException, TimeoutError)
            ),
            wait=wait_exponential_jitter(initial=1, max=30),
            stop=stop_never,
            reraise=True,
        ):
            with attempt:
                async for evt in self._stream():
                    yield evt

    async def _stream(self) -> AsyncIterator[Event]:
        async with websockets.connect(
            self.wss, max_size=2**24, ping_interval=20, ping_timeout=20
        ) as ws:
            await ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "logsSubscribe",
                        "params": [
                            {"mentions": [PUMP_PROGRAM]},
                            {"commitment": "processed"},
                        ],
                    }
                )
            )
            self.connected = True
            logger.info("Subscribed to pump.fun logs")
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)
                    msg = json.loads(raw)
                    if msg.get("method") != "logsNotification":
                        continue
                    value = msg["params"]["result"]["value"]
                    if value.get("err") is not None:
                        continue
                    logs = value.get("logs") or []
                    sig = value.get("signature", "")
                    now = time.time()
                    for evt in decode_logs(logs, sig, now):
                        if evt.kind == "create" and self._ti_parser is not None:
                            info = self._ti_parser.parse(logs, sig)
                            if info is not None:
                                self.token_infos[evt.mint] = info
                        if self.recorder:
                            self.recorder.write(evt)
                        if self._nats is not None:
                            await self._nats.publish(
                                f"pump.{evt.kind}", json.dumps(to_dict(evt)).encode()
                            )
                        self.last_event_at = now
                        yield evt
            finally:
                self.connected = False
