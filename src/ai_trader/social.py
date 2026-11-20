"""Social and trend signals for new coins.

Three free sources, plus one optional paid one:

* **Coin metadata.** Every pump.fun launch points at a JSON file (its `uri`)
  that may list a Twitter/X link, a Telegram group and a website. Real teams
  usually fill these in; throwaway launches usually don't. A link to an X
  *profile* is worth more than a link to someone else's tweet or a community.
* **Narratives on pump.fun itself.** Words from the names of coins that are
  pumping right now (reached `hot_progress` of the curve in the last hour).
  A new coin riding the same meme gets a narrative score.
* **DexScreener trending.** Solana tokens with paid boosts and fresh profiles,
  and the words in their names. A boost on the coin itself means someone is
  paying to market it.
* **X mentions** (optional, needs `X_BEARER_TOKEN`): recent tweets naming the
  coin's ticker or mint, via the X API's recent search. Rate limited.

None of these are proof of anything. They are weak evidence that is easy to
fake, which is why they only move the score by `social.weight` and never
override a hard reject.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import aiohttp

from utils.logger import get_logger

if TYPE_CHECKING:
    from ai_trader.config import SocialSettings
    from ai_trader.market import CoinState, Features

logger = get_logger(__name__)

_WORD = re.compile(r"[a-z0-9]{3,}")
_STOP = {
    "the", "and", "coin", "token", "sol", "solana", "pump", "fun", "official",
    "inu", "meme", "for", "you", "with", "this", "that", "are", "new", "first",
    "real", "community", "ceo", "cto", "dev", "100x", "1000x", "moon", "launch",
}
_DEX = "https://api.dexscreener.com"


def words(*texts: str) -> set[str]:
    out: set[str] = set()
    for t in texts:
        out.update(w for w in _WORD.findall((t or "").lower()) if w not in _STOP)
    return out


@dataclass
class Meta:
    twitter: str = ""
    telegram: str = ""
    website: str = ""
    description: str = ""
    fetched: bool = False

    @property
    def twitter_kind(self) -> str:
        t = self.twitter.lower()
        if not t:
            return "none"
        if "/status/" in t:
            return "tweet"
        if "/communities/" in t or "/i/" in t:
            return "community"
        if re.search(r"(x|twitter)\.com/[a-z0-9_]{2,15}/?$", t):
            return "profile"
        return "other"


class SocialTracker:
    """Collects the signals in the background; `apply` copies them onto Features."""

    def __init__(self, settings: SocialSettings) -> None:
        self.s = settings
        self.meta: dict[str, Meta] = {}
        self._sem = asyncio.Semaphore(settings.metadata_concurrency)
        self._session: aiohttp.ClientSession | None = None
        #: (ts, words) of coins that got hot on pump.fun.
        self._hot: deque[tuple[float, frozenset[str]]] = deque()
        self._hot_mints: set[str] = set()
        self.dex_words: dict[str, float] = {}
        self.dex_mints: set[str] = set()
        self.dex_ok = False
        self._x_calls: deque[float] = deque()
        self.x_mentions: dict[str, tuple[float, int]] = {}
        self._tasks: set[asyncio.Task] = set()
        self.stats = {"meta_ok": 0, "meta_fail": 0, "dex_polls": 0, "x_calls": 0}

    # ----------------------------------------------------------------- setup
    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.s.http_timeout_s),
            headers={"User-Agent": "pump-ai-trader/1.0"},
        )
        if self.s.dexscreener:
            self._spawn(self._dex_loop())

    async def close(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        if self._session is not None:
            await self._session.close()

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ------------------------------------------------------------- metadata
    def on_create(self, coin: CoinState) -> None:
        if not self.s.fetch_metadata or not coin.uri or self._session is None:
            return
        if len(self.meta) > 8000:
            for k in list(self.meta)[:2000]:
                self.meta.pop(k, None)
        self.meta[coin.mint] = Meta()
        self._spawn(self._fetch_meta(coin.mint, coin.uri))

    def _url(self, uri: str) -> str | None:
        if uri.startswith("ipfs://"):
            uri = "https://ipfs.io/ipfs/" + uri[7:]
        if "/ipfs/" in uri and self.s.ipfs_gateway:
            uri = self.s.ipfs_gateway.rstrip("/") + "/ipfs/" + uri.split("/ipfs/", 1)[1]
        if not uri.startswith("https://"):
            return None  # never fetch plain http or local addresses
        return uri

    async def _fetch_meta(self, mint: str, uri: str) -> None:
        url = self._url(uri)
        if url is None or self._session is None:
            return
        async with self._sem:
            try:
                async with self._session.get(url) as r:
                    if r.status != 200:
                        self.stats["meta_fail"] += 1
                        return
                    raw = await r.content.read(64_000)
                import json  # noqa: PLC0415

                data = json.loads(raw)
            except (aiohttp.ClientError, TimeoutError, ValueError):
                self.stats["meta_fail"] += 1
                return
        if not isinstance(data, dict):
            return
        ext = data.get("extensions") if isinstance(data.get("extensions"), dict) else {}

        def pick(*keys: str) -> str:
            for k in keys:
                v = data.get(k) or ext.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()[:200]
            return ""

        m = self.meta.get(mint)
        if m is None:
            return
        m.twitter = pick("twitter", "x")
        m.telegram = pick("telegram")
        m.website = pick("website")
        m.description = pick("description")[:300]
        m.fetched = True
        self.stats["meta_ok"] += 1

    # ------------------------------------------------------------ narratives
    def on_progress(self, coin: CoinState, now: float) -> None:
        if coin.mint in self._hot_mints or coin.progress < self.s.hot_progress:
            return
        self._hot_mints.add(coin.mint)
        self._hot.append((now, frozenset(words(coin.name, coin.symbol))))

    def _narrative(self, coin: CoinState, now: float) -> tuple[float, list[str]]:
        while self._hot and now - self._hot[0][0] > self.s.narrative_window_s:
            self._hot.popleft()
        mine = words(coin.name, coin.symbol)
        if not mine:
            return 0.0, []
        best, hits = 0.0, []
        for w in mine:
            n = sum(1 for _, ws in self._hot if w in ws)
            if coin.mint in self._hot_mints and n:
                n -= 1  # don't count the coin as its own trend
            internal = 1 - math.exp(-n / 2)
            external = self.dex_words.get(w, 0.0)
            v = max(internal, 0.8 * external)
            if v > 0.15:
                hits.append(w)
            best = max(best, v)
        return best, sorted(hits)[:5]

    # ----------------------------------------------------------- dexscreener
    async def _dex_loop(self) -> None:
        while True:
            try:
                await self._dex_poll()
            except Exception as e:  # noqa: BLE001 - keep polling whatever happens
                logger.warning(f"DexScreener poll failed: {type(e).__name__}: {e}")
            await asyncio.sleep(self.s.dexscreener_poll_s)

    async def _get(self, url: str) -> Any:
        assert self._session is not None
        async with self._session.get(url) as r:
            if r.status != 200:
                return None
            return await r.json(content_type=None)

    async def _dex_poll(self) -> None:
        addrs: list[str] = []
        for path in ("/token-boosts/top/v1", "/token-boosts/latest/v1", "/token-profiles/latest/v1"):
            data = await self._get(_DEX + path)
            if not isinstance(data, list):
                continue
            for item in data:
                if isinstance(item, dict) and item.get("chainId") == "solana":
                    a = str(item.get("tokenAddress", ""))
                    if a and a not in addrs:
                        addrs.append(a)
        self.dex_mints = set(addrs)
        counts: dict[str, float] = {}
        for i in range(0, min(len(addrs), 90), 30):
            pairs = await self._get(f"{_DEX}/tokens/v1/solana/{','.join(addrs[i : i + 30])}")
            if not isinstance(pairs, list):
                continue
            seen: set[str] = set()
            for p in pairs:
                base = p.get("baseToken") or {}
                a = base.get("address", "")
                if a in seen:
                    continue
                seen.add(a)
                ch = (p.get("priceChange") or {}).get("h1") or 0
                if ch <= 0:
                    continue  # only narratives that are going up
                for w in words(base.get("name", ""), base.get("symbol", "")):
                    counts[w] = counts.get(w, 0) + 1
        self.dex_words = {w: 1 - math.exp(-n / 1.5) for w, n in counts.items()}
        self.dex_ok = True
        self.stats["dex_polls"] += 1

    # ------------------------------------------------------------------ X
    async def x_count(self, coin: CoinState) -> int:
        """Recent tweets naming the ticker or mint; -1 when unknown."""
        token = self.s.x_bearer_token
        if not token or self._session is None:
            return -1
        cached = self.x_mentions.get(coin.mint)
        now = time.monotonic()
        if cached and now - cached[0] < 120:
            return cached[1]
        while self._x_calls and now - self._x_calls[0] > 900:
            self._x_calls.popleft()
        if len(self._x_calls) >= self.s.x_max_calls_per_15min:
            return cached[1] if cached else -1
        self._x_calls.append(now)
        self.stats["x_calls"] += 1
        sym = re.sub(r"[^A-Za-z0-9_]", "", coin.symbol)[:15]
        q = f'"{coin.mint}"' + (f" OR ${sym}" if sym else "")
        try:
            async with self._session.get(
                "https://api.x.com/2/tweets/counts/recent",
                params={"query": q, "granularity": "hour"},
                headers={"Authorization": f"Bearer {token}"},
            ) as r:
                if r.status != 200:
                    return -1
                data = await r.json()
            n = int((data.get("meta") or {}).get("total_tweet_count", 0))
        except (aiohttp.ClientError, TimeoutError, ValueError):
            return -1
        self.x_mentions[coin.mint] = (now, n)
        return n

    # ----------------------------------------------------------------- apply
    async def apply(self, coin: CoinState, f: Features, now: float) -> Features:
        """A copy of `f` with the social fields filled in."""
        m = self.meta.get(coin.mint) or Meta()
        kind = m.twitter_kind
        narrative, hits = self._narrative(coin, now)
        boosted = coin.mint in self.dex_mints
        x = await self.x_count(coin)
        presence = (
            {"profile": 0.35, "tweet": 0.15, "community": 0.2, "other": 0.1}.get(kind, 0.0)
            + (0.25 if m.telegram else 0.0)
            + (0.25 if m.website else 0.0)
            + (0.15 if len(m.description) >= 40 else 0.0)
        )
        parts = [(0.35, min(presence, 1.0)), (0.45, narrative)]
        parts.append((0.2, 1.0 if boosted else 0.0))
        if x >= 0:
            parts.append((0.35, math.tanh(x / 25)))
        return replace(
            f,
            has_twitter=bool(m.twitter),
            has_telegram=bool(m.telegram),
            has_website=bool(m.website),
            twitter_kind=kind,
            narrative_score=narrative,
            narrative_words=" ".join(hits),
            dex_boosted=boosted,
            x_mentions=x,
            social_score=sum(w * v for w, v in parts) / sum(w for w, _ in parts),
            social_checked=True,
        )
