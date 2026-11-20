"""Per-coin state built from the event stream, and the features the scorers read."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field

from ai_trader.events import (
    INITIAL_REAL_TOKEN_RESERVES,
    TOTAL_SUPPLY,
    CompleteEvt,
    CreateEvt,
    Event,
    TradeEvt,
    price_from_reserves,
)

#: Reserves of a fresh pump.fun curve.
INITIAL_V_SOL = 30 * 10**9
INITIAL_V_TOK = 1_073_000_000 * 10**6


@dataclass
class CoinState:
    mint: str
    created_at: float
    name: str = ""
    symbol: str = ""
    creator: str = ""
    is_mayhem: bool = False
    sol_quoted: bool = True
    v_sol: int = INITIAL_V_SOL
    v_tok: int = INITIAL_V_TOK
    real_tok: int = INITIAL_REAL_TOKEN_RESERVES
    completed: bool = False
    last_trade_at: float = 0.0
    last_buy_at: float = 0.0
    peak_price: float = 0.0
    first_price: float = 0.0
    n_buys: int = 0
    n_sells: int = 0
    buy_sol: float = 0.0
    sell_sol: float = 0.0
    dev_buy_tokens: int = 0
    creator_sold: bool = False
    #: Net raw tokens each wallet holds, from its own buys and sells.
    holders: dict[str, int] = field(default_factory=dict)
    trades_by_wallet: dict[str, int] = field(default_factory=dict)
    buyers: set[str] = field(default_factory=set)
    sellers: set[str] = field(default_factory=set)
    #: (ts, price, is_buy, sol) for momentum windows; bounded.
    recent: deque = field(default_factory=lambda: deque(maxlen=600))
    last_scored_at: float = 0.0

    @property
    def price(self) -> float:
        return price_from_reserves(self.v_sol, self.v_tok)

    @property
    def progress(self) -> float:
        return max(0.0, 1 - self.real_tok / INITIAL_REAL_TOKEN_RESERVES)

    @property
    def mcap_sol(self) -> float:
        return self.price * TOTAL_SUPPLY / 10**6

    def age(self, now: float) -> float:
        return now - self.created_at


@dataclass(frozen=True)
class Features:
    """What the scorers see about one coin at one moment."""

    mint: str
    symbol: str
    name: str
    age_s: float
    progress: float
    mcap_sol: float
    price: float
    n_trades: int
    n_buys: int
    n_sells: int
    unique_buyers: int
    unique_sellers: int
    buy_sol: float
    sell_sol: float
    net_flow_sol: float
    buy_sell_ratio: float
    #: Price change since the first trade, and over the last 30 s.
    ret_total: float
    ret_30s: float
    drawdown_from_peak: float
    #: Trades per minute in the last 30 s vs. over the coin's life.
    rate_30s: float
    rate_life: float
    #: Net buy SOL in the last 30 s.
    flow_30s: float
    dev_buy_share: float
    creator_sold: bool
    creator_share: float
    top1_share: float
    top10_share: float
    #: Share of trades made by wallets with 5+ trades in this coin (bots/wash).
    churn_share: float
    #: Coins with the same symbol seen in the last hour (copycat spam).
    symbol_clones: int
    is_mayhem: bool
    creator_launches: int
    creator_best_progress: float
    creator_rug_rate: float
    seconds_since_last_buy: float

    def as_prompt_dict(self) -> dict:
        d = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in self.__dict__.items()
        }
        d.pop("mint", None)
        d.pop("price", None)
        return d


class CreatorBook:
    """What each creator's earlier coins did, learned from the stream itself."""

    def __init__(self) -> None:
        self.launches: dict[str, list[str]] = {}
        self.best_progress: dict[str, float] = {}
        self.rugs: dict[str, int] = {}

    def on_create(self, creator: str, mint: str) -> None:
        self.launches.setdefault(creator, []).append(mint)

    def on_close(self, coin: CoinState) -> None:
        """Called when a coin leaves the tracker: remember how far it got."""
        c = coin.creator
        self.best_progress[c] = max(self.best_progress.get(c, 0.0), coin.progress)
        if coin.creator_sold and coin.progress < 0.2:
            self.rugs[c] = self.rugs.get(c, 0) + 1

    def stats(self, creator: str) -> tuple[int, float, float]:
        n = len(self.launches.get(creator, []))
        prior = max(n - 1, 0)
        rug_rate = self.rugs.get(creator, 0) / prior if prior else 0.0
        return prior, self.best_progress.get(creator, 0.0), min(rug_rate, 1.0)

    def to_rows(self) -> list[tuple[str, int, float, int]]:
        return [
            (c, len(m), self.best_progress.get(c, 0.0), self.rugs.get(c, 0))
            for c, m in self.launches.items()
        ]

    def load_rows(self, rows: list[tuple[str, int, float, int]]) -> None:
        for creator, n, best, rugs in rows:
            self.launches[creator] = ["?"] * int(n)
            self.best_progress[creator] = float(best)
            self.rugs[creator] = int(rugs)


class MarketTracker:
    """Holds the state of every recent coin and turns it into features."""

    def __init__(self, max_coins: int = 5000, forget_after_s: float = 3600) -> None:
        self.coins: OrderedDict[str, CoinState] = OrderedDict()
        self.creators = CreatorBook()
        self.max_coins = max_coins
        self.forget_after_s = forget_after_s
        self._symbols: deque[tuple[float, str]] = deque()
        #: Mints the trader holds; never forgotten while held.
        self.pinned: set[str] = set()

    def apply(self, evt: Event) -> CoinState | None:
        if isinstance(evt, CreateEvt):
            coin = CoinState(
                mint=evt.mint,
                created_at=evt.ts,
                name=evt.name,
                symbol=evt.symbol,
                creator=evt.creator,
                is_mayhem=evt.is_mayhem,
                sol_quoted=evt.sol_quoted,
            )
            self.coins[evt.mint] = coin
            self.creators.on_create(evt.creator, evt.mint)
            self._symbols.append((evt.ts, evt.symbol.upper()))
            self._evict(evt.ts)
            return coin
        coin = self.coins.get(evt.mint)
        if coin is None:
            return None
        if isinstance(evt, CompleteEvt):
            coin.completed = True
            return coin
        assert isinstance(evt, TradeEvt)
        self._trade(coin, evt)
        return coin

    def _trade(self, coin: CoinState, t: TradeEvt) -> None:
        coin.v_sol, coin.v_tok, coin.real_tok = t.v_sol, t.v_tok, t.real_tok
        price = coin.price
        if not coin.first_price:
            coin.first_price = price
        coin.peak_price = max(coin.peak_price, price)
        coin.last_trade_at = t.ts
        coin.trades_by_wallet[t.user] = coin.trades_by_wallet.get(t.user, 0) + 1
        if t.is_buy:
            coin.n_buys += 1
            coin.buy_sol += t.sol
            coin.buyers.add(t.user)
            coin.last_buy_at = t.ts
            coin.holders[t.user] = coin.holders.get(t.user, 0) + t.tokens
            if t.user == coin.creator and coin.n_buys == 1:
                coin.dev_buy_tokens = t.tokens
        else:
            coin.n_sells += 1
            coin.sell_sol += t.sol
            coin.sellers.add(t.user)
            coin.holders[t.user] = max(0, coin.holders.get(t.user, 0) - t.tokens)
            if t.user == coin.creator:
                coin.creator_sold = True
        coin.recent.append((t.ts, price, t.is_buy, t.sol))

    def _evict(self, now: float) -> None:
        while self._symbols and now - self._symbols[0][0] > 3600:
            self._symbols.popleft()
        while self.coins:
            mint, coin = next(iter(self.coins.items()))
            too_many = len(self.coins) > self.max_coins
            too_old = (
                now - max(coin.created_at, coin.last_trade_at) > self.forget_after_s
            )
            if not (too_many or too_old):
                break
            self.coins.popitem(last=False)
            if mint in self.pinned:
                self.coins[mint] = coin  # still held: keep it, at the young end
                if len(self.pinned) >= len(self.coins):
                    break
                continue
            self.creators.on_close(coin)

    def features(self, coin: CoinState, now: float) -> Features:
        n_trades = coin.n_buys + coin.n_sells
        age = max(coin.age(now), 1e-6)
        window = [r for r in coin.recent if now - r[0] <= 30]
        p30 = window[0][1] if window else coin.price
        flow_30 = sum(s if b else -s for _, _, b, s in window)
        held = sorted((v for v in coin.holders.values() if v > 0), reverse=True)
        top1 = held[0] / TOTAL_SUPPLY if held else 0.0
        top10 = sum(held[:10]) / TOTAL_SUPPLY
        churners = sum(n for n in coin.trades_by_wallet.values() if n >= 5)
        launches, best, rug_rate = self.creators.stats(coin.creator)
        sym = coin.symbol.upper()
        clones = sum(1 for _, s in self._symbols if s == sym) - 1
        return Features(
            mint=coin.mint,
            symbol=coin.symbol,
            name=coin.name,
            age_s=age,
            progress=coin.progress,
            mcap_sol=coin.mcap_sol,
            price=coin.price,
            n_trades=n_trades,
            n_buys=coin.n_buys,
            n_sells=coin.n_sells,
            unique_buyers=len(coin.buyers),
            unique_sellers=len(coin.sellers),
            buy_sol=coin.buy_sol,
            sell_sol=coin.sell_sol,
            net_flow_sol=coin.buy_sol - coin.sell_sol,
            buy_sell_ratio=coin.buy_sol / max(coin.sell_sol, 0.01),
            ret_total=coin.price / coin.first_price - 1 if coin.first_price else 0.0,
            ret_30s=coin.price / p30 - 1 if p30 else 0.0,
            drawdown_from_peak=1 - coin.price / coin.peak_price
            if coin.peak_price
            else 0.0,
            rate_30s=len(window) * 2.0,
            rate_life=n_trades / (age / 60),
            flow_30s=flow_30,
            dev_buy_share=coin.dev_buy_tokens / TOTAL_SUPPLY,
            creator_sold=coin.creator_sold,
            creator_share=coin.holders.get(coin.creator, 0) / TOTAL_SUPPLY,
            top1_share=top1,
            top10_share=top10,
            churn_share=churners / n_trades if n_trades else 0.0,
            symbol_clones=max(clones, 0),
            is_mayhem=coin.is_mayhem,
            creator_launches=launches,
            creator_best_progress=best,
            creator_rug_rate=rug_rate,
            seconds_since_last_buy=now - coin.last_buy_at if coin.last_buy_at else age,
        )
