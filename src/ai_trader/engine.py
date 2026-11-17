"""The trading loop: events in, scores, risk checks, orders and exits out.

The engine never reads the clock itself; every call passes `now`. Live, that is
the wall clock; in a backtest it is the recorded event time, so a day of data
replays in seconds with exactly the decisions the live loop would have made.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ai_trader.broker import curve_sell
from ai_trader.events import TOKEN_DECIMALS, CreateEvt, Event, TradeEvt
from ai_trader.market import CoinState, MarketTracker
from ai_trader.risk import RiskManager
from ai_trader.scoring import EnsembleScorer, Score
from utils.logger import get_logger

if TYPE_CHECKING:
    from ai_trader.broker import Broker, Fill
    from ai_trader.config import AITraderSettings
    from ai_trader.store import Store

logger = get_logger(__name__)


@dataclass
class Position:
    mint: str
    symbol: str
    opened_at: float
    tokens_raw: int
    cost_sol: float
    entry_price: float
    peak_price: float
    score: float
    tp_done: bool = False
    realized_sol: float = 0.0
    #: An order for this coin is waiting to fill; no new exit decisions until then.
    busy: bool = False
    reasons: list[str] = field(default_factory=list)


@dataclass
class Order:
    due: float
    mint: str
    side: str  # "buy" | "sell"
    sol: float = 0.0
    tokens_raw: int = 0
    reason: str = ""
    score: Score | None = None


class Engine:
    def __init__(
        self,
        settings: AITraderSettings,
        broker: Broker,
        store: Store,
        *,
        inline_scoring: bool = False,
        run_id: str | None = None,
    ) -> None:
        self.s = settings
        self.broker = broker
        self.store = store
        self.mode = "live" if broker.live else "paper"
        self.run_id = run_id or f"{self.mode}-{uuid.uuid4().hex[:8]}"
        self.tracker = MarketTracker()
        self.tracker.creators.load_rows(store.load_creators())
        self.scorer = EnsembleScorer(settings.strategy, settings.llm)
        self.risk = RiskManager(settings.risk, settings.kill_file)
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.inline_scoring = inline_scoring
        self._scoring: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        self._last_equity = 0.0
        self._last_creators = time.time()
        self._cash_cache = 0.0
        self.latency = 0.0 if broker.live else settings.costs.latency_s
        self.started_at = time.time()
        self.stats = {"events": 0, "coins": 0, "scored": 0, "rejected": 0, "bought": 0}
        self.recent_decisions: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- startup
    async def start(self) -> None:
        self._cash_cache = await self.broker.balance_sol()
        if self.mode == "live":
            await self._resume_live()

    async def _resume_live(self) -> None:
        from trading.position_store import decode_token_info  # noqa: PLC0415

        for row in self.store.open_positions("live"):
            pos = Position(
                mint=row["mint"],
                symbol=row["symbol"],
                opened_at=row["opened_at"],
                tokens_raw=row["tokens_raw"],
                cost_sol=row["cost_sol"],
                entry_price=row["entry_price"],
                peak_price=row["peak_price"],
                score=row["score"],
                tp_done=row["tp_done"],
                realized_sol=row["realized_sol"],
            )
            self.positions[pos.mint] = pos
            coin = CoinState(mint=pos.mint, created_at=pos.opened_at, symbol=pos.symbol)
            self.tracker.coins[pos.mint] = coin
            self.tracker.pinned.add(pos.mint)
            if row.get("token_info") and hasattr(self.broker, "token_infos"):
                import json  # noqa: PLC0415

                self.broker.token_infos[pos.mint] = decode_token_info(
                    json.loads(row["token_info"])
                )
            logger.warning(
                f"Resumed open position {pos.symbol} ({pos.mint}); its price updates "
                "with the next trade on its curve"
            )

    # ------------------------------------------------------------------ events
    async def on_event(self, evt: Event, now: float) -> None:
        self.stats["events"] += 1
        coin = self.tracker.apply(evt)
        if isinstance(evt, CreateEvt):
            self.stats["coins"] += 1
        if coin is None:
            return
        await self._process_orders(now)
        pos = self.positions.get(coin.mint)
        if pos is not None:
            await self._check_exit(pos, coin, now)
        elif isinstance(evt, TradeEvt):
            await self._maybe_score(coin, now)

    async def on_tick(self, now: float) -> None:
        """Time-based work: fills that came due, holds that ran out, the kill switch."""
        await self._process_orders(now)
        killed = self.risk.kill_requested()
        for pos in list(self.positions.values()):
            coin = self.tracker.coins.get(pos.mint)
            if coin is None:
                continue
            if killed and not pos.busy:
                self._sell(pos, now, pos.tokens_raw, "kill switch")
            else:
                await self._check_exit(pos, coin, now)
        if now - self._last_equity >= 15:
            self._last_equity = now
            self._snapshot(now)
            self.store.flush()
        if self.mode == "live" and now - self._last_creators >= 900:
            self._last_creators = now
            self.store.save_creators(self.tracker.creators.to_rows())

    # ---------------------------------------------------------------- scoring
    async def _maybe_score(self, coin: CoinState, now: float) -> None:
        st = self.s.strategy
        age = coin.age(now)
        if (
            age < st.min_age_s
            or age > st.max_age_s
            or coin.completed
            or not coin.sol_quoted
        ):
            return
        if now - coin.last_scored_at < st.rescore_every_s or coin.mint in self._scoring:
            return
        if any(o.mint == coin.mint for o in self.orders):
            return
        coin.last_scored_at = now
        if self.inline_scoring:
            await self._score_and_act(coin, now)
        else:
            self._scoring.add(coin.mint)
            self._spawn(self._score_and_act(coin, now))

    async def _score_and_act(self, coin: CoinState, now: float) -> None:
        try:
            features = self.tracker.features(coin, now)
            score = await self.scorer.score(features)
            self.stats["scored"] += 1
            if not self.inline_scoring:
                now = time.time()  # Claude may have taken a while
            if score.rejected:
                self.stats["rejected"] += 1
                self._decide(now, coin, score, "reject", score.rejected, features)
                return
            if score.value < self.s.strategy.buy_threshold:
                self._decide(
                    now, coin, score, "pass", f"score {score.value:.0f}", features
                )
                return
            if coin.mint in self.positions:
                return
            sol, why_not = self.risk.size(
                now=now,
                score=score.value,
                threshold=self.s.strategy.buy_threshold,
                bankroll_sol=self._bankroll(),
                open_count=len(self.positions)
                + sum(o.side == "buy" for o in self.orders),
                exposure_sol=self._exposure(),
                unrealized_sol=self._unrealized(),
            )
            if why_not:
                self._decide(now, coin, score, "blocked", why_not, features)
                return
            self._decide(now, coin, score, "buy", f"{sol} SOL", features)
            self.orders.append(
                Order(
                    now + self.latency,
                    coin.mint,
                    "buy",
                    sol=sol,
                    reason="; ".join(score.reasons),
                    score=score,
                )
            )
            self.risk.on_buy(now)
            await self._process_orders(now)
        finally:
            self._scoring.discard(coin.mint)

    def _decide(
        self,
        now: float,
        coin: CoinState,
        score: Score,
        action: str,
        reason: str,
        features: Any,
    ) -> None:
        row = {
            "ts": now,
            "run_id": self.run_id,
            "mint": coin.mint,
            "symbol": coin.symbol,
            "score": round(score.value, 2),
            "heuristic": round(score.heuristic, 2),
            "llm": score.llm,
            "action": action,
            "reason": reason,
            "features": features.as_prompt_dict() if action != "reject" else None,
        }
        self.store.decision(row)
        if action != "reject":
            self.recent_decisions = (
                [{**row, "reasons": score.reasons}] + self.recent_decisions
            )[:50]
            logger.info(
                f"{action.upper():7} {coin.symbol:>10} score {score.value:5.1f}  {reason}"
            )

    # ------------------------------------------------------------------ exits
    async def _check_exit(self, pos: Position, coin: CoinState, now: float) -> None:
        if pos.busy:
            return
        st = self.s.strategy
        price = coin.price
        pos.peak_price = max(pos.peak_price, price)
        change = price / pos.entry_price - 1 if pos.entry_price else 0.0
        reason = None
        fraction = 1.0
        if coin.completed:
            reason = "coin graduated"
        elif coin.creator_sold and st.exit_on_creator_sell:
            reason = "creator sold"
        elif change <= -st.stop_loss:
            reason = f"stop loss {change:+.0%}"
        elif not pos.tp_done and change >= st.take_profit:
            reason, fraction = (
                f"take profit {change:+.0%}",
                st.take_profit_sell_fraction,
            )
        elif pos.tp_done and price <= pos.peak_price * (1 - st.trailing_stop):
            reason = f"trailing stop {price / pos.peak_price - 1:+.0%} from peak"
        elif now - pos.opened_at >= st.max_hold_s:
            reason = "max hold time"
        elif (
            now - (coin.last_buy_at or pos.opened_at) >= st.stale_after_s and change < 0
        ):
            reason = f"no buyers for {st.stale_after_s:.0f}s"
        if reason:
            tokens = pos.tokens_raw if fraction >= 1 else int(pos.tokens_raw * fraction)
            if fraction < 1:
                pos.tp_done = True
            self._sell(pos, now, tokens, reason)
            await self._process_orders(now)

    def _sell(self, pos: Position, now: float, tokens_raw: int, reason: str) -> None:
        pos.busy = True
        self.orders.append(
            Order(
                now + self.latency,
                pos.mint,
                "sell",
                tokens_raw=tokens_raw,
                reason=reason,
            )
        )

    # ----------------------------------------------------------------- orders
    async def _process_orders(self, now: float) -> None:
        due = [o for o in self.orders if o.due <= now]
        if not due:
            return
        self.orders = [o for o in self.orders if o.due > now]
        for order in due:
            if self.broker.live:
                self._spawn(self._execute(order, now))
            else:
                await self._execute(order, now)

    async def _execute(self, order: Order, now: float) -> None:
        coin = self.tracker.coins.get(order.mint)
        if coin is None:
            return
        if order.side == "buy":
            fill = await self.broker.buy(coin, order.sol)
            self._cash_cache = await self.broker.balance_sol()
            if not fill.ok:
                logger.warning(f"Buy {coin.symbol} failed: {fill.error}")
                return
            self._on_bought(coin, order, fill, now)
        else:
            pos = self.positions.get(order.mint)
            if pos is None:
                return
            fill = await self.broker.sell(coin, min(order.tokens_raw, pos.tokens_raw))
            self._cash_cache = await self.broker.balance_sol()
            if not fill.ok:
                pos.busy = False  # try again on the next check
                logger.warning(f"Sell {coin.symbol} failed: {fill.error}")
                return
            self._on_sold(coin, pos, order, fill, now)

    def _on_bought(self, coin: CoinState, order: Order, fill: Fill, now: float) -> None:
        pos = Position(
            mint=coin.mint,
            symbol=coin.symbol,
            opened_at=now,
            tokens_raw=fill.tokens_raw,
            cost_sol=fill.sol,
            entry_price=fill.price,
            peak_price=coin.price,
            score=order.score.value if order.score else 0.0,
            reasons=order.score.reasons if order.score else [],
        )
        self.positions[coin.mint] = pos
        self.tracker.pinned.add(coin.mint)
        self.stats["bought"] += 1
        self._log_trade(now, coin, "buy", fill, None, order.reason[:300])
        self._persist(pos)
        logger.info(f"BOUGHT  {coin.symbol:>10} {fill.sol:.4f} SOL at {fill.price:.3e}")

    def _on_sold(
        self, coin: CoinState, pos: Position, order: Order, fill: Fill, now: float
    ) -> None:
        share = fill.tokens_raw / pos.tokens_raw if pos.tokens_raw else 1.0
        cost_part = pos.cost_sol * share
        pnl = fill.sol - cost_part
        pos.realized_sol += pnl
        pos.cost_sol -= cost_part
        pos.tokens_raw -= fill.tokens_raw
        pos.busy = False
        self._log_trade(now, coin, "sell", fill, pnl, order.reason)
        logger.info(
            f"SOLD    {coin.symbol:>10} {fill.sol:.4f} SOL  pnl {pnl:+.4f}  ({order.reason})"
        )
        if pos.tokens_raw <= 0 or pos.cost_sol < 1e-6:
            self.risk.on_close(now, pos.realized_sol)
            del self.positions[coin.mint]
            self.tracker.pinned.discard(coin.mint)
            self.store.remove_position(coin.mint)
        else:
            self._persist(pos)

    def _log_trade(
        self,
        now: float,
        coin: CoinState,
        side: str,
        fill: Fill,
        pnl: float | None,
        reason: str,
    ) -> None:
        self.store.trade(
            {
                "ts": now,
                "run_id": self.run_id,
                "mode": self.mode,
                "mint": coin.mint,
                "symbol": coin.symbol,
                "side": side,
                "sol": fill.sol,
                "tokens_raw": fill.tokens_raw,
                "price": fill.price,
                "fees_sol": fill.fees_sol,
                "pnl_sol": pnl,
                "reason": reason,
                "signature": fill.signature,
            }
        )

    def _persist(self, pos: Position) -> None:
        info = None
        infos = getattr(self.broker, "token_infos", None)
        if infos and pos.mint in infos:
            from trading.position_store import encode_token_info  # noqa: PLC0415

            info = encode_token_info(infos[pos.mint])
        self.store.upsert_position(
            {
                "mint": pos.mint,
                "run_id": self.run_id,
                "mode": self.mode,
                "symbol": pos.symbol,
                "opened_at": pos.opened_at,
                "tokens_raw": pos.tokens_raw,
                "cost_sol": pos.cost_sol,
                "entry_price": pos.entry_price,
                "peak_price": pos.peak_price,
                "tp_done": pos.tp_done,
                "realized_sol": pos.realized_sol,
                "score": pos.score,
                "token_info": info,
            }
        )

    # ------------------------------------------------------------- accounting
    def position_value(self, pos: Position) -> float:
        coin = self.tracker.coins.get(pos.mint)
        if coin is None or pos.tokens_raw <= 0:
            return 0.0
        got, _ = curve_sell(
            coin.v_sol, coin.v_tok, pos.tokens_raw, self.s.costs.fee_bps
        )
        return got

    def _exposure(self) -> float:
        return sum(p.cost_sol for p in self.positions.values()) + sum(
            o.sol for o in self.orders if o.side == "buy"
        )

    def _unrealized(self) -> float:
        return sum(self.position_value(p) - p.cost_sol for p in self.positions.values())

    def _bankroll(self) -> float:
        return self._cash_cache

    def equity(self) -> dict[str, float]:
        open_value = sum(self.position_value(p) for p in self.positions.values())
        return {
            "cash_sol": self._cash_cache,
            "open_value_sol": open_value,
            "equity_sol": self._cash_cache + open_value,
        }

    def _snapshot(self, now: float) -> None:
        self.store.equity(
            {"ts": now, "run_id": self.run_id, "mode": self.mode, **self.equity()}
        )

    def snapshot_now(self, now: float) -> None:
        self._snapshot(now)

    # ------------------------------------------------------------------ misc
    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def position_rows(self) -> list[dict[str, Any]]:
        rows = []
        for p in self.positions.values():
            coin = self.tracker.coins.get(p.mint)
            value = self.position_value(p)
            rows.append(
                {
                    "mint": p.mint,
                    "symbol": p.symbol,
                    "opened_at": p.opened_at,
                    "cost_sol": round(p.cost_sol, 5),
                    "value_sol": round(value, 5),
                    "pnl_pct": round((value / p.cost_sol - 1) * 100, 1)
                    if p.cost_sol
                    else 0.0,
                    "entry_price": p.entry_price,
                    "price": coin.price if coin else None,
                    "progress": round(coin.progress * 100, 1) if coin else None,
                    "tokens": p.tokens_raw / 10**TOKEN_DECIMALS,
                    "tp_done": p.tp_done,
                    "busy": p.busy,
                    "score": round(p.score, 1),
                    "reasons": p.reasons,
                }
            )
        return rows

    async def liquidate_all(self, now: float, reason: str) -> None:
        for pos in list(self.positions.values()):
            if not pos.busy:
                self._sell(pos, now, pos.tokens_raw, reason)
        await self._process_orders(now + self.latency)

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
