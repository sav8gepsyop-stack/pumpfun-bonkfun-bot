"""Where orders go: a paper broker that fills on the real curve math, or the chain.

Both return the same `Fill`, so the engine, the risk limits, the store and the
dashboard can't tell paper from live, which is what makes a paper run a fair
rehearsal.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ai_trader.events import LAMPORTS, TOKEN_DECIMALS
from utils.logger import get_logger

if TYPE_CHECKING:
    from ai_trader.config import AITraderSettings
    from ai_trader.market import CoinState
    from interfaces.core import TokenInfo

logger = get_logger(__name__)


@dataclass(frozen=True)
class Fill:
    ok: bool
    tokens_raw: int = 0
    #: SOL that left (buy, positive) or reached (sell, positive) the wallet,
    #: fees included.
    sol: float = 0.0
    price: float = 0.0
    fees_sol: float = 0.0
    signature: str | None = None
    error: str | None = None


def curve_buy(
    v_sol: int, v_tok: int, real_tok: int, sol_in: float, fee_bps: int
) -> tuple[int, float]:
    """Raw tokens a buy of `sol_in` SOL gets on a pump.fun curve, and the fee paid.

    The program prices the buy on one lamport less than it takes in; this
    reproduces recorded on-chain fills to the raw unit.
    """
    fee = sol_in * fee_bps / 10_000
    lamports = int((sol_in - fee) * LAMPORTS) - 1
    if lamports <= 0:
        return 0, fee
    tokens = v_tok * lamports // (v_sol + lamports)
    return min(tokens, real_tok), fee


def curve_sell(
    v_sol: int, v_tok: int, tokens_raw: int, fee_bps: int
) -> tuple[float, float]:
    """SOL a sell of `tokens_raw` gets on the curve, after fees, and the fee paid."""
    gross = (tokens_raw * v_sol // (v_tok + tokens_raw)) / LAMPORTS
    fee = gross * fee_bps / 10_000
    return gross - fee, fee


class Broker(Protocol):
    live: bool

    async def balance_sol(self) -> float: ...
    async def buy(self, coin: CoinState, sol: float) -> Fill: ...
    async def sell(self, coin: CoinState, tokens_raw: int) -> Fill: ...
    async def close(self) -> None: ...


class PaperBroker:
    """Fills at the coin's current curve state; charges fees and network cost.

    The engine waits `costs.latency_s` between deciding and calling the broker,
    so a fill uses the curve as it was when a real order would have landed,
    including whatever other traders did in between. Our own size moves the
    price along the curve (slippage) on both the buy and the sell, but is not
    fed back into the market, so a paper round trip pays the price impact
    twice: paper results are slightly pessimistic, by design.
    """

    live = False

    def __init__(self, settings: AITraderSettings) -> None:
        self.c = settings.costs
        self.cash = settings.risk.paper_bankroll_sol

    async def balance_sol(self) -> float:
        return self.cash

    async def buy(self, coin: CoinState, sol: float) -> Fill:
        if coin.completed:
            return Fill(ok=False, error="coin graduated before the fill")
        if sol + self.c.tx_cost_sol > self.cash:
            return Fill(ok=False, error="paper wallet too small")
        tokens, fee = curve_buy(
            coin.v_sol, coin.v_tok, coin.real_tok, sol, self.c.fee_bps
        )
        if tokens <= 0:
            return Fill(ok=False, error="curve has no tokens left")
        spent = sol + self.c.tx_cost_sol
        self.cash -= spent
        return Fill(
            ok=True,
            tokens_raw=tokens,
            sol=spent,
            price=sol / (tokens / 10**TOKEN_DECIMALS),
            fees_sol=fee + self.c.tx_cost_sol,
        )

    async def sell(self, coin: CoinState, tokens_raw: int) -> Fill:
        got, fee = curve_sell(coin.v_sol, coin.v_tok, tokens_raw, self.c.fee_bps)
        got -= self.c.tx_cost_sol
        self.cash += got
        return Fill(
            ok=True,
            tokens_raw=tokens_raw,
            sol=got,
            price=(got + self.c.tx_cost_sol) / (tokens_raw / 10**TOKEN_DECIMALS)
            if tokens_raw
            else 0,
            fees_sol=fee + self.c.tx_cost_sol,
        )

    async def close(self) -> None:
        return None


class LiveBroker:
    """Sends real transactions through the repo's own buyer and seller."""

    live = True

    def __init__(self, settings: AITraderSettings) -> None:
        # Imported here so paper mode never loads wallet or signing code.
        from core.client import SolanaClient  # noqa: PLC0415
        from core.priority_fee.manager import PriorityFeeManager  # noqa: PLC0415
        from core.wallet import Wallet  # noqa: PLC0415

        if not settings.private_key:
            raise ValueError("live mode needs SOLANA_PRIVATE_KEY in the environment")  # noqa: TRY003
        rpc = settings.feed.rpc_endpoint
        if not rpc:
            raise ValueError("live mode needs SOLANA_NODE_RPC_ENDPOINT")  # noqa: TRY003
        self.s = settings
        self.client = SolanaClient(rpc, max_rps=25)
        self.wallet = Wallet(settings.private_key)
        self.fees = PriorityFeeManager(
            client=self.client,
            enable_dynamic_fee=False,
            enable_fixed_fee=True,
            fixed_fee=200_000,
            extra_fee=0.0,
            hard_cap=400_000,
        )
        #: TokenInfo per mint, filled by the feed from each coin's create logs.
        self.token_infos: dict[str, TokenInfo] = {}
        self._lock = asyncio.Lock()

    @property
    def address(self) -> str:
        return str(self.wallet.pubkey)

    async def balance_sol(self) -> float:
        return await self.client.get_sol_balance(self.wallet.pubkey) / LAMPORTS

    def _info(self, coin: CoinState) -> TokenInfo | None:
        return self.token_infos.get(coin.mint)

    async def buy(self, coin: CoinState, sol: float) -> Fill:
        from trading.platform_aware import PlatformAwareBuyer  # noqa: PLC0415

        info = self._info(coin)
        if info is None:
            return Fill(
                ok=False, error="no create data for this coin; can't build the buy"
            )
        buyer = PlatformAwareBuyer(
            self.client,
            self.wallet,
            self.fees,
            amount=sol,
            slippage=self.s.risk.buy_slippage,
            max_retries=2,
            extreme_fast_mode=False,
        )
        async with self._lock:
            result = await buyer.execute(info)
        if not result.success:
            return Fill(
                ok=False,
                error=str(result.failure_reason or result.error_message),
                signature=result.tx_signature,
            )
        d = result.deltas
        spent = abs(d.quote_raw) / LAMPORTS if d and d.quote_raw else sol
        if d:
            spent += d.network_fee_lamports / LAMPORTS
        tokens = result.amount_raw or int((result.amount or 0) * 10**TOKEN_DECIMALS)
        return Fill(
            ok=True,
            tokens_raw=tokens,
            sol=spent,
            price=result.price or 0.0,
            signature=result.tx_signature,
        )

    async def sell(self, coin: CoinState, tokens_raw: int) -> Fill:
        from trading.platform_aware import PlatformAwareSeller  # noqa: PLC0415

        info = self._info(coin)
        if info is None:
            return Fill(ok=False, error="no create data for this coin")
        seller = PlatformAwareSeller(
            self.client,
            self.wallet,
            self.fees,
            slippage=self.s.risk.sell_slippage,
            max_retries=2,
        )
        async with self._lock:
            result = await seller.execute(
                info,
                token_amount=tokens_raw / 10**TOKEN_DECIMALS,
                token_price=coin.price,
                token_amount_raw=tokens_raw,
            )
        if not result.success:
            return Fill(
                ok=False,
                error=str(result.failure_reason or result.error_message),
                signature=result.tx_signature,
            )
        d = result.deltas
        got = (
            d.quote_raw / LAMPORTS
            if d and d.quote_raw
            else (result.amount or 0) * (result.price or 0)
        )
        if d:
            got -= d.network_fee_lamports / LAMPORTS
        return Fill(
            ok=True,
            tokens_raw=tokens_raw,
            sol=got,
            price=result.price or 0.0,
            signature=result.tx_signature,
        )

    async def close(self) -> None:
        await self.client.close()
