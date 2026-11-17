"""pump.fun events the trader reacts to, decoded from transaction logs.

Every pump.fun buy and sell emits a TradeEvent carrying the curve's reserves
after the trade, so the price, flow and holder balances of every coin can be
followed from the log stream alone, with no RPC call per coin.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utils.idl_parser import IDLParser

#: A pump.fun coin has 6 decimals; SOL has 9.
TOKEN_DECIMALS = 6
LAMPORTS = 1_000_000_000
#: Tokens the curve sells before the coin graduates (raw units).
INITIAL_REAL_TOKEN_RESERVES = 793_100_000 * 10**TOKEN_DECIMALS
TOTAL_SUPPLY = 1_000_000_000 * 10**TOKEN_DECIMALS

#: The quote mint a SOL-paired coin reports: Pubkey::default(), or wrapped SOL.
SOL_QUOTES = {
    "11111111111111111111111111111111",
    "So11111111111111111111111111111111111111112",
}

_IDL = Path(__file__).resolve().parents[2] / "idl" / "pump_fun_idl.json"
_parser: IDLParser | None = None


def _idl() -> IDLParser:
    global _parser  # noqa: PLW0603 - one parser per process, built on first use
    if _parser is None:
        _parser = IDLParser(str(_IDL))
    return _parser


@dataclass(frozen=True)
class CreateEvt:
    ts: float
    mint: str
    name: str
    symbol: str
    uri: str
    creator: str
    bonding_curve: str
    is_mayhem: bool
    sol_quoted: bool
    signature: str = ""
    kind: str = field(default="create", init=False)


@dataclass(frozen=True)
class TradeEvt:
    ts: float
    mint: str
    user: str
    is_buy: bool
    sol: float  # SOL in (buy) or out (sell), before fees
    tokens: int  # raw token units
    v_sol: int  # virtual SOL reserves after the trade, lamports
    v_tok: int  # virtual token reserves after the trade, raw
    real_tok: int  # real token reserves after the trade, raw
    creator: str
    signature: str = ""
    kind: str = field(default="trade", init=False)

    @property
    def price(self) -> float:
        """SOL per whole token after this trade."""
        return price_from_reserves(self.v_sol, self.v_tok)

    @property
    def progress(self) -> float:
        """Share of the curve's tokens sold: 1.0 is graduation."""
        return 1 - self.real_tok / INITIAL_REAL_TOKEN_RESERVES


@dataclass(frozen=True)
class CompleteEvt:
    ts: float
    mint: str
    signature: str = ""
    kind: str = field(default="complete", init=False)


Event = CreateEvt | TradeEvt | CompleteEvt


def price_from_reserves(v_sol: int, v_tok: int) -> float:
    if v_tok <= 0:
        return 0.0
    return (v_sol / LAMPORTS) / (v_tok / 10**TOKEN_DECIMALS)


def to_dict(evt: Event) -> dict[str, Any]:
    return asdict(evt)


def from_dict(d: dict[str, Any]) -> Event:
    d = dict(d)
    kind = d.pop("kind")
    cls = {"create": CreateEvt, "trade": TradeEvt, "complete": CompleteEvt}[kind]
    return cls(**d)


def decode_logs(logs: list[str], signature: str, received_at: float) -> list[Event]:
    """Every create, trade and graduation in one transaction's logs, in order.

    Events are stamped with `received_at` (wall clock, sub-second), not the
    block time, which has one-second resolution.

    Coins paired against anything but SOL are reported (sol_quoted False on the
    create) but their trades are dropped, since the trader only spends SOL.
    """
    out: list[Event] = []
    parser = _idl()
    for line in logs:
        if not line.startswith("Program data: "):
            continue
        try:
            raw = base64.b64decode(line[len("Program data: ") :])
        except ValueError:
            continue
        decoded = parser.decode_event_data(raw)
        if not decoded:
            continue
        name, f = decoded["event_name"], decoded["fields"]
        try:
            if name == "CreateEvent":
                out.append(
                    CreateEvt(
                        ts=received_at,
                        mint=str(f["mint"]),
                        name=str(f.get("name", "")),
                        symbol=str(f.get("symbol", "")),
                        uri=str(f.get("uri", "")),
                        creator=str(f.get("creator") or f.get("user")),
                        bonding_curve=str(f.get("bonding_curve", "")),
                        is_mayhem=bool(f.get("is_mayhem_mode", False)),
                        sol_quoted=str(f.get("quote_mint", SOL_QUOTES_DEFAULT))
                        in SOL_QUOTES,
                        signature=signature,
                    )
                )
            elif name == "TradeEvent":
                if str(f.get("quote_mint", SOL_QUOTES_DEFAULT)) not in SOL_QUOTES:
                    continue
                out.append(
                    TradeEvt(
                        ts=received_at,
                        mint=str(f["mint"]),
                        user=str(f["user"]),
                        is_buy=bool(f["is_buy"]),
                        sol=int(f["sol_amount"]) / LAMPORTS,
                        tokens=int(f["token_amount"]),
                        v_sol=int(f["virtual_sol_reserves"]),
                        v_tok=int(f["virtual_token_reserves"]),
                        real_tok=int(f["real_token_reserves"]),
                        creator=str(f.get("creator", "")),
                        signature=signature,
                    )
                )
            elif name == "CompleteEvent":
                out.append(
                    CompleteEvt(
                        ts=received_at,
                        mint=str(f["mint"]),
                        signature=signature,
                    )
                )
        except (KeyError, TypeError, ValueError):
            continue
    return out


SOL_QUOTES_DEFAULT = "11111111111111111111111111111111"
