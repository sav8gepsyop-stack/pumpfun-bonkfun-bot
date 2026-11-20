"""A synthetic pump.fun market, for tests and for checking the pipeline end to end.

It produces creates and trades that obey the real curve math, from a mix of
coin "stories" (rugs, bundled pumps, duds, organic runners, graduations).
It is NOT evidence of how the strategy does on the real market: the mix and
behaviour are invented. Only a recording of the live feed answers that.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

from ai_trader.events import (
    INITIAL_REAL_TOKEN_RESERVES,
    LAMPORTS,
    CompleteEvt,
    CreateEvt,
    Event,
    TradeEvt,
)
from ai_trader.market import INITIAL_V_SOL, INITIAL_V_TOK

STORIES = {
    # name: (weight, buyers, organic, dev_dumps, bundle, drift)
    # Roughly the real mix: almost everything dies, ~1% graduates.
    "dud": (0.58, 6, 0.2, False, False, -0.3),
    "rug": (0.16, 25, 0.4, True, False, 0.2),
    "bundle_dump": (0.10, 30, 0.1, False, True, 0.3),
    # Looks organic (many fresh wallets, steady inflow), then they all dump.
    "sybil_pump": (0.08, 60, 0.85, False, False, 0.6),
    "runner": (0.07, 120, 0.8, False, False, 0.5),
    "graduate": (0.01, 400, 0.9, False, False, 1.0),
}


def _addr(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_letters + string.digits, k=44))


@dataclass
class _Curve:
    v_sol: int = INITIAL_V_SOL
    v_tok: int = INITIAL_V_TOK
    real_tok: int = INITIAL_REAL_TOKEN_RESERVES

    def buy(self, sol: float) -> int:
        lamports = int(sol * 0.99 * LAMPORTS)
        tokens = min(self.v_tok * lamports // (self.v_sol + lamports), self.real_tok)
        self.v_sol += lamports
        self.v_tok -= tokens
        self.real_tok -= tokens
        return tokens

    def sell(self, tokens: int) -> float:
        lamports = tokens * self.v_sol // (self.v_tok + tokens)
        self.v_sol -= lamports
        self.v_tok += tokens
        self.real_tok += tokens
        return lamports / LAMPORTS


def generate(
    n_coins: int = 300, seed: int = 7, start: float = 1_790_000_000.0
) -> list[Event]:
    rng = random.Random(seed)
    names = list(STORIES)
    weights = [STORIES[n][0] for n in names]
    events: list[Event] = []
    t0 = start
    for _ in range(n_coins):
        t0 += rng.expovariate(1 / 20)  # a launch every ~20 s
        story = rng.choices(names, weights)[0]
        _, n_buyers, organic, dev_dumps, bundle, drift = STORIES[story]
        mint, creator = _addr(rng) + "pump", _addr(rng)
        sym = rng.choice(["PEPE", "DOGE", "CAT", "AI", "TRUMP", "WIF", "MOON"]) + str(
            rng.randint(1, 99)
        )
        events.append(
            CreateEvt(
                t0,
                mint,
                sym.title(),
                sym,
                "",
                creator,
                _addr(rng),
                False,
                sol_quoted=True,
            )
        )
        curve = _Curve()
        holders: dict[str, int] = {}
        t = t0

        def trade(user: str, is_buy: bool, amount: float, ts: float) -> None:
            if is_buy:
                tok = curve.buy(amount)
                if tok <= 0:
                    return
                holders[user] = holders.get(user, 0) + tok
                events.append(
                    TradeEvt(
                        ts,
                        mint,
                        user,
                        True,
                        amount,
                        tok,
                        curve.v_sol,
                        curve.v_tok,
                        curve.real_tok,
                        creator,
                    )
                )
            else:
                tok = min(holders.get(user, 0), int(amount))
                if tok <= 0:
                    return
                holders[user] -= tok
                sol = curve.sell(tok)
                events.append(
                    TradeEvt(
                        ts,
                        mint,
                        user,
                        False,
                        sol,
                        tok,
                        curve.v_sol,
                        curve.v_tok,
                        curve.real_tok,
                        creator,
                    )
                )

        trade(creator, True, rng.uniform(0.2, 2.5 if story != "bundle_dump" else 6), t)
        bundlers = [_addr(rng) for _ in range(rng.randint(3, 6))] if bundle else []
        for b in bundlers:
            trade(b, True, rng.uniform(1, 3), t + 0.4)
        wallets = [_addr(rng) for _ in range(n_buyers)]
        bots = [_addr(rng) for _ in range(3)]
        life = (
            rng.uniform(120, 900)
            if story in ("runner", "graduate")
            else rng.uniform(40, 300)
        )
        dump_at = t + rng.uniform(30, 120)
        dumped = False
        while t < t0 + life and curve.real_tok > 0:
            t += rng.expovariate(1 / (1.0 + 3 * (1 - organic)))
            phase = (t - t0) / life
            if (dev_dumps or bundle or story == "sybil_pump") and not dumped and t >= dump_at:
                dumpers = [creator, *bundlers]
                if story == "sybil_pump":
                    dumpers = [w for w in wallets if holders.get(w)]
                for w in dumpers:
                    trade(w, False, holders.get(w, 0), t)
                dumped = True
                continue
            if rng.random() > organic:  # bot churn: tiny round trips
                b = rng.choice(bots)
                if rng.random() < 0.5 or not holders.get(b):
                    trade(b, True, rng.uniform(0.01, 0.1), t)
                else:
                    trade(b, False, holders[b], t)
                continue
            p_buy = 0.5 + 0.35 * drift * (1 - phase) - (0.25 if dumped else 0)
            w = rng.choice(wallets)
            if rng.random() < p_buy:
                trade(w, True, rng.lognormvariate(-1.2, 0.9), t)
            elif holders.get(w):
                trade(w, False, holders[w] * rng.uniform(0.3, 1.0), t)
        if curve.real_tok <= 0:
            events.append(CompleteEvt(t + 0.5, mint))
        # Late sell-off of whoever is still holding, as interest dies.
        if story != "graduate":
            for w, tok in list(holders.items()):
                if tok and rng.random() < 0.6:
                    t += rng.expovariate(1 / 4)
                    trade(w, False, tok, t)
    events.sort(key=lambda e: e.ts)
    return events
