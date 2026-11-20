"""Hard risk limits. Nothing is bought unless every one of them says yes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from ai_trader.config import RiskSettings


def utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")


@dataclass
class RiskState:
    day: str = ""
    realized_today: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    cooldown_until: float = 0.0
    paused: bool = False
    killed: bool = False
    halted_reason: str | None = None
    events: list[str] = field(default_factory=list)


class RiskManager:
    def __init__(self, settings: RiskSettings, kill_file: Path) -> None:
        self.s = settings
        self.kill_file = kill_file
        self.state = RiskState()

    def _roll(self, now: float) -> None:
        day = utc_day(now)
        if day != self.state.day:
            self.state.day = day
            self.state.realized_today = 0.0
            self.state.trades_today = 0
            if self.state.halted_reason and self.state.halted_reason.startswith(
                "daily"
            ):
                self.state.halted_reason = None

    def kill_requested(self) -> bool:
        if self.kill_file.exists():
            self.state.killed = True
        return self.state.killed

    def size(
        self,
        *,
        now: float,
        score: float,
        threshold: float,
        bankroll_sol: float,
        open_count: int,
        exposure_sol: float,
        unrealized_sol: float,
    ) -> tuple[float, str | None]:
        """The SOL to spend on a buy, or 0 and why not."""
        self._roll(now)
        s, st = self.s, self.state
        if self.kill_requested():
            return 0, "kill switch is on"
        if st.paused:
            return 0, "paused from the dashboard"
        if now < st.cooldown_until:
            return 0, f"cooling down after {s.max_consecutive_losses} losses in a row"
        day_pnl = st.realized_today + min(unrealized_sol, 0)
        if day_pnl <= -s.max_daily_loss_sol:
            st.halted_reason = "daily loss limit hit"
            return (
                0,
                f"daily loss {day_pnl:.3f} SOL hit the {s.max_daily_loss_sol} SOL limit",
            )
        if st.trades_today >= s.max_trades_per_day:
            return 0, f"{st.trades_today} trades today, the daily limit"
        if open_count >= s.max_open_positions:
            return 0, f"{open_count} positions open, the limit"
        spendable = bankroll_sol - s.reserve_sol
        room = s.max_total_exposure_sol - exposure_sol
        # Scale from half size at the threshold to full size at 100.
        conviction = 0.5 + 0.5 * max(
            0.0, min(1.0, (score - threshold) / max(100 - threshold, 1))
        )
        want = min(s.max_position_sol, bankroll_sol * s.max_position_pct) * conviction
        # Never let one more loss breach the daily limit by more than one stop.
        left_today = s.max_daily_loss_sol + day_pnl
        amount = min(want, room, spendable, max(left_today, 0) * 4)
        if amount < s.min_position_sol:
            return 0, f"size {amount:.4f} SOL below the {s.min_position_sol} minimum"
        return round(amount, 4), None

    def on_buy(self, now: float) -> None:
        self._roll(now)
        self.state.trades_today += 1

    def on_close(self, now: float, pnl_sol: float) -> None:
        self._roll(now)
        st = self.state
        st.realized_today += pnl_sol
        if pnl_sol < 0:
            st.consecutive_losses += 1
            if st.consecutive_losses >= self.s.max_consecutive_losses:
                st.cooldown_until = now + self.s.cooldown_s
                st.consecutive_losses = 0
                st.events.append(f"{time.strftime('%H:%M:%S')} cooldown started")
        else:
            st.consecutive_losses = 0
