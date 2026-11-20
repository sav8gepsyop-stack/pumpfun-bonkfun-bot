"""Settings for the AI trader, validated with pydantic and read from one YAML file.

`${VAR}` in any string value is filled from the environment (and `.env`), so
the YAML never holds a secret. The private key is only ever read from the
environment, and only in live mode.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

#: What the operator must set, besides `mode: live`, before real SOL moves.
LIVE_CONFIRM_ENV = "AI_TRADER_LIVE"
LIVE_CONFIRM_VALUE = "I_UNDERSTAND_I_CAN_LOSE_IT_ALL"

_ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)\}")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeedSettings(_Strict):
    """Where market data comes from."""

    wss_endpoint: str = "${SOLANA_NODE_WSS_ENDPOINT}"
    rpc_endpoint: str = "${SOLANA_NODE_RPC_ENDPOINT}"
    #: Write every decoded event to data/feed/*.jsonl.gz, for backtests later.
    record: bool = True
    #: Optional: publish decoded events to NATS (subjects pump.create / pump.trade).
    nats_url: str | None = None


class StrategySettings(_Strict):
    """When to look at a coin, what makes it a buy, and when to get out."""

    #: Seconds to watch a new coin before it may be scored. Flow needs time to show.
    min_age_s: float = 20
    #: Coins older than this are no longer candidates.
    max_age_s: float = 600
    #: Re-score a candidate at most this often.
    rescore_every_s: float = 10
    #: Minimum trades and distinct buyers before a coin is scored at all.
    min_trades: int = 12
    min_unique_buyers: int = 8
    #: Final score (0-100) needed to buy.
    buy_threshold: float = 68
    #: Hard rejects.
    max_top10_share: float = 0.45
    max_dev_buy_share: float = 0.12
    reject_if_creator_sold: bool = True
    reject_mayhem: bool = True
    min_progress: float = 0.03
    max_progress: float = 0.85
    #: Exits.
    stop_loss: float = 0.25
    take_profit: float = 0.60
    take_profit_sell_fraction: float = 0.5
    trailing_stop: float = 0.22
    #: Keep trading every day: when fewer than min_buys_per_day buys have been
    #: made today and none in the last fallback_after_s, the bar drops to
    #: fallback_threshold. Hard rejects, rug checks and risk limits still apply.
    #: Those buys are tagged [fallback] so the report can judge them apart.
    min_buys_per_day: int = 0
    fallback_after_s: float = 3600
    fallback_threshold: float = 55
    #: Optional second take-profit level: at this gain, sell this share of what
    #: is left. 0 turns it off.
    take_profit_2: float = 0.0
    take_profit_2_sell_fraction: float = 0.5
    max_hold_s: float = 900
    #: Sell if nobody has bought for this long and the position is under water.
    stale_after_s: float = 90
    exit_on_creator_sell: bool = True
    #: Pre-rug checks. Entry: skip a coin whose launch was bundled/sniped this
    #: heavily, whose early wallets already sold this much, or whose combined
    #: rug_risk (0..1) is above max_rug_risk_entry.
    max_bundle_share: float = 0.20
    max_early_sold_share: float = 0.50
    max_rug_risk_entry: float = 0.55
    #: Exit: sell everything when rug_risk reaches this while holding, or when
    #: whales dump this share of supply within 30 s.
    exit_rug_risk: float = 0.70
    exit_whale_sold_30s: float = 0.03


class RiskSettings(_Strict):
    """Hard limits. The engine checks every one before each buy."""

    #: Paper mode only: the bankroll the simulation starts with.
    paper_bankroll_sol: float = 2.0
    #: Largest single buy, in SOL, whatever the score.
    max_position_sol: float = 0.1
    #: Largest single buy as a share of the bankroll.
    max_position_pct: float = 0.05
    #: Smallest buy worth sending (fees eat anything smaller).
    min_position_sol: float = 0.01
    max_open_positions: int = 3
    #: All open positions together, in SOL at cost.
    max_total_exposure_sol: float = 0.3
    #: Stop buying for the rest of the UTC day once this much is lost (realized
    #: plus open positions marked to market).
    max_daily_loss_sol: float = 0.3
    max_trades_per_day: int = 40
    #: After this many losing trades in a row, pause buying for cooldown_s.
    max_consecutive_losses: int = 4
    cooldown_s: float = 1800
    #: Never spend below this wallet balance (fees and rent need SOL).
    reserve_sol: float = 0.05
    #: Buy slippage tolerance; the sniper's 30% default is far too loose here.
    buy_slippage: float = 0.15
    sell_slippage: float = 0.25

    @model_validator(mode="after")
    def _sane(self) -> RiskSettings:
        if self.min_position_sol > self.max_position_sol:
            raise ValueError("min_position_sol is above max_position_sol")
        if not 0 < self.max_position_pct <= 0.5:
            raise ValueError("max_position_pct must be in (0, 0.5]")
        for name in (
            "max_position_sol",
            "max_total_exposure_sol",
            "max_daily_loss_sol",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be above 0")
        if self.max_open_positions < 1 or self.max_trades_per_day < 1:
            raise ValueError(
                "max_open_positions and max_trades_per_day must be at least 1"
            )
        if self.reserve_sol < 0:
            raise ValueError("reserve_sol can't be negative")
        return self


class LLMSettings(_Strict):
    """Optional Claude scoring on top of the heuristic model."""

    enabled: bool = False
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high"] = "low"
    #: Share of the final score that comes from Claude (0..1).
    weight: float = 0.35
    #: Only coins whose heuristic score is at least this are sent to Claude.
    min_heuristic_score: float = 55
    max_calls_per_hour: int = 120
    timeout_s: float = 20
    #: Send traces to Langfuse when LANGFUSE_PUBLIC_KEY/SECRET_KEY are set.
    langfuse: bool = True


class SocialSettings(_Strict):
    """Social media and trend signals (see ai_trader/social.py)."""

    enabled: bool = True
    #: Share of the final score that comes from the social/trend score (0..1).
    weight: float = 0.25
    #: Fetch each new coin's metadata JSON for its Twitter/Telegram/website.
    fetch_metadata: bool = True
    #: Gateway used for ipfs URIs (ipfs.io rate-limits hard).
    ipfs_gateway: str = "https://pump.mypinata.cloud"
    metadata_concurrency: int = 6
    http_timeout_s: float = 6
    #: Skip coins with no Twitter, Telegram or website at all.
    require_socials: bool = False
    #: A coin counts as "hot" (its name words become a narrative) at this curve progress.
    hot_progress: float = 0.30
    narrative_window_s: float = 3600
    dexscreener: bool = True
    dexscreener_poll_s: float = 60
    #: Optional X API bearer token for tweet counts; empty turns X off.
    x_bearer_token: str = "${X_BEARER_TOKEN}"
    x_max_calls_per_15min: int = 50


class CostSettings(_Strict):
    """What the paper broker charges, to match what a live trade costs."""

    #: pump.fun protocol + creator fee per side, in basis points.
    fee_bps: int = 125
    #: Network + priority fee per transaction, in SOL.
    tx_cost_sol: float = 0.00003
    #: Seconds between deciding and the order landing; fills use the curve then.
    latency_s: float = 1.5


class GuiSettings(_Strict):
    host: str = "127.0.0.1"
    port: int = 8787
    #: Optional: require this token (?token=...) on every request.
    access_token: str | None = None


class AITraderSettings(_Strict):
    name: str = "ai-trader"
    mode: Literal["paper", "live"] = "paper"
    data_dir: str = "data"
    private_key: str | None = None
    feed: FeedSettings = Field(default_factory=FeedSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    social: SocialSettings = Field(default_factory=SocialSettings)
    costs: CostSettings = Field(default_factory=CostSettings)
    gui: GuiSettings = Field(default_factory=GuiSettings)

    @model_validator(mode="after")
    def _sane(self) -> AITraderSettings:
        if not 0 < self.strategy.stop_loss < 1:
            raise ValueError("strategy.stop_loss must be in (0, 1)")
        if not 0 <= self.social.weight <= 1:
            raise ValueError("social.weight must be in [0, 1]")
        if not 0 <= self.llm.weight <= 1:
            raise ValueError("llm.weight must be in [0, 1]")
        return self

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def kill_file(self) -> Path:
        """Create this file (or press Kill in the GUI) to stop buying and sell all."""
        return self.data_path / "KILL"

    def live_confirmed(self) -> bool:
        return os.environ.get(LIVE_CONFIRM_ENV) == LIVE_CONFIRM_VALUE


def _expand(value: object) -> object:
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_settings(path: str | Path | None = None) -> AITraderSettings:
    """Read the YAML (or defaults when there is none), filling ${VAR}s from the env."""
    load_dotenv(override=False)
    raw: dict = {}
    if path is not None:
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    # The key never lives in the YAML; read it from the environment, live only.
    raw.pop("private_key", None)
    settings = AITraderSettings.model_validate(_expand(raw))
    if settings.mode == "live":
        settings.private_key = os.environ.get("SOLANA_PRIVATE_KEY") or None
    return settings
