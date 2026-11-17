# AI trader

`ai_trader` watches every pump.fun launch and trade, scores each young coin
from its on-chain behaviour (optionally with Claude's opinion on top), buys the
few that pass, and manages the exit. It runs in **paper mode by default**:
real market data, simulated fills, no SOL spent.

> **Read this first.** Almost every pump.fun coin goes to zero, many within
> minutes, and a large share are coordinated pump-and-dumps built to look like
> organic demand. No scoring model, AI or otherwise, can promise a profit here.
> This bot is a disciplined way to take small, capped bets and measure whether
> they pay. Paper trade first, read the report, and only put in money you can
> lose entirely.

## Setup

1. Install (see the main README): `uv sync && uv pip install -e .`
2. `cp .env.example .env`, then fill in at least
   `SOLANA_NODE_RPC_ENDPOINT` and `SOLANA_NODE_WSS_ENDPOINT` (a Chainstack
   Solana mainnet node works; public endpoints drop connections).
3. For live mode later, create a **dedicated** trading wallet:
   `pump_bot_wallet` (or any new keypair) and put its key in `.env` as
   `SOLANA_PRIVATE_KEY`. Never reuse your main wallet's key. Nobody, including
   this bot's dashboard, ever needs to see it; never paste it into a chat.
4. Optional: `ANTHROPIC_API_KEY` to turn on Claude scoring (`llm.enabled: true`
   in `config/ai-trader.yaml`), and `LANGFUSE_*` to trace those calls.

Settings live in [`config/ai-trader.yaml`](../config/ai-trader.yaml); every
setting is commented. It sits outside `bots/` so `pump_bot` never picks it up.

## Commands

```bash
ai_trader demo                 # the dashboard on an invented market; needs nothing set up
ai_trader run                  # paper trade the live market + dashboard on http://127.0.0.1:8787
ai_trader report               # report on the latest paper or live run
ai_trader backtest             # replay what `run` recorded to data/feed/, with paper fills
ai_trader backtest --synthetic 3000   # plumbing check on an invented market
ai_trader address              # the trading wallet's public address
ai_trader kill [--off]         # sell everything and stop buying / re-arm
ai_trader withdraw --to <your wallet> --amount 0.5|all
```

## The dashboard

`ai_trader run` serves it on `127.0.0.1:8787` (localhost only by default).

- **Fund the trading wallet**: the wallet's public address, a QR code
  (`solana:` URI, scannable by mobile wallets), its balance in SOL and USD, and
  **Deposit with Phantom**, which builds a plain SOL transfer that you approve
  in Phantom. The private key never reaches the page. Taking money out is a
  CLI command (`ai_trader withdraw`), never a button.
- **Risk controls**: pause buying, the kill switch (sells everything, blocks
  buys until re-armed), and the risk limits, editable for the running session.
- Equity chart, open positions with the reasons each was bought, the latest AI
  decisions (buy / pass / blocked and why), and every trade with its Solscan link.

POSTs from other origins are refused. If you bind it to anything but
localhost, set `gui.access_token` and open `/?token=...`.

## How it decides

1. **Feed.** One `logsSubscribe` on the pump.fun program delivers every
   create, buy, sell and graduation. Each TradeEvent carries the curve's
   reserves, so prices, flows and per-wallet balances come from the stream
   itself, with no RPC call per coin. The feed is recorded to
   `data/feed/*.jsonl.gz` for backtests.
2. **Features** per coin (after `min_age_s`): distinct buyers and sellers,
   net SOL inflow (total and last 30 s), buy/sell ratio, 30 s return, trade
   pace vs. its average, drawdown from peak, curve progress, top-1 and top-10
   holder shares, dev buy size, whether the creator sold, wallet churn (bots
   and wash trading), same-symbol clones, and the creator's record across the
   coins the bot has seen them launch.
3. **Hard rejects** first: mayhem mode, creator sold, too few trades or
   buyers, concentrated holders, a big dev buy, curve too empty or nearly
   graduated, a creator who rugged before, or a move already half reversed.
4. **Heuristic score** (0-100): weighted evidence for broad, fresh,
   accelerating demand with a safe holder spread. Every score keeps its top
   reasons, shown on the dashboard.
5. **Claude** (optional): coins scoring at least `llm.min_heuristic_score` go
   to Claude with the same numbers, asking for a calibrated 0-100 score,
   verdict and red flags as structured JSON. The final score blends the two by
   `llm.weight`. Calls are capped per hour; coin names are passed as untrusted
   text. Model, effort and weight are in the config.
6. **Risk manager** sizes the buy (half size at the threshold, full size near
   100) and refuses it when any limit is hit.
7. **Exits**: stop loss, take profit on part of the position then a trailing
   stop on the rest, max hold time, no buyers for `stale_after_s` while under
   water, the creator selling, or graduation.

### Risk limits

| Setting | Default | What it does |
|---|---|---|
| `max_position_sol` / `max_position_pct` | 0.1 / 5% | Largest single buy |
| `max_open_positions` | 3 | Concurrent positions |
| `max_total_exposure_sol` | 0.3 | All open positions together, at cost |
| `max_daily_loss_sol` | 0.3 | Realized + open losses per UTC day; buying stops when hit |
| `max_trades_per_day` | 40 | Buys per UTC day |
| `max_consecutive_losses` / `cooldown_s` | 4 / 30 min | Pause after a losing streak |
| `reserve_sol` | 0.05 | Never spend below this balance (fees, rent) |
| kill switch | | `data/KILL`, the dashboard button or `ai_trader kill` |

## Paper vs. live

Paper fills use the exact pump.fun curve formula (it reproduces recorded
on-chain buys to the raw unit, see `tests/regression/verify_ai_trader.py`) on
the curve as it stands `costs.latency_s` after the decision, plus the protocol
and creator fee (`fee_bps`) and a network fee per transaction. Your own size
moves the price on both legs, so paper is slightly pessimistic. What paper
can't show: failed or late transactions, MEV, and slippage beyond the curve.

Live mode sends orders through the repo's own `PlatformAwareBuyer` and
`PlatformAwareSeller` (the sniper's transaction code), with the buy built from
the coin's create logs, so a buy needs no extra RPC call. A graduated coin's
sell goes to PumpSwap automatically. Open live positions are kept in
`data/<name>.duckdb` and resume after a restart; Ctrl+C does not sell them, the
kill switch does.

To go live, all three are needed:

1. `mode: live` in the config (or `ai_trader run --live`),
2. `SOLANA_PRIVATE_KEY` in `.env`,
3. `AI_TRADER_LIVE=I_UNDERSTAND_I_CAN_LOSE_IT_ALL` in `.env`.

Before that: run paper mode for at least a few days, run `ai_trader report`,
and look at the verdict, the profit factor, and the "By entry score" table. If
higher scores don't earn more than lower ones, the scorer has no edge on the
real market and live trading will lose the fees on every trade. Start live
with the smallest sizes and compare live fills with paper.

## Data

`data/` (git-ignored) holds `<name>.duckdb` with tables `decisions` (every
score and why), `trades`, `positions`, `equity` and `creators`, the feed
recordings, backtests and reports. Query it with DuckDB or polars:

```python
import duckdb
duckdb.connect("data/ai-trader.duckdb", read_only=True).sql(
    "SELECT action, count(*) FROM decisions GROUP BY 1").show()
```

(Stop the trader first or copy the file; DuckDB allows one writer.)

Set `feed.nats_url` to publish every decoded event to NATS
(`pump.create`, `pump.trade`, `pump.complete`) for your own consumers.

## Libraries

| Library | Used for |
|---|---|
| pydantic | Validated settings (`config.py`) |
| tenacity | Feed reconnects with backoff |
| polars | Report statistics and frames |
| duckdb | The decision, trade, position and equity store |
| anthropic | Claude scoring |
| langfuse | Optional traces of every Claude call |
| nats-py | Optional event bus |
| ccxt | SOL/USD price for the dashboard |
| Lightweight Charts | Equity charts in the dashboard and reports (from a CDN) |

cryptofeed and freqtrade are deliberately not used: both trade centralized
exchanges, and neither can see or trade pump.fun's on-chain curves.
