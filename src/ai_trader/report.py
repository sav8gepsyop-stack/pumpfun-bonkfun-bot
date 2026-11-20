"""Performance report for one run (paper, live or backtest), computed with polars.

Writes `report.md`, `report.json` and a self-contained `report.html` (equity
chart via TradingView Lightweight Charts from a CDN) next to the store.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from ai_trader.store import Store

LWC = "https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"


def _round_trips(trades: pl.DataFrame) -> pl.DataFrame:
    if trades.is_empty():
        return pl.DataFrame()
    buys = (
        trades.filter(pl.col("side") == "buy")
        .group_by("mint")
        .agg(
            pl.col("symbol").first(),
            pl.col("ts").min().alias("opened"),
            pl.col("sol").sum().alias("cost"),
        )
    )
    sells = (
        trades.filter(pl.col("side") == "sell")
        .group_by("mint")
        .agg(
            pl.col("ts").max().alias("closed"),
            pl.col("sol").sum().alias("proceeds"),
            pl.col("pnl_sol").sum().alias("pnl"),
            pl.col("reason").last().alias("exit_reason"),
        )
    )
    rt = buys.join(sells, on="mint", how="inner").with_columns(
        (pl.col("pnl") / pl.col("cost")).alias("ret"),
        (pl.col("closed") - pl.col("opened")).alias("hold_s"),
        pl.col("exit_reason").str.replace(r" [-+].*$", "").alias("exit_kind"),
    )
    return rt.sort("opened")


def compute(store: Store, run_id: str, start_cash: float) -> dict[str, Any]:
    trades = store.frame("SELECT * FROM trades WHERE run_id = ? ORDER BY ts", [run_id])
    equity = store.frame(
        "SELECT ts, equity_sol FROM equity WHERE run_id = ? ORDER BY ts", [run_id]
    )
    decisions = store.frame(
        "SELECT action, count(*) AS n FROM decisions WHERE run_id = ? GROUP BY action",
        [run_id],
    )
    buy_scores = store.frame(
        "SELECT mint, max(score) AS score, "
        "bool_or(reason LIKE '%fallback%') AS fallback, "
        "bool_or(reason LIKE '%mayhem%') AS mayhem, "
        "max(TRY_CAST(json_extract(features, '$.social_score') AS DOUBLE)) AS social "
        "FROM decisions WHERE run_id = ? AND action = 'buy' GROUP BY mint",
        [run_id],
    )
    rt = _round_trips(trades)
    out: dict[str, Any] = {"run_id": run_id, "start_cash_sol": start_cash}
    out["decisions"] = {r["action"]: r["n"] for r in decisions.to_dicts()}
    n = rt.height
    out["round_trips"] = n
    if n == 0:
        out["verdict"] = "No completed trades yet: nothing to judge."
        out["trades"] = []
        out["equity"] = equity.to_dicts()
        return out
    wins = rt.filter(pl.col("pnl") > 0)
    losses = rt.filter(pl.col("pnl") <= 0)
    gross_win = float(wins["pnl"].sum()) if wins.height else 0.0
    gross_loss = -float(losses["pnl"].sum()) if losses.height else 0.0
    total = float(rt["pnl"].sum())
    fees = float(trades["fees_sol"].sum())
    out |= {
        "net_pnl_sol": total,
        "return_pct": 100 * total / start_cash if start_cash else None,
        "win_rate": wins.height / n,
        "avg_win_sol": gross_win / wins.height if wins.height else 0.0,
        "avg_loss_sol": -gross_loss / losses.height if losses.height else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss else math.inf,
        "expectancy_sol": total / n,
        "avg_return_per_trade": float(rt["ret"].mean()),
        "median_hold_s": float(rt["hold_s"].median()),
        "fees_sol": fees,
        "best": rt.sort("pnl", descending=True)
        .head(1)
        .select("symbol", "pnl")
        .to_dicts(),
        "worst": rt.sort("pnl").head(1).select("symbol", "pnl").to_dicts(),
    }
    # Max drawdown on realized equity (open positions marked at the snapshots too).
    if equity.height:
        eq = equity["equity_sol"]
        peak = eq.cum_max()
        out["max_drawdown_pct"] = float(((peak - eq) / peak).max() * 100)
    else:
        out["max_drawdown_pct"] = None
    by_exit = (
        rt.group_by("exit_kind")
        .agg(
            pl.len().alias("n"),
            pl.col("pnl").sum().alias("pnl"),
            (pl.col("pnl") > 0).mean().alias("win_rate"),
        )
        .sort("n", descending=True)
    )
    out["by_exit"] = by_exit.to_dicts()
    # Does a higher score mean a better trade? If not, the scorer has no edge.
    scored = rt.join(buy_scores, on="mint", how="left").with_columns(
        (pl.col("score") // 10 * 10).alias("bucket")
    )
    out["by_score"] = (
        scored.group_by("bucket")
        .agg(
            pl.len().alias("n"),
            pl.col("ret").mean().alias("avg_ret"),
            (pl.col("pnl") > 0).mean().alias("win_rate"),
        )
        .sort("bucket")
        .to_dicts()
    )
    # Do the daily-activity (fallback) entries and the social signals pay?
    def split(col: pl.Expr, name: str) -> list[dict[str, Any]]:
        return (
            scored.with_columns(col.alias(name))
            .group_by(name)
            .agg(
                pl.len().alias("n"),
                pl.col("pnl").sum().alias("pnl"),
                pl.col("ret").mean().alias("avg_ret"),
                (pl.col("pnl") > 0).mean().alias("win_rate"),
            )
            .sort(name)
            .to_dicts()
        )

    out["by_entry"] = split(
        pl.when(pl.col("mayhem").fill_null(False))
        .then(pl.lit("mayhem scalp"))
        .when(pl.col("fallback").fill_null(False))
        .then(pl.lit("fallback"))
        .otherwise(pl.lit("normal")),
        "entry",
    )
    out["by_social"] = split(
        pl.when(pl.col("social").is_null())
        .then(pl.lit("n/a"))
        .when(pl.col("social") >= 0.4)
        .then(pl.lit("strong (>=0.4)"))
        .otherwise(pl.lit("weak (<0.4)")),
        "social_bucket",
    )
    out["trades"] = rt.select(
        "symbol", "mint", "opened", "cost", "pnl", "ret", "hold_s", "exit_reason"
    ).to_dicts()
    out["equity"] = equity.to_dicts()
    out["verdict"] = _verdict(out)
    return out


def _verdict(r: dict[str, Any]) -> str:
    n, pf = r["round_trips"], r["profit_factor"]
    if n < 100:
        return (
            f"Only {n} completed trades: far too few to tell skill from luck. Keep paper "
            "trading until there are several hundred before risking real SOL."
        )
    if r["net_pnl_sol"] <= 0 or pf < 1.0:
        return "Loses money after fees. Do not go live with these settings."
    if pf < 1.3:
        return (
            "Slightly profitable on paper, but the edge is thin enough that live slippage, "
            "failed transactions and latency can erase it. Not ready for real money."
        )
    return (
        "Profitable on paper with a meaningful edge. If you go live, start with the "
        "smallest sizes, keep the limits tight, and compare live fills against this run."
    )


def _fmt(x: Any, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if math.isinf(x):
            return "inf"
        return f"{x * 100:.1f}%" if pct else f"{x:.4f}"
    return str(x)


def write(report: dict[str, Any], out_dir: Path, title: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "report.json",
        "md": out_dir / "report.md",
        "html": out_dir / "report.html",
    }
    paths["json"].write_text(json.dumps(report, indent=2, default=str))
    lines = [
        f"# {title}",
        "",
        f"**Verdict:** {report['verdict']}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key, label, pct in [
        ("round_trips", "Completed trades", False),
        ("net_pnl_sol", "Net P&L (SOL)", False),
        ("return_pct", "Return on starting bankroll (%)", False),
        ("win_rate", "Win rate", True),
        ("avg_win_sol", "Average win (SOL)", False),
        ("avg_loss_sol", "Average loss (SOL)", False),
        ("profit_factor", "Profit factor", False),
        ("expectancy_sol", "Expectancy per trade (SOL)", False),
        ("max_drawdown_pct", "Max drawdown (%)", False),
        ("fees_sol", "Fees paid (SOL)", False),
        ("median_hold_s", "Median hold (s)", False),
    ]:
        lines.append(f"| {label} | {_fmt(report.get(key), pct)} |")
    lines += ["", f"Decisions: {report.get('decisions')}", ""]
    if report.get("by_exit"):
        lines += [
            "## By exit reason",
            "",
            "| Exit | Trades | P&L (SOL) | Win rate |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {r['exit_kind']} | {r['n']} | {r['pnl']:.4f} | {r['win_rate']:.0%} |"
            for r in report["by_exit"]
        ]
    if report.get("by_score"):
        lines += [
            "",
            "## By entry score (is the scorer calibrated?)",
            "",
            "| Score | Trades | Avg return | Win rate |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {_fmt(r['bucket'])} | {r['n']} | {_fmt(r['avg_ret'], True)} | {_fmt(r['win_rate'], True)} |"
            for r in report["by_score"]
        ]
    for key, label, col in (
        ("by_entry", "By entry type (do forced daily trades and mayhem scalps pay?)", "entry"),
        ("by_social", "By social/trend score (do the signals help?)", "social_bucket"),
    ):
        if report.get(key):
            lines += [
                "",
                f"## {label}",
                "",
                "| Group | Trades | P&L SOL | Avg return | Win rate |",
                "|---|---|---|---|---|",
            ]
            lines += [
                f"| {r[col]} | {r['n']} | {r['pnl']:.4f} | {_fmt(r['avg_ret'], True)} | {_fmt(r['win_rate'], True)} |"
                for r in report[key]
            ]
    paths["md"].write_text("\n".join(lines) + "\n")
    paths["html"].write_text(_html(report, title, "\n".join(lines)))
    return paths


def _html(report: dict[str, Any], title: str, md: str) -> str:
    eq = [
        {"time": int(r["ts"]), "value": round(r["equity_sol"], 6)}
        for r in report.get("equity", [])
    ]
    dedup: dict[int, dict] = {p["time"]: p for p in eq}
    rows = "".join(
        f"<tr><td>{html.escape(str(t['symbol']))}</td><td>{t['cost']:.4f}</td><td class={'up' if t['pnl'] > 0 else 'down'}>{t['pnl']:+.4f}</td>"
        f"<td>{t['ret'] * 100:+.1f}%</td><td>{t['hold_s']:.0f}s</td><td>{html.escape(str(t['exit_reason']))}</td></tr>"
        for t in report.get("trades", [])[-300:]
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><script src="{LWC}"></script>
<style>body{{font:14px system-ui,sans-serif;margin:0 auto;max-width:1000px;padding:16px;background:#0f1115;color:#e6e6e6}}
pre{{white-space:pre-wrap;background:#171a21;padding:12px;border-radius:8px}}table{{border-collapse:collapse;width:100%}}
td,th{{padding:4px 8px;border-bottom:1px solid #2a2f3a;text-align:left}}.up{{color:#3fb950}}.down{{color:#f85149}}
#chart{{height:320px}}</style></head><body><h1>{html.escape(title)}</h1>
<div id="chart"></div><pre>{html.escape(md)}</pre><h2>Trades (latest 300)</h2>
<table><tr><th>Coin</th><th>Cost</th><th>P&amp;L</th><th>Return</th><th>Held</th><th>Exit</th></tr>{rows}</table>
<script>const data={json.dumps(sorted(dedup.values(), key=lambda p: p["time"]))};
if(window.LightweightCharts&&data.length){{const c=LightweightCharts.createChart(document.getElementById('chart'),{{layout:{{background:{{color:'#0f1115'}},textColor:'#ccc'}},grid:{{vertLines:{{color:'#1e222b'}},horzLines:{{color:'#1e222b'}}}},timeScale:{{timeVisible:true}}}});
c.addAreaSeries({{lineColor:'#58a6ff',topColor:'rgba(88,166,255,.3)',bottomColor:'rgba(88,166,255,0)'}}).setData(data);c.timeScale().fitContent();}}</script>
</body></html>"""
