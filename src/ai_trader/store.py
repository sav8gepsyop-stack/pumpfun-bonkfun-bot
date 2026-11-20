"""Everything the trader decides and does, in one DuckDB file per bot.

Tables: decisions (every score), trades (every fill), positions (what is open,
so a live run survives a restart), equity (snapshots for the chart), creators
(what each creator's coins did). The dashboard and the report read from here,
and polars turns it into statistics.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    ts DOUBLE, run_id VARCHAR, mint VARCHAR, symbol VARCHAR, score DOUBLE,
    heuristic DOUBLE, llm DOUBLE, action VARCHAR, reason VARCHAR, features JSON
);
CREATE TABLE IF NOT EXISTS trades (
    ts DOUBLE, run_id VARCHAR, mode VARCHAR, mint VARCHAR, symbol VARCHAR,
    side VARCHAR, sol DOUBLE, tokens_raw BIGINT, price DOUBLE, fees_sol DOUBLE,
    pnl_sol DOUBLE, reason VARCHAR, signature VARCHAR
);
CREATE TABLE IF NOT EXISTS positions (
    mint VARCHAR PRIMARY KEY, run_id VARCHAR, mode VARCHAR, symbol VARCHAR,
    opened_at DOUBLE, tokens_raw BIGINT, cost_sol DOUBLE, entry_price DOUBLE,
    peak_price DOUBLE, tp_done BOOLEAN, realized_sol DOUBLE, score DOUBLE,
    token_info JSON
);
CREATE TABLE IF NOT EXISTS equity (
    ts DOUBLE, run_id VARCHAR, mode VARCHAR, cash_sol DOUBLE, open_value_sol DOUBLE,
    equity_sol DOUBLE
);
CREATE TABLE IF NOT EXISTS creators (
    creator VARCHAR PRIMARY KEY, launches INTEGER, best_progress DOUBLE, rugs INTEGER
);
"""


def _frame(cursor: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """A result as a polars frame, without needing pyarrow."""
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    return pl.DataFrame(rows, schema=cols, orient="row", infer_schema_length=None)


_DECISION_COLS = [
    "ts",
    "run_id",
    "mint",
    "symbol",
    "score",
    "heuristic",
    "llm",
    "action",
    "reason",
    "features",
]


class Store:
    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = duckdb.connect(str(path))
        self.db.execute(_SCHEMA)
        # One connection, used from the event loop and the dashboard handlers.
        self._lock = threading.Lock()
        self._decisions: list[tuple] = []

    def _exec(self, sql: str, params: list | tuple | None = None) -> None:
        with self._lock:
            self.db.execute(sql, params or [])

    def decision(self, row: dict[str, Any]) -> None:
        """Buffered: decisions are many and cheap to lose; flushed in batches."""
        self._decisions.append(
            (
                row["ts"],
                row["run_id"],
                row["mint"],
                row["symbol"],
                row["score"],
                row.get("heuristic"),
                row.get("llm"),
                row["action"],
                row["reason"],
                json.dumps(row.get("features") or {}),
            )
        )
        if len(self._decisions) >= 2000:
            self.flush()

    def flush(self) -> None:
        if not self._decisions:
            return
        rows, self._decisions = self._decisions, []
        df = pl.DataFrame(
            rows, schema=_DECISION_COLS, orient="row", infer_schema_length=None
        )
        with self._lock:
            self.db.register("_new_decisions", df)
            self.db.execute("INSERT INTO decisions SELECT * FROM _new_decisions")
            self.db.unregister("_new_decisions")

    def trade(self, row: dict[str, Any]) -> None:
        self._exec(
            "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                row["ts"],
                row["run_id"],
                row["mode"],
                row["mint"],
                row["symbol"],
                row["side"],
                row["sol"],
                row["tokens_raw"],
                row["price"],
                row["fees_sol"],
                row.get("pnl_sol"),
                row["reason"],
                row.get("signature"),
            ],
        )

    def upsert_position(self, row: dict[str, Any]) -> None:
        self._exec(
            "INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                row["mint"],
                row["run_id"],
                row["mode"],
                row["symbol"],
                row["opened_at"],
                row["tokens_raw"],
                row["cost_sol"],
                row["entry_price"],
                row["peak_price"],
                row["tp_done"],
                row["realized_sol"],
                row["score"],
                json.dumps(row.get("token_info")) if row.get("token_info") else None,
            ],
        )

    def remove_position(self, mint: str) -> None:
        self._exec("DELETE FROM positions WHERE mint = ?", [mint])

    def open_positions(self, mode: str) -> list[dict[str, Any]]:
        with self._lock:
            df = _frame(
                self.db.execute("SELECT * FROM positions WHERE mode = ?", [mode])
            )
        return df.to_dicts()

    def equity(self, row: dict[str, Any]) -> None:
        self._exec(
            "INSERT INTO equity VALUES (?,?,?,?,?,?)",
            [
                row["ts"],
                row["run_id"],
                row["mode"],
                row["cash_sol"],
                row["open_value_sol"],
                row["equity_sol"],
            ],
        )

    def save_creators(self, rows: list[tuple[str, int, float, int]]) -> None:
        if not rows:
            return
        df = pl.DataFrame(
            rows, schema=["creator", "launches", "best_progress", "rugs"], orient="row"
        )
        with self._lock:
            self.db.register("_new_creators", df)
            self.db.execute(
                "INSERT OR REPLACE INTO creators SELECT * FROM _new_creators"
            )
            self.db.unregister("_new_creators")

    def load_creators(self) -> list[tuple[str, int, float, int]]:
        with self._lock:
            return self.db.execute("SELECT * FROM creators").fetchall()

    def frame(self, sql: str, params: list | None = None) -> pl.DataFrame:
        self.flush()
        with self._lock:
            return _frame(self.db.execute(sql, params or []))

    def close(self) -> None:
        self.flush()
        with self._lock:
            self.db.close()
