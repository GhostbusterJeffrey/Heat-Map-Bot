from __future__ import annotations

import io
import sqlite3
import threading
import uuid
import zlib
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class SnapshotRun:
    run_id: str
    ticker: str
    provider: str
    captured_at: datetime


@dataclass(frozen=True, slots=True)
class SnapshotStats:
    runs: int
    chains: int
    contracts: int
    oldest: datetime | None
    newest: datetime | None


@dataclass(frozen=True, slots=True)
class OIContractChange:
    expiration: date
    side: str
    strike: float
    contract_symbol: str | None
    old_open_interest: int
    new_open_interest: int
    change: int


@dataclass(frozen=True, slots=True)
class OIComparison:
    ticker: str
    expiration: date
    older_at: datetime
    newer_at: datetime
    call_oi_change: int
    put_oi_change: int
    total_oi_change: int
    contracts_compared: int
    top_increases: tuple[OIContractChange, ...]
    top_decreases: tuple[OIContractChange, ...]


class OptionSnapshotStore:
    """Thread-safe SQLite store for raw public option-chain observations.

    Each provider session is a snapshot run. Expiration chains are stored as
    zlib-compressed pandas table JSON so Alpha 4 can begin accumulating OI and
    quote history without adding a heavyweight database dependency.
    """

    def __init__(self, path: Path, *, retention_days: int = 730) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = max(int(retention_days), 1)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS snapshot_runs (
                    run_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    captured_at_utc TEXT NOT NULL,
                    source_as_of_utc TEXT,
                    spot REAL,
                    trailing_dividend_yield REAL,
                    model_version TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS option_chains (
                    run_id TEXT NOT NULL,
                    expiration TEXT NOT NULL,
                    captured_at_utc TEXT NOT NULL,
                    call_count INTEGER NOT NULL,
                    put_count INTEGER NOT NULL,
                    calls_zlib BLOB NOT NULL,
                    puts_zlib BLOB NOT NULL,
                    PRIMARY KEY (run_id, expiration),
                    FOREIGN KEY (run_id) REFERENCES snapshot_runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_snapshot_runs_ticker_time
                    ON snapshot_runs(ticker, captured_at_utc);
                CREATE INDEX IF NOT EXISTS idx_option_chains_expiration
                    ON option_chains(expiration);
                """
            )

    def begin_run(
        self,
        *,
        ticker: str,
        provider: str,
        model_version: str,
        captured_at: datetime | None = None,
    ) -> SnapshotRun:
        observed = _utc(captured_at or datetime.now(UTC))
        run = SnapshotRun(
            run_id=uuid.uuid4().hex,
            ticker=ticker.upper().strip(),
            provider=provider,
            captured_at=observed,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO snapshot_runs (
                    run_id, ticker, provider, captured_at_utc, model_version
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.ticker,
                    run.provider,
                    observed.isoformat(),
                    model_version,
                ),
            )
        return run

    def update_underlying(
        self,
        run_id: str,
        *,
        source_as_of: datetime,
        spot: float,
        trailing_dividend_yield: float,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE snapshot_runs
                SET source_as_of_utc = ?, spot = ?, trailing_dividend_yield = ?
                WHERE run_id = ?
                """,
                (
                    _utc(source_as_of).isoformat(),
                    float(spot),
                    float(trailing_dividend_yield),
                    run_id,
                ),
            )

    def record_chain(
        self,
        run_id: str,
        *,
        expiration: date,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        captured_at: datetime | None = None,
    ) -> None:
        observed = _utc(captured_at or datetime.now(UTC))
        calls_blob = _compress_frame(calls)
        puts_blob = _compress_frame(puts)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO option_chains (
                    run_id, expiration, captured_at_utc, call_count, put_count,
                    calls_zlib, puts_zlib
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    expiration.isoformat(),
                    observed.isoformat(),
                    int(len(calls)),
                    int(len(puts)),
                    calls_blob,
                    puts_blob,
                ),
            )

    def load_chain(self, run_id: str, expiration: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT calls_zlib, puts_zlib FROM option_chains
                WHERE run_id = ? AND expiration = ?
                """,
                (run_id, expiration.isoformat()),
            ).fetchone()
        if row is None:
            raise KeyError(f"No saved chain for {run_id} {expiration.isoformat()}.")
        return _decompress_frame(row[0]), _decompress_frame(row[1])

    def prune(self, *, now: datetime | None = None) -> int:
        cutoff = _utc(now or datetime.now(UTC)) - timedelta(days=self.retention_days)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM snapshot_runs WHERE captured_at_utc < ?",
                (cutoff.isoformat(),),
            )
            return max(int(cursor.rowcount), 0)

    def compare_latest(
        self,
        ticker: str,
        *,
        expiration: date | None = None,
        top_n: int = 5,
    ) -> OIComparison:
        symbol = ticker.upper().strip()
        with self._lock, self._connect() as connection:
            parameters: list[object] = [symbol]
            expiry_filter = ""
            if expiration is not None:
                expiry_filter = " AND c.expiration = ?"
                parameters.append(expiration.isoformat())
            runs = connection.execute(
                f"""
                SELECT r.run_id, r.captured_at_utc
                FROM snapshot_runs r
                JOIN option_chains c ON c.run_id = r.run_id
                WHERE r.ticker = ? {expiry_filter}
                GROUP BY r.run_id, r.captured_at_utc
                ORDER BY r.captured_at_utc DESC
                LIMIT 30
                """,
                parameters,
            ).fetchall()
            if len(runs) < 2:
                raise ValueError(
                    f"At least two saved {symbol} chain snapshots are required before OI changes can be measured."
                )

            selected = None
            selected_expiration = expiration
            for newer_index in range(len(runs) - 1):
                newer_id, newer_time = runs[newer_index]
                for older_index in range(newer_index + 1, len(runs)):
                    older_id, older_time = runs[older_index]
                    if selected_expiration is None:
                        newer_expiries = {
                            row[0]
                            for row in connection.execute(
                                "SELECT expiration FROM option_chains WHERE run_id = ?",
                                (newer_id,),
                            ).fetchall()
                        }
                        older_expiries = {
                            row[0]
                            for row in connection.execute(
                                "SELECT expiration FROM option_chains WHERE run_id = ?",
                                (older_id,),
                            ).fetchall()
                        }
                        common = sorted(newer_expiries & older_expiries)
                        if not common:
                            continue
                        chosen = date.fromisoformat(common[0])
                    else:
                        chosen = selected_expiration
                    selected = (newer_id, newer_time, older_id, older_time, chosen)
                    break
                if selected is not None:
                    break

        if selected is None:
            raise ValueError(f"No two {symbol} snapshots share a comparable expiration.")
        newer_id, newer_time, older_id, older_time, chosen = selected
        newer_calls, newer_puts = self.load_chain(newer_id, chosen)
        older_calls, older_puts = self.load_chain(older_id, chosen)
        changes = _compare_oi_frames(
            chosen,
            older_calls,
            older_puts,
            newer_calls,
            newer_puts,
        )
        if not changes:
            raise ValueError("The two snapshots contained no comparable option contracts.")
        call_change = sum(item.change for item in changes if item.side == "call")
        put_change = sum(item.change for item in changes if item.side == "put")
        increases = tuple(sorted((item for item in changes if item.change > 0), key=lambda item: item.change, reverse=True)[: max(top_n, 1)])
        decreases = tuple(sorted((item for item in changes if item.change < 0), key=lambda item: item.change)[: max(top_n, 1)])
        return OIComparison(
            ticker=symbol,
            expiration=chosen,
            older_at=_parse_datetime(older_time),
            newer_at=_parse_datetime(newer_time),
            call_oi_change=int(call_change),
            put_oi_change=int(put_change),
            total_oi_change=int(call_change + put_change),
            contracts_compared=len(changes),
            top_increases=increases,
            top_decreases=decreases,
        )

    def stats(self) -> SnapshotStats:
        with self._lock, self._connect() as connection:
            runs_row = connection.execute(
                "SELECT COUNT(*), MIN(captured_at_utc), MAX(captured_at_utc) FROM snapshot_runs"
            ).fetchone()
            chains_row = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(call_count + put_count), 0) FROM option_chains"
            ).fetchone()
        runs = int(runs_row[0] if runs_row else 0)
        oldest = _parse_datetime(runs_row[1]) if runs_row and runs_row[1] else None
        newest = _parse_datetime(runs_row[2]) if runs_row and runs_row[2] else None
        return SnapshotStats(
            runs=runs,
            chains=int(chains_row[0] if chains_row else 0),
            contracts=int(chains_row[1] if chains_row else 0),
            oldest=oldest,
            newest=newest,
        )


def _compress_frame(frame: pd.DataFrame) -> bytes:
    safe = frame.copy() if frame is not None else pd.DataFrame()
    payload = safe.to_json(orient="table", date_format="iso", index=False)
    return zlib.compress(payload.encode("utf-8"), level=9)


def _decompress_frame(blob: bytes) -> pd.DataFrame:
    payload = zlib.decompress(blob).decode("utf-8")
    return pd.read_json(io.StringIO(payload), orient="table")


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return _utc(parsed)


def _compare_oi_frames(
    expiration: date,
    older_calls: pd.DataFrame,
    older_puts: pd.DataFrame,
    newer_calls: pd.DataFrame,
    newer_puts: pd.DataFrame,
) -> list[OIContractChange]:
    old = pd.concat(
        [_oi_frame(older_calls, "call"), _oi_frame(older_puts, "put")],
        ignore_index=True,
    )
    new = pd.concat(
        [_oi_frame(newer_calls, "call"), _oi_frame(newer_puts, "put")],
        ignore_index=True,
    )
    merged = old.merge(new, on=["key", "side", "strike"], how="outer", suffixes=("_old", "_new"))
    merged["open_interest_old"] = pd.to_numeric(merged["open_interest_old"], errors="coerce").fillna(0).astype(int)
    merged["open_interest_new"] = pd.to_numeric(merged["open_interest_new"], errors="coerce").fillna(0).astype(int)
    changes: list[OIContractChange] = []
    for _, row in merged.iterrows():
        old_oi = int(row["open_interest_old"])
        new_oi = int(row["open_interest_new"])
        symbol = row.get("contract_symbol_new")
        if pd.isna(symbol) or not symbol:
            symbol = row.get("contract_symbol_old")
        changes.append(
            OIContractChange(
                expiration=expiration,
                side=str(row["side"]),
                strike=float(row["strike"]),
                contract_symbol=None if pd.isna(symbol) else str(symbol),
                old_open_interest=old_oi,
                new_open_interest=new_oi,
                change=new_oi - old_oi,
            )
        )
    return changes


def _oi_frame(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    source = frame.copy() if frame is not None else pd.DataFrame()
    if "strike" not in source.columns:
        return pd.DataFrame(columns=["key", "side", "strike", "contract_symbol", "open_interest"])
    strike = pd.to_numeric(source["strike"], errors="coerce")
    oi_source = source["openInterest"] if "openInterest" in source.columns else pd.Series(0, index=source.index)
    oi = pd.to_numeric(oi_source, errors="coerce").fillna(0).clip(lower=0).astype(int)
    symbols = source["contractSymbol"] if "contractSymbol" in source.columns else pd.Series(None, index=source.index)
    result = pd.DataFrame({
        "side": side,
        "strike": strike,
        "contract_symbol": symbols,
        "open_interest": oi,
    }).dropna(subset=["strike"])
    result["key"] = result["contract_symbol"].where(
        result["contract_symbol"].notna() & (result["contract_symbol"].astype(str) != ""),
        result["side"] + ":" + result["strike"].map(lambda value: f"{float(value):.8f}"),
    )
    return result[["key", "side", "strike", "contract_symbol", "open_interest"]]
