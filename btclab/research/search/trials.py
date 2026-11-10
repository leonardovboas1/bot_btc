"""DuckDB trial ledger. Every evaluated config is one row (honest N), deduped by spec hash."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = ROOT / "research" / "trials.duckdb"

_HASH_KEYS = ("tf", "indicators", "long", "short", "exit_long", "exit_short", "risk", "costs")

COLUMNS = ("stage", "tf", "spec_json", "spec_hash", "parent_hash", "family", "tag", "fold_json",
           "median_oos_sharpe", "mean_is_sharpe", "n_pos_folds", "is_trades", "min_oos_trades",
           "oos_sharpe", "max_dd", "profit_factor", "oos_returns")
LIGHT = tuple(c for c in COLUMNS if c != "oos_returns")


def spec_hash(spec: dict) -> str:
    """Hash of the economically relevant spec content (the name is excluded)."""
    d = {k: spec.get(k) for k in _HASH_KEYS}
    d["costs"] = d["costs"] or "BINANCE_PERP_TAKER"
    d["risk"] = d["risk"] or {}
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


class TrialStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else DEFAULT_DB
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path))
        self.con.execute("CREATE SEQUENCE IF NOT EXISTS trials_seq START 1")
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS trials(
                id BIGINT DEFAULT nextval('trials_seq'),
                stage VARCHAR, tf VARCHAR, spec_json VARCHAR, spec_hash VARCHAR UNIQUE,
                parent_hash VARCHAR, family VARCHAR, tag VARCHAR, fold_json VARCHAR,
                median_oos_sharpe DOUBLE, mean_is_sharpe DOUBLE, n_pos_folds INTEGER,
                is_trades INTEGER, min_oos_trades INTEGER, oos_sharpe DOUBLE, max_dd DOUBLE,
                profit_factor DOUBLE, oos_returns BLOB,
                survivor BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT current_timestamp)""")

    def close(self):
        self.con.close()

    def count(self, stage: Optional[str] = None, tf: Optional[str] = None) -> int:
        q, a = "SELECT count(*) FROM trials WHERE 1=1", []
        if stage:
            q += " AND stage=?"; a.append(stage)
        if tf:
            q += " AND tf=?"; a.append(tf)
        return int(self.con.execute(q, a).fetchone()[0])

    def existing_hashes(self) -> set:
        return {r[0] for r in self.con.execute("SELECT spec_hash FROM trials").fetchall()}

    def insert_many(self, records: Iterable[dict]) -> int:
        n = 0
        for r in records:
            vals = [r.get(c) for c in COLUMNS]
            self.con.execute(
                f"INSERT INTO trials({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))}) "
                "ON CONFLICT (spec_hash) DO NOTHING", vals)
            n += 1
        return n

    def fetch(self, stage: Optional[str] = None, tf: Optional[str] = None, *, survivors: bool = False,
              family: Optional[str] = None, blobs: bool = False):
        cols = COLUMNS if blobs else LIGHT
        q = f"SELECT {','.join(cols)}, survivor FROM trials WHERE 1=1"
        a = []
        if stage:
            q += " AND stage=?"; a.append(stage)
        if tf:
            q += " AND tf=?"; a.append(tf)
        if family:
            q += " AND family=?"; a.append(family)
        if survivors:
            q += " AND survivor"
        q += " ORDER BY id"
        df = self.con.execute(q, a).fetchdf()
        return df

    def set_survivors(self, stage: str, tf: str, hashes: Iterable[str]) -> None:
        self.con.execute("UPDATE trials SET survivor=FALSE WHERE stage=? AND tf=?", [stage, tf])
        hs = list(hashes)
        if hs:
            self.con.execute("UPDATE trials SET survivor=TRUE WHERE spec_hash IN (SELECT unnest(?))", [hs])

    def returns_matrix(self, hashes: Optional[list] = None, tf: Optional[str] = None):
        """(T x N) float matrix of OOS daily returns (columns in ``hashes`` order) and the hash list."""
        if hashes is None:
            q, a = "SELECT spec_hash, oos_returns FROM trials WHERE oos_returns IS NOT NULL", []
            if tf:
                q += " AND tf=?"; a.append(tf)
            rows = self.con.execute(q + " ORDER BY id", a).fetchall()
        else:
            got = dict(self.con.execute(
                "SELECT spec_hash, oos_returns FROM trials WHERE spec_hash IN (SELECT unnest(?))", [hashes]).fetchall())
            rows = [(h, got[h]) for h in hashes if h in got]
        hs = [r[0] for r in rows]
        if not rows:
            return np.empty((0, 0)), hs
        cols = [np.frombuffer(r[1], dtype=np.float32).astype(np.float64) for r in rows]
        return np.column_stack(cols), hs
