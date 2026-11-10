"""Observed liquidations: recorder events (binance/bybit/okx) + Gate 5m aggregates as bars."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EVENT_VENUES = ("binance", "bybit", "okx")
ALL_VENUES = EVENT_VENUES + ("gate",)
MAX_EVENTS = 5000
MAX_SPAN = pd.Timedelta("31D")


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def observed(symbol: str, start, end, venues=ALL_VENUES, liq_root: Path | None = None,
             gate_root: Path | None = None) -> dict:
    from src.data import gate, liq_recorder
    end, start = _utc(end), _utc(start)
    if start >= end:
        raise ValueError("start deve ser anterior a end")
    if end - start > MAX_SPAN:
        raise ValueError("intervalo maximo: 31 dias")
    venues = tuple(v for v in venues if v in ALL_VENUES) or ALL_VENUES
    events, bars, notes = [], [], []
    for v in venues:
        if v == "gate":
            d = gate.load(symbol, gate_root) if gate_root else gate.load(symbol)
            if d is None:
                notes.append("gate: sem dados")
                continue
            d = d[(d.index >= start) & (d.index <= end)]
            d = d[(d["long_liq_usd"] > 0) | (d["short_liq_usd"] > 0)]
            bars += [{"t": int(t.timestamp()), "step": 300, "venue": "gate",
                      "long_usd": round(float(r.long_liq_usd), 2), "short_usd": round(float(r.short_liq_usd), 2)}
                     for t, r in zip(d.index, d.itertuples())]
        else:
            df = liq_recorder.read_events(v, start, end, liq_root or liq_recorder.OUT)
            df = df[df["symbol"] == symbol]
            events += [{"t": int(r.ts.timestamp() * 1000) / 1000, "venue": v, "side": r.side,
                        "price": float(r.price), "qty": float(r.qty), "usd": round(float(r.usd), 2)}
                       for r in df.itertuples()]
    if len(events) > MAX_EVENTS:
        notes.append(f"events truncated to the {MAX_EVENTS} largest")
        events = sorted(sorted(events, key=lambda e: -e["usd"])[:MAX_EVENTS], key=lambda e: e["t"])
    else:
        events.sort(key=lambda e: e["t"])
    return {"symbol": symbol, "start": int(start.timestamp()), "end": int(end.timestamp()),
            "venues": list(venues), "side_meaning": "side = posicao liquidada (long|short)",
            "events": events, "bars": bars, "notes": notes}
