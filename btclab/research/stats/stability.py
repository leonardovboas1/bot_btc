"""Parameter-neighbourhood stability score.

For every point of a parameter grid, the median score of the neighbours whose
every parameter lies within ±tol (30 % by default) of the point's own value,
expressed as a ratio to the point's own score. A robust optimum has a ratio
close to 1 (its neighbourhood performs about as well); an isolated spike has a
ratio well below 1. plan.md: neighbour median >= 70 % of the peak score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["parameter_stability", "stability_at_peak", "StabilityResult"]


def _neighbour_mask(values: np.ndarray, tol: float, abs_tol: float) -> np.ndarray:
    """(n x n) boolean: j is a neighbour of i on this parameter."""
    v = values[:, None]
    band = np.maximum(np.abs(v) * tol, abs_tol)
    return np.abs(values[None, :] - v) <= band


def parameter_stability(
    grid: pd.DataFrame,
    param_cols: Sequence[str],
    score_col: str,
    tol: float = 0.30,
    abs_tol: float = 1e-12,
    include_self: bool = False,
    min_neighbours: int = 1,
) -> pd.DataFrame:
    """Neighbourhood stability for every row of ``grid``.

    Returns a copy of ``grid`` with columns ``n_neighbours``,
    ``neighbour_median``, ``neighbour_min`` and ``stability_ratio``
    (= neighbour_median / own score; NaN when there are fewer than
    ``min_neighbours`` neighbours or the own score is <= 0).

    Parameters are compared on their numeric values (categorical params must be
    encoded first or excluded; identical values are always neighbours).
    """
    df = grid.reset_index(drop=True)
    n = len(df)
    score = df[score_col].to_numpy(dtype=float)
    mask = np.ones((n, n), dtype=bool)
    for col in param_cols:
        vals = df[col].to_numpy(dtype=float)
        mask &= _neighbour_mask(vals, tol, abs_tol)
    if not include_self:
        np.fill_diagonal(mask, False)
    mask &= np.isfinite(score)[None, :]

    n_nb = mask.sum(axis=1)
    med = np.full(n, np.nan)
    mn = np.full(n, np.nan)
    for i in range(n):
        if n_nb[i] >= max(1, min_neighbours):
            s = score[mask[i]]
            med[i] = np.median(s)
            mn[i] = s.min()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(score > 0, med / score, np.nan)

    out = df.copy()
    out["n_neighbours"] = n_nb
    out["neighbour_median"] = med
    out["neighbour_min"] = mn
    out["stability_ratio"] = ratio
    return out


@dataclass
class StabilityResult:
    peak_index: int
    peak_score: float
    peak_params: dict
    n_neighbours: int
    neighbour_median: float
    neighbour_min: float
    stability_ratio: float
    tol: float
    threshold: float
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def stability_at_peak(
    grid: pd.DataFrame,
    param_cols: Sequence[str],
    score_col: str,
    tol: float = 0.30,
    threshold: float = 0.70,
    peak_index: Optional[int] = None,
    **kwargs,
) -> StabilityResult:
    """Stability of the best point (or ``peak_index``): neighbour median / peak score."""
    table = parameter_stability(grid, param_cols, score_col, tol=tol, **kwargs)
    if peak_index is None:
        peak_index = int(table[score_col].idxmax())
    row = table.loc[peak_index]
    ratio = float(row["stability_ratio"])
    return StabilityResult(
        peak_index=int(peak_index),
        peak_score=float(row[score_col]),
        peak_params={c: row[c] for c in param_cols},
        n_neighbours=int(row["n_neighbours"]),
        neighbour_median=float(row["neighbour_median"]),
        neighbour_min=float(row["neighbour_min"]),
        stability_ratio=ratio,
        tol=float(tol),
        threshold=float(threshold),
        passed=bool(np.isfinite(ratio) and ratio >= threshold),
    )
