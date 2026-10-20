"""MMWR-week aggregation and smoothing, and Spearman correlation on the
smoothed series.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# Weeks of overlap a correlation needs before it is reported. The
# `min_overlap` argument to `spearman_on_smoothed` is the lower floor at
# which the rank correlation can be computed at all.
MIN_OVERLAP_WEEKS = 10

# Minimum non-zero weeks required on each side of a paired comparison.
MIN_NONZERO_WEEKS = 2

# Source -> smoothing kernel. Count-like clinical series use an arithmetic
# mean; relative abundance and PCR concentration use a log1p-shifted mean.
SOURCE_KERNELS = {
    "pcr": "log1p",
    "mgs": "log1p",
    "clinical": "arithmetic",
}


# =============================================================================
# Public API
# =============================================================================

def calculate_mmwr_smoothed_trend(df, date_col, value_col, smooth_window=5,
                                   kernel="log1p"):
    """Aggregate to MMWR weeks and apply a centered moving average.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date_col`` and ``value_col``.
    date_col : str
        Column name holding sample dates (anything ``pd.to_datetime``
        can parse).
    value_col : str
        Column name holding the per-sample value to smooth.
    smooth_window : int, default 5
        Window width in MMWR weeks for the centered moving average.
        ``min_periods=1`` allows partial windows at the series edges so
        weeks near the end of one source's data are not dropped when
        correlated against a longer comparison source.
    kernel : {"log1p", "arithmetic"}, default "log1p"
        Aggregation operator used for both within-week and multi-week
        averaging. ``"log1p"`` uses ``expm1(mean(log1p(x)))``, a
        log1p-shifted mean appropriate for relative-abundance and
        concentration data with multiplicative noise. ``"arithmetic"``
        uses a plain ``mean(x)`` and is appropriate for clinical count
        and rate data.

    Returns
    -------
    pd.DataFrame
        Columns: ``date_col`` (MMWR-week midpoint Wednesday),
        ``smoothed_values``, ``mmwr_year``, ``mmwr_week``. Empty if the
        input has fewer than two non-null rows.
    """
    import epiweeks

    df_clean = df[[date_col, value_col]].copy().dropna().sort_values(date_col)
    if len(df_clean) < 2:
        return pd.DataFrame()

    df_clean["date"] = pd.to_datetime(df_clean[date_col])

    mmwr_rows = []
    for _, row in df_clean.iterrows():
        ew = epiweeks.Week.fromdate(row["date"])
        mmwr_rows.append({
            "mmwr_year": ew.year, "mmwr_week": ew.week,
            "value": row[value_col],
        })
    mmwr_df = pd.DataFrame(mmwr_rows)
    if kernel == "log1p":
        mmwr_df["agg_value"] = np.log1p(mmwr_df["value"])
    elif kernel == "arithmetic":
        mmwr_df["agg_value"] = mmwr_df["value"].astype(float)
    else:
        raise ValueError(
            f"kernel must be 'log1p' or 'arithmetic', got {kernel!r}")

    weekly = mmwr_df.groupby(["mmwr_year", "mmwr_week"]).agg(
        agg_value=("agg_value", "mean")
    ).reset_index().sort_values(["mmwr_year", "mmwr_week"])

    # Reindex onto a contiguous MMWR-week grid so the centered rolling
    # window spans calendar weeks rather than present-row indices.
    start = epiweeks.Week(int(weekly["mmwr_year"].iloc[0]),
                          int(weekly["mmwr_week"].iloc[0]))
    end = epiweeks.Week(int(weekly["mmwr_year"].iloc[-1]),
                        int(weekly["mmwr_week"].iloc[-1]))
    grid_rows = []
    w = start
    while w <= end:
        grid_rows.append({"mmwr_year": w.year, "mmwr_week": w.week})
        w += 1
    grid = pd.DataFrame(grid_rows)
    weekly = grid.merge(weekly, on=["mmwr_year", "mmwr_week"], how="left")
    present = weekly["agg_value"].notna()

    weekly["date"] = weekly.apply(
        lambda r: epiweeks.Week(int(r["mmwr_year"]), int(r["mmwr_week"])).startdate()
        + pd.Timedelta(days=3),
        axis=1,
    )
    rolled = weekly["agg_value"].rolling(
        window=smooth_window, center=True, min_periods=1).mean()
    weekly["smoothed_values"] = np.expm1(rolled) if kernel == "log1p" else rolled

    # Drop the synthetic grid weeks so downstream joins only see weeks
    # that actually had observations.
    weekly = weekly[present].reset_index(drop=True)

    return pd.DataFrame({
        date_col: weekly["date"],
        "smoothed_values": weekly["smoothed_values"],
        "mmwr_year": weekly["mmwr_year"],
        "mmwr_week": weekly["mmwr_week"],
    })


def smooth_value_series(df, date_col, value_col, kernel="log1p"):
    """Filter to non-negative non-null values and apply MMWR smoothing.

    Zeros are retained — they are real measurements. NaN and negative
    values are dropped. ``kernel`` is forwarded to
    :func:`calculate_mmwr_smoothed_trend`.

    Returns the smoothed DataFrame, or ``None`` if no rows survive the
    filter or the smoother returns empty.
    """
    valid = df[df[value_col].notna() & (df[value_col] >= 0)].copy()
    if valid.empty:
        return None
    s = calculate_mmwr_smoothed_trend(valid, date_col, value_col,
                                      kernel=kernel)
    return s if not s.empty else None


def smooth_for_source(df, date_col, value_col, source):
    """Smooth a per-sample series with the kernel appropriate to ``source``.

    Thin wrapper around :func:`smooth_value_series` that looks up the
    correct kernel for the series type from :data:`SOURCE_KERNELS`. Use
    this everywhere a clinical or wastewater series is smoothed so the
    kernel choice is single-sourced.

    ``source="clinical"`` covers both counts (positives, tests) and the
    positivity rate. A rate is not Poisson, but it is a bounded linear
    quantity on a 0-100 scale rather than a multiplicative one, so the
    arithmetic kernel is the deliberate choice there too.
    """
    kernel = SOURCE_KERNELS.get(source, "log1p")
    return smooth_value_series(df, date_col, value_col, kernel=kernel)


def has_nonzero_floor(s, min_nonzero=MIN_NONZERO_WEEKS):
    """True if a smoothed series has at least ``min_nonzero`` positive
    MMWR-week values."""
    if s is None or s.empty:
        return False
    return int((s["smoothed_values"] > 0).sum()) >= min_nonzero


def passes_inclusion_floors(s1, s2, min_overlap=MIN_OVERLAP_WEEKS,
                              min_nonzero=MIN_NONZERO_WEEKS):
    """True if a paired (s1, s2) smoothed pair clears the reporting
    criterion: at least ``min_overlap`` overlapping MMWR weeks AND at
    least ``min_nonzero`` positive weeks on each side."""
    if s1 is None or s2 is None:
        return False
    if not (has_nonzero_floor(s1, min_nonzero)
            and has_nonzero_floor(s2, min_nonzero)):
        return False
    merged = pd.merge(
        s1[["mmwr_year", "mmwr_week"]],
        s2[["mmwr_year", "mmwr_week"]],
        on=["mmwr_year", "mmwr_week"], how="inner")
    return len(merged) >= min_overlap


def spearman_on_smoothed(s1, s2, min_overlap=3):
    """Spearman R on two MMWR-smoothed series, joined on (year, week).

    Parameters
    ----------
    s1, s2 : pd.DataFrame or None
        Smoothed series as returned by
        :func:`calculate_mmwr_smoothed_trend`. ``None`` is treated as
        "no data".
    min_overlap : int, default 3
        Minimum overlapping weeks required to compute Spearman. This is
        the *computation* floor for the early return; the *reporting*
        threshold is the module-level constant ``MIN_OVERLAP_WEEKS``,
        which consumers pass in explicitly when they want paper-grade
        results.

    Returns
    -------
    (spearman_r, p_value, n_overlap_weeks)
        ``(NaN, NaN, 0)`` if either input is ``None``;
        ``(NaN, NaN, n_overlap)`` if the overlap is below ``min_overlap``.
    """
    if s1 is None or s2 is None:
        return np.nan, np.nan, 0
    merged = pd.merge(
        s1[["mmwr_year", "mmwr_week", "smoothed_values"]].rename(
            columns={"smoothed_values": "v1"}),
        s2[["mmwr_year", "mmwr_week", "smoothed_values"]].rename(
            columns={"smoothed_values": "v2"}),
        on=["mmwr_year", "mmwr_week"], how="inner",
    )
    n = len(merged)
    if n < min_overlap:
        return np.nan, np.nan, n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = spearmanr(merged["v1"], merged["v2"])
    return r, p, n
