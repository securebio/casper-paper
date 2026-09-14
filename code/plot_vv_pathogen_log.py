#!/usr/bin/env python3
"""
Supplementary figure: Figure 4c pathogen dynamics on a log abundance scale.

Figure 4c plots fold change from each site's median, which makes trends
comparable across sites but hides the abundance level they sit at. This figure
plots the same grid, sites and smoothing against Tobamovirus-normalized
abundance directly, so absolute levels -- notably the gap between enteric and
respiratory targets -- are visible.

The symlog threshold is set at 1e-8, near the smallest non-zero normalized
abundance observed (about 1e-8 for the respiratory targets). Setting it higher
folds the bulk of the respiratory data into the linear region and flattens
those panels: the median non-zero abundance is 7e-7 to 2e-6 for most
respiratory targets, so a 1e-5 threshold puts the typical value below the log
region entirely.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from plot_pathogen_grid import (
    prepare_pathogen_data as _prepare_pathogen_data,
    render_pathogen_grid,
    N_ROWS,
)
from plot_vv_pathogen_combined import PATHOGENS, DEFAULT_HIGHLIGHT_SITES

# Tobamovirus-genus-normalized clade abundance, the quantity used for the
# time-series panels in Figure 4c and Figure 5.
ABUNDANCE_COL = 'ra_clade_tobamovirus_norm'

# symlog linear region, set to the detection quantum: the median abundance
# contributed by a single read pair is 1.17e-7 (p5 2.1e-8, p25 5.4e-8, measured
# as ra_clade_tobamovirus_norm / clade_counts across all sample-taxon pairs).
# No sample can take a non-zero value below its own quantum, so extending the
# log region further down only buys empty axis -- and the smoothed traces do
# reach it, down to 1.1e-9, because a 5-week log1p mean of mostly-zero weeks
# interpolates below the quantum. Those sub-quantum values are smoothing
# artefacts rather than measurements, and giving them decades of log space is
# what stretched trend lines into near-vertical drops toward zero.
LINTHRESH = 1e-7

# Lowest decade to carry a tick, matching LINTHRESH.
FLOOR_EXP = -7


def _format_abundance_tick(value):
    """Render a decade tick as a power of ten, with 0 kept literal."""
    if value == 0:
        return '0'
    return f'$10^{{{int(round(math.log10(value)))}}}$'


def _abundance_tick_labels(ticks, row):
    """Label every decade, except on the enteric row where they do not fit.

    That row spans the full abundance range, so it labels every other position
    while each decade keeps its tick mark.
    """
    if row == N_ROWS - 1:
        return [_format_abundance_tick(t) if i % 2 == 0 else ''
                for i, t in enumerate(ticks)]
    return [_format_abundance_tick(t) for t in ticks]


def print_abundance_statistics(df, highlight_sites=DEFAULT_HIGHLIGHT_SITES):
    """Report per-pathogen normalized-abundance ranges for the response text."""
    print("\n" + "=" * 78)
    print("Tobamovirus-normalized abundance per sample, by pathogen")
    print("=" * 78)
    print(f"{'Pathogen':<18} {'% zero':>7} {'min>0':>10} {'median>0':>10} "
          f"{'max':>10}")
    print("-" * 78)
    for taxid, label in PATHOGENS:
        values = df[df['taxid'] == taxid][ABUNDANCE_COL].dropna()
        if values.empty:
            print(f"{label:<18} {'no data':>7}")
            continue
        nonzero = values[values > 0]
        print(f"{label:<18} {100 * (values == 0).mean():6.1f}% "
              f"{nonzero.min():10.2e} {nonzero.median():10.2e} "
              f"{values.max():10.2e}")
    print("=" * 78 + "\n")


def plot_vv_pathogen_log(save_path=None, highlight_sites=None):
    """Draw the 4x4 pathogen grid against log Tobamovirus-normalized abundance.

    Returns:
        fig: Matplotlib figure
    """
    return render_pathogen_grid(
        value_col=ABUNDANCE_COL,
        linthresh=LINTHRESH,
        ylabel='Wastewater sequencing signal\n(pathogen / tobamovirus read pairs)',
        tick_labeller=_abundance_tick_labels,
        y_floor_exp=FLOOR_EXP,
        ylabel_offset=0.09,
        # Respiratory rows span only ~4 decades; halving the linear-region
        # scale keeps the 0-to-1e-8 band from taking a quarter of the axis.
        linscale=0.5,
        save_path=save_path,
        highlight_sites=highlight_sites,
    )


if __name__ == '__main__':
    print_abundance_statistics(_prepare_pathogen_data())
    plot_vv_pathogen_log(save_path='../figures/vv_pathogen_log.png')
    print("Saved figures/vv_pathogen_log.png")
