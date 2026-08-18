#!/usr/bin/env python3
"""
Supplementary figure: absolute read counts behind the Figure 4c pathogen panel.

Figure 4c plots fold change from the site median of Tobamovirus-normalized
abundance, and the log-scale supplement plots Tobamovirus-normalized abundance.
Neither shows how many reads actually underlie each point. This figure plots the
same 4x4 pathogen grid, same sites and same smoothing, against unnormalized
`clade_counts` -- the read pairs assigned to each pathogen clade by the VV
pipeline -- so the detection depth behind every trend is visible directly.

The y-axis is symlog with a linear region below 1 read, so zero-count samples
are drawn on the axis floor rather than dropped. That distinction matters here:
across the highlighted sites, several respiratory targets are zero in the
majority of samples, and the count scale is what makes those non-detections
legible as non-detections.

Counts are not depth-corrected. Sequencing depth varies about 1.4x across the
interquartile range of samples (p25 0.82e9, p75 1.17e9 read pairs), so counts
are broadly but not exactly comparable between samples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from plot_pathogen_grid import (
    prepare_pathogen_data as _prepare_pathogen_data,
    render_pathogen_grid,
    row_top_from_data_max,
    N_ROWS,
)
from plot_vv_pathogen_combined import PATHOGENS, DEFAULT_HIGHLIGHT_SITES

# Column holding read pairs assigned to the pathogen clade by the VV pipeline.
# Full-depth counts, not the shallow-profiling subsample.
COUNT_COL = 'clade_counts'

# symlog linear region: below 1 read the scale is linear, so 0 sits on the
# floor and a 1-read sample stays visually distinct from a non-detection.
LINTHRESH = 1


def _format_count_tick(value):
    """Render a decade tick as a read count: 0, 1, 10, 100, 1k, 10k, 1M."""
    if value == 0:
        return '0'
    magnitude = abs(value)
    if magnitude < 1_000:
        return f'{value:.0f}'
    if magnitude < 1_000_000:
        return f'{value / 1_000:.0f}k'
    return f'{value / 1_000_000:.0f}M'


def _count_tick_labels(ticks, row):
    """Label every decade, except on the enteric row where they do not fit.

    That row spans seven decades, so it labels every other position (0, 10, 1k,
    100k, 10M) while every decade keeps its tick mark.
    """
    if row == N_ROWS - 1:
        return [_format_count_tick(t) if i % 2 == 0 else ''
                for i, t in enumerate(ticks)]
    return [_format_count_tick(t) for t in ticks]


def print_read_count_statistics(df, highlight_sites=DEFAULT_HIGHLIGHT_SITES):
    """Report per-pathogen count distributions for the Methods/response text."""
    print("\n" + "=" * 78)
    print("Absolute read counts per sample, by pathogen")
    print("=" * 78)
    print(f"{'Pathogen':<18} {'n':>6} {'% zero':>7} {'median':>9} "
          f"{'p95':>10} {'max':>10}")
    print("-" * 78)

    for taxid, label in PATHOGENS:
        counts = df[df['taxid'] == taxid][COUNT_COL].dropna()
        if counts.empty:
            print(f"{label:<18} {'no data':>6}")
            continue
        print(f"{label:<18} {len(counts):6d} {100 * (counts == 0).mean():6.1f}% "
              f"{counts.median():9.0f} {counts.quantile(0.95):10.0f} "
              f"{counts.max():10.0f}")

    print("-" * 78)
    print("Highlighted sites only:")
    hl = df[df['site_name'].isin(highlight_sites)]
    for taxid, label in PATHOGENS:
        counts = hl[hl['taxid'] == taxid][COUNT_COL].dropna()
        if counts.empty:
            continue
        print(f"  {label:<16} {100 * (counts == 0).mean():5.1f}% zero, "
              f"median {counts.median():.0f}, max {counts.max():.0f}")
    print("=" * 78 + "\n")


def plot_vv_pathogen_read_counts(save_path=None, highlight_sites=None):
    """Draw the 4x4 pathogen grid against unnormalized clade read counts.

    Returns:
        fig: Matplotlib figure
    """
    return render_pathogen_grid(
        value_col=COUNT_COL,
        linthresh=LINTHRESH,
        ylabel='Read pairs assigned to pathogen clade',
        tick_labeller=_count_tick_labels,
        y_floor_exp=0,
        # Absolute counts are the point of this figure, so the axis encloses
        # every drawn sample rather than clipping above the smoothed peak.
        row_top_rule=row_top_from_data_max,
        save_path=save_path,
        highlight_sites=highlight_sites,
    )


if __name__ == '__main__':
    print_read_count_statistics(_prepare_pathogen_data())
    plot_vv_pathogen_read_counts(save_path='../figures/vv_pathogen_read_counts.png')
    print("Saved figures/vv_pathogen_read_counts.png")
