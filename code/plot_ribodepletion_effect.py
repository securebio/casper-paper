#!/usr/bin/env python3
"""
Supplementary figure: effect of the ribodepletion step on vertebrate-infecting
virus recovery.

A reagent preparation error omitted the RiboPOOLs depletion probes from SB
samples processed between 2025-05-13 and 2025-07-01 inclusive (Methods). MU
never ribodepletes. Boston DITP North and South were split between the two
laboratories over the same period, so for those two systems the MU series is a
control on the same wastewater: anything seasonal shows up in both arms, and
anything protocol-driven shows up in one.

Two quantities are drawn per sample, on twin axes:

    rrna_fraction = rRNA reads / reads passing QC        (left axis, linear)
    vv_fraction   = VV read pairs / total read pairs     (right axis, log)

The VV fraction comes from the dedicated VV pipeline over the whole library
(`calculate_vv_fraction_per_library`); the rRNA fraction comes from the shallow
Kraken profiling subsample. Both are fractions, so neither depends on how many
reads were profiled.

The figure exists because rRNA fraction alone is a poor marker of the omission.
At Boston the rRNA fraction barely moves across the window while the VV fraction
falls by more than an order of magnitude; only at Miami does rRNA show the
textbook rise.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))

from plot_config import (
    LAB_COLORS,
    filter_timeseries_data,
    format_date_axis,
    save_figure,
)
from data_loaders import (
    calculate_vv_fraction_per_library,
    load_all_kraken_data,
    load_all_site_data,
    load_all_vv_family_data,
)

# The omission window, from the Methods. Stated as processing dates; the
# collection dates in the release align with it exactly at all three SB sites,
# with sharp boundaries in both rRNA and VV fraction, so it is applied to the
# collection date here.
OMISSION_START = pd.Timestamp('2025-05-13')
OMISSION_END = pd.Timestamp('2025-07-01')

# Values of the repo-only `ribodepletion_status` metadata column. `none` covers
# MU, which does not ribodeplete at all; the two `ribopool` values distinguish
# SB samples that got the probes from those that did not.
STATUS_NONE = 'none'
STATUS_DEPLETED = 'ribopool'
STATUS_OMITTED = 'ribopool_omitted'

# Shading for the omission window in the SB cells.
OMITTED_BAND_COLOR = '#d62728'

# The three SB sites in this release. Boston has a paired MU arm; Miami does not.
SITES = [
    'Boston DITP North, MA',
    'Boston DITP South, MA',
    'Miami-Dade CDWWTP, FL',
]
# Drawn x-axis limits. The rows themselves are restricted per site to the span
# in which every laboratory at that site was running (`comparison_spans`); these
# are only the plotted limits, with a little padding before the first sample.
DATE_MIN = pd.Timestamp('2025-02-01')
DATE_MAX = pd.Timestamp('2026-01-05')

# Cell layout: (row, col) -> (site, lab), with (2, 0) left for the scatter.
CELLS = {
    (0, 0): ('Boston DITP North, MA', 'MU'),
    (0, 1): ('Boston DITP North, MA', 'SB'),
    (1, 0): ('Boston DITP South, MA', 'MU'),
    (1, 1): ('Boston DITP South, MA', 'SB'),
    (2, 1): ('Miami-Dade CDWWTP, FL', 'SB'),
}
# (2, 0) has no MU arm at Miami and is left blank; it carries the shared legend.
LEGEND_CELL = (2, 0)

# Cells that carry x tick labels: the lowest time-series cell in each column.
X_LABELLED_CELLS = {(1, 0), (2, 1)}

RRNA_COLOR = '#777777'

# Laboratory is encoded by column, so the VV legend key is drawn neutral rather
# than in either lab colour.
SERIES_KEY_COLOR = '#444444'

LAB_ORDER = ['MU', 'SB']

# Grey column header bars, in figure coordinates.
HEADER_BAR_COLOR = '#d9d9d9'
HEADER_BAR_Y = 0.952
HEADER_BAR_HEIGHT = 0.024

TITLE_FONT = 13
LABEL_FONT = 11
TICK_FONT = 10
ANNOT_FONT = 10

FIG_SIZE = (13, 11)

OUTPUT_PATH = '../figures/ribodepletion_effect.png'


def ribodepletion_status(lab, date):
    """Ribodepletion status of one sample, from laboratory and collection date.

    The single definition of the rule. The metadata and table scripts import
    this rather than re-deriving the window.
    """
    if lab != 'SB':
        return STATUS_NONE
    date = pd.Timestamp(date)
    if OMISSION_START <= date <= OMISSION_END:
        return STATUS_OMITTED
    return STATUS_DEPLETED


def comparison_spans(df):
    """Per-site first date on which every laboratory at that site was running.

    Boston was processed in parallel from 2025-02-12, but MU had been running
    there since 2024-03-20. Restricting both arms to the shared span is what
    makes MU a control rather than a different season: the out-of-window
    medians are then taken over the same calendar months. Miami is SB-only, so
    its span simply starts at its own first sample.

    Returns:
        dict of site_name -> pd.Timestamp.
    """
    return {
        site: group.groupby('sequencing_lab')['date'].min().max()
        for site, group in df.groupby('site_name')
    }


def prepare_ribodepletion_data(kraken_data=None, site_data=None,
                               vv_family_data=None):
    """Per-sample rRNA and VV fraction for the three SB sites.

    Returns:
        pd.DataFrame with one row per library and columns ``site_name``,
        ``sequencing_lab``, ``date``, ``rrna_fraction``, ``vv_fraction`` and
        ``ribodepletion_status``.
    """
    if kraken_data is None:
        kraken_data = load_all_kraken_data()
    if site_data is None:
        site_data = load_all_site_data()
    if vv_family_data is None:
        vv_family_data = load_all_vv_family_data()

    # calculate_vv_fraction_per_library carries site_name but neither the
    # laboratory nor the date, so both come from the Kraken table. rRNA fraction
    # is taken from there too, and deliberately not from the VV-family frame:
    # that frame ships a stale `rrna_fraction` which the function drops, but
    # which disagrees with the Kraken value for nearly every SB library here.
    vv = calculate_vv_fraction_per_library(vv_family_data, site_data)
    meta = kraken_data[['sra_accession', 'sequencing_lab', 'date',
                        'rrna_fraction']].drop_duplicates()
    df = vv.merge(meta, on='sra_accession')

    # Applied to the whole frame before any date restriction. The filter drops
    # the *first* MU sample at each Boston system, which is 2024-03-20 and
    # outside the span used here; handing it a pre-restricted frame would make
    # it delete a 2025-02 point instead.
    df = filter_timeseries_data(df)

    df = df[df['site_name'].isin(SITES)]
    df = df[df['date'] <= DATE_MAX]

    # Per-site restriction to the shared laboratory span, so MU and SB
    # out-of-window medians cover the same calendar months.
    spans = comparison_spans(df)
    df = df[[date >= spans[site]
             for site, date in zip(df['site_name'], df['date'])]]

    df = df.copy()
    df['ribodepletion_status'] = [
        ribodepletion_status(lab, date)
        for lab, date in zip(df['sequencing_lab'], df['date'])
    ]

    return df.sort_values(['site_name', 'sequencing_lab', 'date'])


def _fold_change(group):
    """Median VV fraction inside the window against outside it.

    Returns:
        (ratio, n_inside, n_outside). Ratio is None when either side is empty.
    """
    inside = group[group['ribodepletion_status'] == STATUS_OMITTED]
    if group['sequencing_lab'].iloc[0] == 'MU':
        # MU has no omission, so the comparison is the same calendar window.
        inside = group[(group['date'] >= OMISSION_START)
                       & (group['date'] <= OMISSION_END)]
    outside = group.drop(inside.index)
    if inside.empty or outside.empty:
        return None, len(inside), len(outside)
    return (inside['vv_fraction'].median() / outside['vv_fraction'].median(),
            len(inside), len(outside))


def print_ribodepletion_statistics(df=None):
    """Report the effect sizes and the affected accessions."""
    if df is None:
        df = prepare_ribodepletion_data()

    print("\n" + "=" * 78)
    print("Ribodepletion omission: rRNA and VV fraction inside vs outside the "
          "window")
    print("=" * 78)
    print(f"Window: {OMISSION_START.date()} to {OMISSION_END.date()} inclusive "
          f"(collection dates)")
    print("-" * 78)
    print(f"{'Site':<24} {'Lab':<4} {'n in':>5} {'n out':>6} "
          f"{'VV in':>10} {'VV out':>10} {'change':>10} "
          f"{'rRNA in':>8} {'rRNA out':>9}")
    print("-" * 78)
    for site in SITES:
        for lab in ('MU', 'SB'):
            g = df[(df['site_name'] == site) & (df['sequencing_lab'] == lab)]
            if g.empty:
                continue
            ratio, n_in, n_out = _fold_change(g)
            if lab == 'MU':
                inside = g[(g['date'] >= OMISSION_START)
                           & (g['date'] <= OMISSION_END)]
            else:
                inside = g[g['ribodepletion_status'] == STATUS_OMITTED]
            outside = g.drop(inside.index)
            change = '' if ratio is None else (
                f'{"↑" if ratio >= 1 else "↓"} {max(ratio, 1 / ratio):.1f}×')
            print(f"{site:<24} {lab:<4} {n_in:5d} {n_out:6d} "
                  f"{inside['vv_fraction'].median():10.2e} "
                  f"{outside['vv_fraction'].median():10.2e} {change:>10} "
                  f"{inside['rrna_fraction'].median():8.3f} "
                  f"{outside['rrna_fraction'].median():9.3f}")
    print("-" * 78)
    print("At Boston both laboratories processed the same wastewater on the "
          "same dates, so\nthe MU rows are a control: a seasonal decline would "
          "appear in both arms.")

    omitted = df[df['ribodepletion_status'] == STATUS_OMITTED]
    print(f"\nSamples with RiboPOOLs omitted: {len(omitted)}")
    for site, g in omitted.groupby('site_name', sort=False):
        print(f"  {site} (n = {len(g)}):")
        for _, row in g.iterrows():
            print(f"    {row['date'].date()}  {row['sra_accession']}  "
                  f"rRNA {row['rrna_fraction']:.3f}  "
                  f"VV {row['vv_fraction']:.2e}")
    print("=" * 78 + "\n")


def _draw_series(ax, group, value_col, color, linestyle, scatter_zorder,
                 line_zorder):
    """Per-sample values, connected in collection order.

    Deliberately unsmoothed. The claim this figure makes is a step change at a
    known date, and the house 5-week MMWR kernel spans the window edges, so it
    rounds off exactly the transitions that carry the result.
    """
    ax.plot(group['date'], group[value_col], color=color, linestyle=linestyle,
            linewidth=1.6, marker='o', markersize=3.6, alpha=0.9,
            zorder=line_zorder)


def _draw_timeseries_cell(ax, group, lab, site, vv_limits, show_xlabels,
                          show_left_label, show_right_label):
    """One site-and-laboratory cell: rRNA on the left axis, VV on the right."""
    # The band is filled only where the probes were actually omitted. MU never
    # ribodepletes, so filling its band identically would assert a protocol
    # deviation that did not happen there; the outline marks the same calendar
    # interval purely as the reference the comparison needs.
    if lab == 'SB':
        ax.axvspan(OMISSION_START, OMISSION_END,
                   facecolor=OMITTED_BAND_COLOR, alpha=0.10,
                   zorder=0, linewidth=0)
    else:
        ax.axvspan(OMISSION_START, OMISSION_END, facecolor='none',
                   edgecolor='#999999', linestyle=(0, (4, 3)), linewidth=1.0,
                   zorder=0)

    _draw_series(ax, group, 'rrna_fraction', RRNA_COLOR, linestyle='--',
                 scatter_zorder=2, line_zorder=3)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='y', labelsize=TICK_FONT, colors=RRNA_COLOR)
    ax.spines['left'].set_color(RRNA_COLOR)
    if show_left_label:
        ax.set_ylabel('rRNA fraction', fontsize=LABEL_FONT, color=RRNA_COLOR)

    ax2 = ax.twinx()
    _draw_series(ax2, group, 'vv_fraction', LAB_COLORS[lab], linestyle='-',
                 scatter_zorder=4, line_zorder=5)
    ax2.set_yscale('log')
    ax2.set_ylim(vv_limits)
    ax2.tick_params(axis='y', labelsize=TICK_FONT, colors=LAB_COLORS[lab])
    # The twin owns the right spine; the primary's is turned off so the two do
    # not overdraw each other and the visible one is unambiguously the VV axis.
    ax.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(LAB_COLORS[lab])
    ax2.spines['right'].set_linewidth(1.4)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    if show_right_label:
        ax2.set_ylabel('VV read fraction', fontsize=LABEL_FONT,
                       color=LAB_COLORS[lab])

    ax.set_xlim(DATE_MIN, DATE_MAX)
    ax.spines['top'].set_visible(False)
    ax.grid(True, alpha=0.3, which='major', zorder=1)
    ax.set_axisbelow(True)

    # Site only: the laboratory is given once per column by the header bar.
    ax.set_title(site.rsplit(',', 1)[0], fontsize=TITLE_FONT)

    format_date_axis(ax, date_range_days=(DATE_MAX - DATE_MIN).days)
    if show_xlabels:
        ax.tick_params(axis='x', labelsize=TICK_FONT)
    else:
        ax.tick_params(axis='x', labelbottom=False)

    return ax2


def _draw_legend_cell(ax):
    """The blank MU x Miami slot, reused for the shared legend.

    Miami has no MU arm, so this cell holds no data. Putting the legend here
    rather than inside a data cell keeps all five unobstructed. The keys name
    the two series and the two shading conventions; which laboratory a column
    belongs to is given by the header bar, not repeated here.
    """
    ax.axis('off')
    ax.legend(
        handles=[
            Line2D([0], [0], color=RRNA_COLOR, linestyle='--', linewidth=1.6,
                   marker='o', markersize=3.6,
                   label='rRNA fraction (left axis, linear)'),
            Line2D([0], [0], color=SERIES_KEY_COLOR, linestyle='-',
                   linewidth=1.6, marker='o', markersize=3.6,
                   label='VV read fraction (right axis, log)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=OMITTED_BAND_COLOR,
                          alpha=0.10,
                          label='RiboPOOLs omitted (13 May – 1 Jul 2025)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='#999999',
                          linestyle=(0, (4, 3)), linewidth=1.0,
                          label='Same interval, probes never used'),
        ],
        loc='center', frameon=False, fontsize=ANNOT_FONT + 1)


def _draw_column_headers(fig, axes):
    """Grey bars above each column naming the processing laboratory.

    Replaces repeating the laboratory in every cell title. Positions are taken
    from the drawn axes so the bars track the gridspec rather than hard-coded
    figure coordinates.
    """
    for col, lab in enumerate(LAB_ORDER):
        spans = [ax.get_position() for (row, c), ax in axes.items() if c == col]
        x0 = min(s.x0 for s in spans)
        x1 = max(s.x1 for s in spans)
        fig.patches.append(plt.Rectangle(
            (x0, HEADER_BAR_Y), x1 - x0, HEADER_BAR_HEIGHT,
            transform=fig.transFigure, facecolor=HEADER_BAR_COLOR,
            edgecolor='none', zorder=5))
        fig.text((x0 + x1) / 2, HEADER_BAR_Y + HEADER_BAR_HEIGHT / 2, lab,
                 ha='center', va='center', fontsize=TITLE_FONT + 1,
                 fontweight='bold', color='#333333', zorder=6)


def plot_ribodepletion_effect(df=None, save_path=None):
    """Draw the ribodepletion-effect panel grid.

    Returns:
        (fig, axes) where axes maps (row, col) to the primary axis of that cell.
    """
    if df is None:
        df = prepare_ribodepletion_data()

    # Shared VV limits so every cell is read on the same scale.
    vv_limits = (df['vv_fraction'].min() * 0.4,
                 df['vv_fraction'].max() * 2.5)

    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.30,
                          top=0.93, bottom=0.07, left=0.07,
                          right=0.93)

    axes = {}

    leftmost = {row: min(c for (r, c) in CELLS if r == row)
                for row, _ in CELLS}
    rightmost = {row: max(c for (r, c) in CELLS if r == row)
                 for row, _ in CELLS}

    for (row, col), (site, lab) in sorted(CELLS.items()):
        ax = fig.add_subplot(gs[row, col])
        group = df[(df['site_name'] == site) & (df['sequencing_lab'] == lab)]
        _draw_timeseries_cell(
            ax, group, lab, site, vv_limits,
            show_xlabels=(row, col) in X_LABELLED_CELLS,
            show_left_label=col == leftmost[row],
            show_right_label=col == rightmost[row],
        )
        axes[(row, col)] = ax

    ax_legend = fig.add_subplot(gs[LEGEND_CELL])
    _draw_legend_cell(ax_legend)
    axes[LEGEND_CELL] = ax_legend

    _draw_column_headers(fig, axes)

    if save_path:
        save_figure(fig, save_path)
    return fig, axes


def main(save_path=OUTPUT_PATH):
    df = prepare_ribodepletion_data()
    print_ribodepletion_statistics(df)
    return plot_ribodepletion_effect(df, save_path=save_path)


if __name__ == '__main__':
    main()
