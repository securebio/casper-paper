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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter

sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *
from plot_config import _state_from_site_name

from data_loaders import (
    load_all_relative_abundance,
    load_sample_metadata,
)

from plot_vv_pathogen_combined import (
    add_seasonal_shading,
    plot_pathogen_panel_on_axis,
    PATHOGENS,
    N_COLS,
    N_ROWS,
    DEFAULT_HIGHLIGHT_SITES,
    MULTI_LAB_SITES,
)

# Column holding read pairs assigned to the pathogen clade by the VV pipeline.
# Full-depth counts, not the shallow-profiling subsample.
COUNT_COL = 'clade_counts'

# symlog linear region: below 1 read the scale is linear, so 0 sits on the
# floor and a 1-read sample stays visually distinct from a non-detection.
LINTHRESH = 1

FONT_MS = 24
FONT_MS_PANEL = 38


def _format_count_tick(value, _pos=None):
    """Render decade ticks as read counts: 0, 1, 10, 100, 1k, 10k, 1M.

    Powers-of-ten notation is the matplotlib default on a symlog axis, but the
    quantity here is a plain read count and reads more directly unabbreviated
    up to 100 and with SI-style suffixes beyond.
    """
    if value == 0:
        return '0'
    magnitude = abs(value)
    if magnitude < 1_000:
        return f'{value:.0f}'
    if magnitude < 1_000_000:
        return f'{value / 1_000:.0f}k'
    return f'{value / 1_000_000:.0f}M'


def _decade_ticks(y_top):
    """Tick locations for a symlog count axis: 0, then every decade at or below y_top."""
    if y_top < 1:
        return [0]
    n_decades = int(np.floor(np.log10(y_top)))
    return [0] + [10 ** k for k in range(0, n_decades + 1)]


def _prepare_pathogen_data():
    """Load relative abundance, attach sequencing lab, apply timeseries filters."""
    df = load_all_relative_abundance()
    df['state'] = df['site_name'].apply(_state_from_site_name)

    if 'sequencing_lab' not in df.columns:
        meta = load_sample_metadata()
        df = df.merge(meta[['sra_accession', 'sequencing_lab']].drop_duplicates(),
                      on='sra_accession', how='left')

    return filter_timeseries_data(df)


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
    if highlight_sites is None:
        highlight_sites = DEFAULT_HIGHLIGHT_SITES

    df_pathogen = _prepare_pathogen_data()

    location_order = df_pathogen.groupby('site_name').agg({
        'state': 'first',
        'date': 'min',
    }).reset_index()
    location_order = sort_locations_by_state_and_name(location_order)
    pathogen_locations = location_order['site_name'].tolist()

    date_min = df_pathogen['date'].min()
    date_max = df_pathogen['date'].max()
    date_range_days = (date_max - date_min).days
    buffer_days = 7

    fig, axes_pathogen = plt.subplots(
        N_ROWS, N_COLS, figsize=(20, 18),
        gridspec_kw={'hspace': 0.25, 'wspace': 0.15},
    )
    axes_pathogen = np.asarray(axes_pathogen)

    for idx, (taxid, label) in enumerate(PATHOGENS):
        row, col = idx // N_COLS, idx % N_COLS
        ax = axes_pathogen[row, col]

        add_seasonal_shading(ax, date_min, date_max)

        plot_pathogen_panel_on_axis(
            ax, df_pathogen, taxid, label, pathogen_locations, highlight_sites,
            log_scale=True,
            value_col=COUNT_COL,
            linthresh=LINTHRESH,
        )

    # Shared ylim per row, matching the log-scale supplement. Counts span six
    # orders of magnitude between the respiratory and enteric rows, so a single
    # figure-wide limit would flatten the respiratory panels entirely.
    has_lab_col = 'sequencing_lab' in df_pathogen.columns
    for row_idx in range(N_ROWS):
        row_taxids = [PATHOGENS[row_idx * N_COLS + col][0] for col in range(N_COLS)]
        row_data = df_pathogen[df_pathogen['taxid'].isin(row_taxids)]
        if row_data.empty:
            continue

        y_values = []
        for site_name in pathogen_locations:
            site_data = row_data[row_data['site_name'] == site_name]
            is_multi = site_name in MULTI_LAB_SITES and has_lab_col

            if is_multi:
                for lab in ['MU', 'SB']:
                    lab_data = site_data[site_data['sequencing_lab'] == lab]
                    if lab_data.empty:
                        continue
                    is_highlighted = (lab == 'MU' and site_name in highlight_sites)
                    if is_highlighted:
                        y_values.extend(lab_data[COUNT_COL].tolist())
                    elif len(lab_data) >= 2:
                        smoothed = calculate_mmwr_smoothed_trend(
                            lab_data.sort_values('date'), 'date', COUNT_COL)
                        if not smoothed.empty:
                            y_values.extend(smoothed['smoothed_values'].tolist())
            else:
                if site_data.empty:
                    continue
                if site_name in highlight_sites:
                    y_values.extend(site_data[COUNT_COL].tolist())
                elif len(site_data) >= 2:
                    smoothed = calculate_mmwr_smoothed_trend(
                        site_data.sort_values('date'), 'date', COUNT_COL)
                    if not smoothed.empty:
                        y_values.extend(smoothed['smoothed_values'].tolist())

        if y_values:
            y_max = max(y_values)
            if row_idx < N_ROWS - 1:
                # Round up to the next whole decade, giving each respiratory row
                # a labelled decade of headroom above its data.
                y_top = 10.0 ** (np.floor(np.log10(y_max)) + 1)
            else:
                # The enteric row already spans seven decades. Rounding up here
                # would add an eighth that holds no data, so take a small
                # multiplicative headroom instead.
                y_top = y_max * 1.2
            for col in range(N_COLS):
                # Floor at 0 so zero-count samples land on the axis rather than
                # off it; symlog handles the 0-to-1 span linearly.
                axes_pathogen[row_idx, col].set_ylim(0, y_top)

    xlim_left = date_min - pd.Timedelta(days=buffer_days)
    xlim_right = date_max + pd.Timedelta(days=buffer_days)

    for row in range(N_ROWS):
        for col in range(N_COLS):
            ax = axes_pathogen[row, col]
            ax.set_xlim(xlim_left, xlim_right)
            ax.tick_params(axis='x', labelsize=FONT_MS + 2)
            ax.tick_params(axis='y', labelsize=FONT_MS + 4)
            ax.set_title(ax.get_title(), fontsize=FONT_MS + 2)

            # Ticks are placed explicitly at 0 and every decade. The default
            # SymmetricalLogLocator caps the tick count and then strides whole
            # decades to stay under it, which silently drops interior ticks
            # (the enteric row spans 7 decades and lost its 10 that way).
            ticks = _decade_ticks(ax.get_ylim()[1])
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_minor_locator(plt.NullLocator())

            if row == N_ROWS - 1:
                # Nine decades of labels do not fit legibly, so label every
                # other position starting at 0 (0, 10, 1k, 100k, 10M) while
                # every decade keeps its tick mark.
                ax.yaxis.set_major_formatter(FixedFormatter([
                    _format_count_tick(t) if i % 2 == 0 else ''
                    for i, t in enumerate(ticks)
                ]))
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(_format_count_tick))

            if row == N_ROWS - 1:
                format_date_axis(ax, date_range_days=max(date_range_days, 851))
            else:
                ax.tick_params(labelbottom=False)

    # Site legend, ordered to match the main figure (MU sites, then SB).
    handles, labels = axes_pathogen[0, 0].get_legend_handles_labels()
    mu_items, sb_items = [], []
    for h, label in zip(handles, labels):
        if 'Miami' in label or 'FL' in label or 'South Florida' in label:
            sb_items.append((h, label))
        elif 'Boston' in label or 'DITP' in label:
            mu_items.append((h, f'{label} (MU)'))
        else:
            mu_items.append((h, label))
    mu_items.sort(key=lambda x: x[1])

    if len(mu_items) >= 4 and len(sb_items) >= 1:
        ordered = [mu_items[0], mu_items[1], sb_items[0], mu_items[2], mu_items[3]]
    else:
        ordered = mu_items + sb_items

    ordered_handles = [item[0] for item in ordered]
    ordered_labels = [item[1] for item in ordered]
    ordered_handles.append(
        Line2D([0], [0], color='#CCCCCC', linewidth=LINE_WIDTH * 1.5, alpha=0.6))
    ordered_labels.append('Other sites')

    # Layout below mirrors the pathogen section of plot_vv_pathogen_combined_log
    # so the two supplements sit at identical spacing: panels pushed left to
    # -0.07, legend 0.05 above the top row, y-label 0.08 left of the panels.
    # Negative x positions are recovered by save_figure's bbox_inches='tight'.
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.99])

    effective_left = -0.07
    right_edge = 0.98
    effective_width = right_edge - effective_left

    hspace_fraction = 0.06
    total_hspace = hspace_fraction * (N_COLS - 1)
    panel_width = (effective_width - total_hspace * effective_width) / N_COLS
    col_gap = hspace_fraction * effective_width
    panel_height = axes_pathogen[0, 0].get_position().height

    row_positions = [axes_pathogen[row, 0].get_position().y0
                     for row in range(N_ROWS)]

    for row in range(N_ROWS):
        for col in range(N_COLS):
            axes_pathogen[row, col].set_position([
                effective_left + col * (panel_width + col_gap),
                row_positions[row], panel_width, panel_height,
            ])

    # Anchored by their lower edge so the legends grow upward off the top of the
    # figure rather than down over the first row of panel titles. In the log
    # supplement that space is occupied by panels a/b; here the grid reaches the
    # top, and save_figure's bbox_inches='tight' recovers the overflow.
    pathogen_top_y = axes_pathogen[0, 0].get_position().y1
    legend_bottom_y = pathogen_top_y + 0.05

    fig.legend(ordered_handles, ordered_labels, loc='lower left', ncol=3,
               bbox_to_anchor=(-0.08, legend_bottom_y), frameon=False, fontsize=FONT_MS,
               handlelength=1.5, columnspacing=1.0, handletextpad=0.5)

    winter_patch = mpatches.Patch(color='#E6F2FF', alpha=0.5, label='Winter (Dec-Feb)')
    summer_patch = mpatches.Patch(color='#FFF3B0', alpha=0.5, label='Summer (Jun-Aug)')
    fig.legend(handles=[winter_patch, summer_patch], loc='lower right', ncol=1,
               bbox_to_anchor=(0.92, legend_bottom_y), frameon=False, fontsize=FONT_MS)

    # y-label placed after repositioning so it tracks the moved panels.
    pathogen_center_y = (axes_pathogen[0, 0].get_position().y1
                         + axes_pathogen[-1, 0].get_position().y0) / 2
    ylabel_x = axes_pathogen[0, 0].get_position().x0 - 0.06
    fig.text(ylabel_x, pathogen_center_y, 'Read pairs assigned to pathogen clade',
             ha='center', va='center', rotation='vertical', fontsize=FONT_MS + 2)

    if save_path:
        save_figure(fig, save_path)

    return fig


if __name__ == '__main__':
    df = _prepare_pathogen_data()
    print_read_count_statistics(df)
    plot_vv_pathogen_read_counts(save_path='../figures/vv_pathogen_read_counts.png')
    print("Saved figures/vv_pathogen_read_counts.png")
