#!/usr/bin/env python3
"""
Shared renderer for the standalone 4x4 pathogen time-series grid.

Two supplementary figures show the same grid of pathogens, sites and smoothing
as Figure 4c but against a different y quantity: `plot_vv_pathogen_read_counts`
plots unnormalized clade read counts, `plot_vv_pathogen_log` plots
Tobamovirus-normalized abundance on a log scale. Everything except the plotted
column, the symlog threshold and the tick labels is identical between them, so
the grid itself lives here rather than being copied into each script.

Axis sizing follows the rule used by the main-text figure: the top of each axis
is placed so the highest *smoothed* trace among the highlighted sites sits at
`PEAK_AXIS_FRACTION` of the axis height, and non-highlighted (grey) sites are
allowed to run past it. Sizing to raw per-sample values from grey sites instead
lets a single outlying sample compress every highlighted trace into the floor.
Because these axes are symlog rather than linear, the placement is solved
through the symlog transform so it means the same thing it does in Figure 4c.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.scale import SymmetricalLogTransform
from matplotlib.ticker import FixedFormatter, FixedLocator

sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *
from plot_config import _state_from_site_name

from data_loaders import load_all_relative_abundance, load_sample_metadata

from plot_vv_pathogen_combined import (
    add_seasonal_shading,
    plot_pathogen_panel_on_axis,
    PATHOGENS,
    N_COLS,
    N_ROWS,
    DEFAULT_HIGHLIGHT_SITES,
    MULTI_LAB_SITES,
)

# Fraction of axis height at which the tallest highlighted smoothed trace sits.
# Matches the main-text pathogen panel, which scales to `peak / 0.75`.
PEAK_AXIS_FRACTION = 0.75

# symlog linear-region scale. matplotlib's default of 1.0 gives the
# 0-to-linthresh band the same height as one full decade. Figures spanning only
# a few decades can pass a smaller value so the band takes less of the axis.
DEFAULT_LINSCALE = 1.0

FONT_MS = 24


def prepare_pathogen_data():
    """Load relative abundance, attach sequencing lab, apply timeseries filters."""
    df = load_all_relative_abundance()
    df['state'] = df['site_name'].apply(_state_from_site_name)

    if 'sequencing_lab' not in df.columns:
        meta = load_sample_metadata()
        df = df.merge(meta[['sra_accession', 'sequencing_lab']].drop_duplicates(),
                      on='sra_accession', how='left')

    return filter_timeseries_data(df)


def decade_ticks(y_top, y_floor_exp=0):
    """Tick locations for a symlog axis: 0, then every decade up to y_top.

    ``y_floor_exp`` is the exponent of the lowest decade to include, so an
    abundance axis reaching down to 1e-8 starts its decades there rather than
    at 1.
    """
    if y_top <= 0:
        return [0]
    n_decades = int(np.floor(np.log10(y_top)))
    if n_decades < y_floor_exp:
        return [0]
    return [0] + [10.0 ** k for k in range(y_floor_exp, n_decades + 1)]


def top_for_peak(peak, linthresh, fraction=PEAK_AXIS_FRACTION,
                 linscale=DEFAULT_LINSCALE):
    """Axis top placing ``peak`` at ``fraction`` of the height, floor at 0.

    On the linear axis of Figure 4c this is simply ``peak / fraction``. Here the
    axis is symlog, so the same proportion is solved in transformed space, where
    the transform of 0 is 0 and the linear region occupies a fixed share.
    """
    if peak <= 0:
        return linthresh
    transform = SymmetricalLogTransform(10, linthresh, linscale)
    peak_t = float(transform.transform(np.array([peak]))[0])
    return float(transform.inverted().transform(
        np.array([peak_t / fraction]))[0])


def highlighted_smoothed_peak(row_data, locations, highlight_sites, value_col):
    """Largest smoothed value across highlighted sites in ``row_data``."""
    has_lab = 'sequencing_lab' in row_data.columns
    peak = 0.0

    for site_name in locations:
        if site_name not in highlight_sites:
            continue
        site_data = row_data[row_data['site_name'] == site_name]
        # Multi-lab sites are highlighted only on their MU series; the SB series
        # is drawn grey and must not drive the limit.
        if site_name in MULTI_LAB_SITES and has_lab:
            site_data = site_data[site_data['sequencing_lab'] == 'MU']
        if len(site_data) < 2:
            continue
        smoothed = calculate_mmwr_smoothed_trend(
            site_data.sort_values('date'), 'date', value_col)
        if not smoothed.empty:
            peak = max(peak, smoothed['smoothed_values'].max())

    return peak


def row_top_from_highlight_peak(row_data, locations, highlight_sites, value_col,
                               linthresh, row_idx, n_rows,
                               linscale=DEFAULT_LINSCALE):
    """Axis top placing the tallest highlighted smoothed trace at 75% height.

    Mirrors the main-text figure. Grey sites and individual raw samples are
    allowed to run past the top, which is what keeps a single outlying sample
    from compressing every highlighted trace into the floor.
    """
    peak = highlighted_smoothed_peak(row_data, locations, highlight_sites, value_col)
    if peak <= 0:
        return None
    return top_for_peak(peak, linthresh, linscale=linscale)


def row_top_from_data_max(row_data, locations, highlight_sites, value_col,
                          linthresh, row_idx, n_rows,
                          linscale=DEFAULT_LINSCALE):
    """Axis top enclosing every drawn value, so nothing is clipped.

    Used where the plotted magnitude is itself the point and clipping the tallest
    samples would defeat the figure. Takes raw per-sample values for highlighted
    sites (which are drawn as scatter) and smoothed values for grey sites (which
    are drawn as lines only).
    """
    has_lab = 'sequencing_lab' in row_data.columns
    y_values = []

    for site_name in locations:
        site_data = row_data[row_data['site_name'] == site_name]
        if site_data.empty:
            continue
        is_multi = site_name in MULTI_LAB_SITES and has_lab
        series = ([(lab, site_data[site_data['sequencing_lab'] == lab])
                   for lab in ('MU', 'SB')] if is_multi
                  else [(None, site_data)])

        for lab, data in series:
            if data.empty:
                continue
            highlighted = (site_name in highlight_sites
                           and (lab is None or lab == 'MU'))
            if highlighted:
                y_values.extend(data[value_col].tolist())
            elif len(data) >= 2:
                smoothed = calculate_mmwr_smoothed_trend(
                    data.sort_values('date'), 'date', value_col)
                if not smoothed.empty:
                    y_values.extend(smoothed['smoothed_values'].tolist())

    if not y_values:
        return None
    y_max = max(y_values)
    if y_max <= 0:
        return None
    if row_idx < n_rows - 1:
        # Round up to the next whole decade, giving each row a labelled decade
        # of headroom above its data.
        return 10.0 ** (np.floor(np.log10(y_max)) + 1)
    # The final row already spans many decades; rounding up would add one that
    # holds no data, so take a small multiplicative headroom instead.
    return y_max * 1.2


def _site_legend_entries(ax):
    """Legend handles/labels for the highlighted sites, ordered MU then SB."""
    handles, labels = ax.get_legend_handles_labels()
    mu_items, sb_items = [], []
    for handle, label in zip(handles, labels):
        if 'Miami' in label or 'FL' in label or 'South Florida' in label:
            sb_items.append((handle, label))
        elif 'Boston' in label or 'DITP' in label:
            mu_items.append((handle, f'{label} (MU)'))
        else:
            mu_items.append((handle, label))
    mu_items.sort(key=lambda item: item[1])

    if len(mu_items) >= 4 and len(sb_items) >= 1:
        ordered = [mu_items[0], mu_items[1], sb_items[0], mu_items[2], mu_items[3]]
    else:
        ordered = mu_items + sb_items

    ordered_handles = [item[0] for item in ordered]
    ordered_labels = [item[1] for item in ordered]
    ordered_handles.append(
        Line2D([0], [0], color='#CCCCCC', linewidth=LINE_WIDTH * 1.5, alpha=0.6))
    ordered_labels.append('Other sites')
    return ordered_handles, ordered_labels


def render_pathogen_grid(value_col, linthresh, ylabel, tick_labeller,
                         y_floor_exp=0, ylabel_offset=0.06, save_path=None,
                         highlight_sites=None, df_pathogen=None,
                         row_top_rule=row_top_from_highlight_peak,
                         linscale=DEFAULT_LINSCALE):
    """Draw the standalone 4x4 pathogen grid for one y quantity.

    Args:
        value_col: Column of the relative-abundance table to plot.
        linthresh: symlog linear-region width, matched to the quantity. Must sit
            at or below the smallest meaningful non-zero value, otherwise the
            bulk of the data falls inside the linear region and the panels
            flatten.
        ylabel: Figure-level y-axis label.
        tick_labeller: Callable (ticks, row_index) -> list of label strings.
        y_floor_exp: Exponent of the lowest decade tick.
        ylabel_offset: Gap in figure fraction between the leftmost panel and the
            y-axis label. Needs widening for wide tick labels such as powers of
            ten, which would otherwise run into the label.
        save_path: Optional path to save the figure.
        highlight_sites: Sites drawn in color; defaults to the main-text set.
        df_pathogen: Pre-loaded data, else loaded here.
        row_top_rule: Callable deciding each row's axis top. Defaults to the
            main-text 75%-of-height rule; pass `row_top_from_data_max` where
            clipping the tallest samples would defeat the figure.
        linscale: symlog linear-region scale, see DEFAULT_LINSCALE.

    Returns:
        fig: Matplotlib figure
    """
    if highlight_sites is None:
        highlight_sites = DEFAULT_HIGHLIGHT_SITES
    if df_pathogen is None:
        df_pathogen = prepare_pathogen_data()

    location_order = df_pathogen.groupby('site_name').agg({
        'state': 'first', 'date': 'min',
    }).reset_index()
    location_order = sort_locations_by_state_and_name(location_order)
    pathogen_locations = location_order['site_name'].tolist()

    date_min = df_pathogen['date'].min()
    date_max = df_pathogen['date'].max()
    date_range_days = (date_max - date_min).days
    buffer_days = 7

    fig, axes = plt.subplots(
        N_ROWS, N_COLS, figsize=(20, 18),
        gridspec_kw={'hspace': 0.25, 'wspace': 0.15},
    )
    axes = np.asarray(axes)

    for idx, (taxid, label) in enumerate(PATHOGENS):
        row, col = idx // N_COLS, idx % N_COLS
        add_seasonal_shading(axes[row, col], date_min, date_max)
        plot_pathogen_panel_on_axis(
            axes[row, col], df_pathogen, taxid, label, pathogen_locations,
            highlight_sites, log_scale=True, value_col=value_col,
            linthresh=linthresh,
        )
        axes[row, col].set_yscale('symlog', linthresh=linthresh,
                                  linscale=linscale)

    # Limits are shared within a row so panels in the same row stay directly
    # comparable, and are driven by the highlighted smoothed peak rather than
    # raw grey-site values (see module docstring).
    for row_idx in range(N_ROWS):
        row_taxids = [PATHOGENS[row_idx * N_COLS + col][0] for col in range(N_COLS)]
        row_data = df_pathogen[df_pathogen['taxid'].isin(row_taxids)]
        if row_data.empty:
            continue

        y_top = row_top_rule(row_data, pathogen_locations, highlight_sites,
                             value_col, linthresh, row_idx, N_ROWS,
                             linscale=linscale)
        if y_top is None:
            continue

        for col in range(N_COLS):
            # Floor at 0 so zero-value samples land on the axis rather than
            # off it; symlog handles the 0-to-linthresh span linearly.
            axes[row_idx, col].set_ylim(0, y_top)

    xlim_left = date_min - pd.Timedelta(days=buffer_days)
    xlim_right = date_max + pd.Timedelta(days=buffer_days)

    for row in range(N_ROWS):
        for col in range(N_COLS):
            ax = axes[row, col]
            ax.set_xlim(xlim_left, xlim_right)
            ax.tick_params(axis='x', labelsize=FONT_MS + 2)
            ax.tick_params(axis='y', labelsize=FONT_MS + 4)
            ax.set_title(ax.get_title(), fontsize=FONT_MS + 2)

            # Ticks are set explicitly at 0 and every decade. The default
            # SymmetricalLogLocator caps the tick count and then strides whole
            # decades to stay under it, silently dropping interior ticks.
            ticks = decade_ticks(ax.get_ylim()[1], y_floor_exp=y_floor_exp)
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_minor_locator(plt.NullLocator())
            ax.yaxis.set_major_formatter(
                FixedFormatter(tick_labeller(ticks, row)))

            if row == N_ROWS - 1:
                format_date_axis(ax, date_range_days=max(date_range_days, 851))
            else:
                ax.tick_params(labelbottom=False)

    handles, labels = _site_legend_entries(axes[0, 0])

    # Panels are pushed left to -0.07 and the legend sits 0.05 above the top
    # row, matching the spacing of the other pathogen supplements. Negative x
    # positions are recovered by save_figure's bbox_inches='tight'.
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.99])

    effective_left, right_edge = -0.07, 0.98
    effective_width = right_edge - effective_left
    hspace_fraction = 0.06
    total_hspace = hspace_fraction * (N_COLS - 1)
    panel_width = (effective_width - total_hspace * effective_width) / N_COLS
    col_gap = hspace_fraction * effective_width
    panel_height = axes[0, 0].get_position().height
    row_positions = [axes[row, 0].get_position().y0 for row in range(N_ROWS)]

    for row in range(N_ROWS):
        for col in range(N_COLS):
            axes[row, col].set_position([
                effective_left + col * (panel_width + col_gap),
                row_positions[row], panel_width, panel_height,
            ])

    # Anchored by their lower edge so the legends grow upward off the top of
    # the figure rather than down over the first row of panel titles.
    legend_bottom_y = axes[0, 0].get_position().y1 + 0.05

    fig.legend(handles, labels, loc='lower left', ncol=3,
               bbox_to_anchor=(-0.08, legend_bottom_y), frameon=False,
               fontsize=FONT_MS, handlelength=1.5, columnspacing=1.0,
               handletextpad=0.5)

    winter_patch = mpatches.Patch(color='#E6F2FF', alpha=0.5, label='Winter (Dec-Feb)')
    summer_patch = mpatches.Patch(color='#FFF3B0', alpha=0.5, label='Summer (Jun-Aug)')
    fig.legend(handles=[winter_patch, summer_patch], loc='lower right', ncol=1,
               bbox_to_anchor=(0.92, legend_bottom_y), frameon=False,
               fontsize=FONT_MS)

    center_y = (axes[0, 0].get_position().y1 + axes[-1, 0].get_position().y0) / 2
    fig.text(axes[0, 0].get_position().x0 - ylabel_offset, center_y, ylabel,
             ha='center', va='center', rotation='vertical', fontsize=FONT_MS + 2)

    if save_path:
        save_figure(fig, save_path)

    return fig
