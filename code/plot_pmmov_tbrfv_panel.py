#!/usr/bin/env python3
"""
Combined PMMoV and TBRFV fraction over time panel.

Creates a two-row, three-column figure:
- Left column: Time series (MU-sequenced)
- Middle column: Time series (SB-sequenced)
- Right column: PMMoV vs ToBRFV correlation (MU top, SB bottom)

Gray title boxes and legend appear only on the top row.
Date x-axis labels appear only on the bottom row.
Both rows share x-axis within each column (left/middle).

Smoothing approach:
- Data is aggregated to MMWR epidemiological weeks using geometric mean
  (arithmetic mean of log1p-transformed values, then expm1 back-transformation)
- A 5-week centered moving average is applied to produce smoothed time series
  (CDC recommendation for site-level data)
- Smoothed values are plotted at the MMWR week midpoint (Wednesday)
- Raw scatter points are shown for highlighted sites only
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Import plotting config (includes calculate_mmwr_smoothed_trend for consistent smoothing)
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *
from plot_config import _state_from_site_name, MULTI_LAB_SITES, filter_timeseries_data
from data_loaders import load_all_kraken_data, load_all_site_data


# Default sites to highlight in color
DEFAULT_SITES = [
    'Columbia WWTP, MO',
    'Boston DITP North, MA',
    'Miami-Dade CDWWTP, FL',
    'Riverside WQCP, CA',
    'Chicago (CHI-A), IL',
]


def plot_pmmov_tbrfv_panel(kraken_data, virus_data, save_path=None,
                            highlight_sites=None):
    """
    Create a 2x3 figure showing PMMoV and TBRFV fractions over time across sites,
    plus PMMoV vs ToBRFV correlation scatter plots.

    Top row: PMMoV fraction (MU | SB | MU correlation)
    Bottom row: TBRFV fraction (MU | SB | SB correlation)

    Args:
        kraken_data: DataFrame from load_all_kraken_data()
        virus_data: DataFrame from load_all_site_data() (for total_read_pairs)
        save_path: Optional path to save figure
        highlight_sites: List of site names to highlight in color (default: DEFAULT_SITES)

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    if highlight_sites is None:
        highlight_sites = DEFAULT_SITES
    # Use kraken_data directly (already has total_read_pairs)
    df = kraken_data.copy()

    # Scale Kraken subset counts up to total read pairs, then compute fraction.
    # n_reads_profiled is the pre-QC Kraken subset size.
    df['scaling_factor'] = df['total_read_pairs'] / df['n_reads_profiled']

    df['scaled_pmmov_counts'] = df['n_reads_pmmov_12239_clade_non_rrna'] * df['scaling_factor']
    df['pmmov_fraction'] = df['scaled_pmmov_counts'] / df['total_read_pairs']

    df['scaled_tbrfv_counts'] = df['n_reads_tobrfv_3432872_clade_non_rrna'] * df['scaling_factor']
    df['tbrfv_fraction'] = df['scaled_tbrfv_counts'] / df['total_read_pairs']

    df['state'] = df['site_name'].apply(_state_from_site_name)

    # Add sequencing_lab before filtering (needed for lab-specific first-sample drops)
    if 'sequencing_lab' not in df.columns:
        from data_loaders import load_sample_metadata
        meta = load_sample_metadata()
        df = df.merge(meta[['sra_accession', 'sequencing_lab']], on='sra_accession', how='left')

    # Apply standard time series filtering
    df = filter_timeseries_data(df)

    # Sort locations alphabetically within states for consistent legend order
    # Multi-lab sites (e.g. Boston DITP) appear in both MU and SB lists
    site_labs = df.groupby(['site_name', 'sequencing_lab']).size().reset_index()[['site_name', 'sequencing_lab']]
    site_labs['state'] = site_labs['site_name'].apply(_state_from_site_name)
    site_labs = sort_locations_by_state_and_name(site_labs)

    mu_locations = site_labs[site_labs['sequencing_lab'] == 'MU']['site_name'].tolist()
    sb_locations = site_labs[site_labs['sequencing_lab'] == 'SB']['site_name'].tolist()

    # Calculate width ratios based on time periods covered
    date_min_global = df['date'].min()
    date_max_global = df['date'].max()
    buffer_days = 7

    # MU panel: use full time range
    mu_date_min = date_min_global - pd.Timedelta(days=buffer_days)
    mu_date_max = date_max_global + pd.Timedelta(days=buffer_days)

    # SB panel: get actual SB data date range and add same buffer
    sb_df = df[df['sequencing_lab'] == 'SB']
    sb_date_min_data = sb_df['date'].min()
    sb_date_min = sb_date_min_data - pd.Timedelta(days=buffer_days)
    sb_date_max = date_max_global + pd.Timedelta(days=buffer_days)

    mu_days = (mu_date_max - mu_date_min).days
    sb_days = (sb_date_max - sb_date_min).days

    # Width ratio should reflect time periods
    mu_to_sb_ratio = mu_days / sb_days if sb_days > 0 else 1

    # Create figure with 2 rows x 3 columns
    fig = plt.figure(figsize=(22, 8.5))

    # Create a 2x3 grid (more top margin for legend)
    # Third column is a square-ish correlation plot
    corr_width = 0.8  # relative width for correlation column
    gs = fig.add_gridspec(2, 3, width_ratios=[mu_to_sb_ratio, 1, corr_width],
                          wspace=0.12, hspace=0.12,
                          top=0.84, bottom=0.14, left=0.05, right=0.98)

    # Top row: PMMoV fraction
    ax_pmmov_mu = fig.add_subplot(gs[0, 0])
    ax_pmmov_sb = fig.add_subplot(gs[0, 1], sharey=ax_pmmov_mu)

    # Bottom row: TBRFV fraction (share x-axis with top row, share y-axis with top row)
    ax_tbrfv_mu = fig.add_subplot(gs[1, 0], sharex=ax_pmmov_mu, sharey=ax_pmmov_mu)
    ax_tbrfv_sb = fig.add_subplot(gs[1, 1], sharex=ax_pmmov_sb, sharey=ax_pmmov_mu)

    # Right column: correlation scatter plots
    ax_corr_mu = fig.add_subplot(gs[0, 2])
    ax_corr_sb = fig.add_subplot(gs[1, 2])

    # Filter data by sequencing lab for each column
    mu_df = df[df['sequencing_lab'] == 'MU']
    sb_df = df[df['sequencing_lab'] == 'SB']

    # Don't highlight multi-lab sites in SB panels (they are highlighted in MU only)
    sb_highlight = [s for s in highlight_sites if s not in MULTI_LAB_SITES]

    # Plot PMMoV fraction on top row
    _plot_pmmov_lines_on_axis(ax_pmmov_mu, mu_locations, mu_df, highlight_sites)
    _plot_pmmov_lines_on_axis(ax_pmmov_sb, sb_locations, sb_df, sb_highlight)

    # Plot TBRFV fraction on bottom row
    _plot_tbrfv_lines_on_axis(ax_tbrfv_mu, mu_locations, mu_df, highlight_sites)
    _plot_tbrfv_lines_on_axis(ax_tbrfv_sb, sb_locations, sb_df, sb_highlight)

    # Format PMMoV axes (top row)
    ax_pmmov_mu.set_ylabel('PMMoV fraction', fontsize=FONT_SIZE_LARGE)
    ax_pmmov_mu.set_yscale('log')
    ax_pmmov_mu.set_ylim(1e-6, None)  # Hardcode lower limit
    ax_pmmov_mu.set_xlim(mu_date_min, mu_date_max)
    ax_pmmov_mu.tick_params(axis='both', labelsize=FONT_SIZE_LARGE)
    ax_pmmov_mu.tick_params(labelbottom=False)  # No x-tick labels on top row
    ax_pmmov_mu.grid(True, alpha=0.3, zorder=1)
    ax_pmmov_mu.set_axisbelow(True)
    ax_pmmov_mu.spines['top'].set_visible(False)
    ax_pmmov_mu.spines['right'].set_visible(False)

    ax_pmmov_sb.set_xlim(sb_date_min, sb_date_max)
    ax_pmmov_sb.tick_params(axis='both', labelsize=FONT_SIZE_LARGE)
    ax_pmmov_sb.tick_params(labelleft=False, labelbottom=False)  # No y-tick or x-tick labels
    ax_pmmov_sb.grid(True, alpha=0.3, zorder=1)
    ax_pmmov_sb.set_axisbelow(True)
    ax_pmmov_sb.spines['top'].set_visible(False)
    ax_pmmov_sb.spines['right'].set_visible(False)

    # Format TBRFV axes (bottom row)
    ax_tbrfv_mu.set_ylabel('ToBRFV fraction', fontsize=FONT_SIZE_LARGE)
    ax_tbrfv_mu.set_xlim(mu_date_min, mu_date_max)
    ax_tbrfv_mu.tick_params(axis='both', labelsize=FONT_SIZE_LARGE)
    format_date_axis(ax_tbrfv_mu, date_range_days=731)
    ax_tbrfv_mu.grid(True, alpha=0.3, zorder=1)
    ax_tbrfv_mu.set_axisbelow(True)
    ax_tbrfv_mu.spines['top'].set_visible(False)
    ax_tbrfv_mu.spines['right'].set_visible(False)

    ax_tbrfv_sb.set_xlim(sb_date_min, sb_date_max)
    ax_tbrfv_sb.tick_params(axis='both', labelsize=FONT_SIZE_LARGE)
    ax_tbrfv_sb.tick_params(labelleft=False)  # No y-tick labels
    format_date_axis(ax_tbrfv_sb, date_range_days=731)  # Use same quarterly tick frequency
    ax_tbrfv_sb.grid(True, alpha=0.3, zorder=1)
    ax_tbrfv_sb.set_axisbelow(True)
    ax_tbrfv_sb.spines['top'].set_visible(False)
    ax_tbrfv_sb.spines['right'].set_visible(False)

    # =========================================================================
    # Right column: PMMoV vs ToBRFV correlation scatter plots
    # =========================================================================

    # Filter to samples with both values > 0
    corr_df = df[(df['pmmov_fraction'] > 0) & (df['tbrfv_fraction'] > 0)].copy()

    for ax_corr, seq_lab, lab_label, lab_highlight in [
            (ax_corr_mu, 'MU', 'MU-sequenced', highlight_sites),
            (ax_corr_sb, 'SB', 'SB-sequenced', sb_highlight)]:
        lab_data = corr_df[corr_df['sequencing_lab'] == seq_lab]

        if len(lab_data) > 2:
            log_pmmov = np.log10(lab_data['pmmov_fraction'])
            log_tbrfv = np.log10(lab_data['tbrfv_fraction'])

            # Plot non-highlighted sites in gray
            non_highlight = lab_data[~lab_data['site_name'].isin(lab_highlight)]
            ax_corr.scatter(non_highlight['pmmov_fraction'], non_highlight['tbrfv_fraction'],
                           color='#CCCCCC', s=SCATTER_SIZE_SMALL * 2.5,
                           alpha=0.6, edgecolors='none', zorder=2)

            # Plot highlighted sites in their respective colors
            for site_name in lab_highlight:
                loc_corr = lab_data[lab_data['site_name'] == site_name]
                if len(loc_corr) > 0:
                    color = get_location_color(site_name)
                    ax_corr.scatter(loc_corr['pmmov_fraction'], loc_corr['tbrfv_fraction'],
                                   color=color, s=SCATTER_SIZE_SMALL * 2.5,
                                   alpha=0.6, edgecolors='none', zorder=3)

            # Compute Pearson correlation on log-transformed values
            r, p = stats.pearsonr(log_pmmov, log_tbrfv)
            n = len(lab_data)

            # Best fit line in log space
            slope, intercept = np.polyfit(log_pmmov, log_tbrfv, 1)

            # Print statistics
            ratio = 10**intercept
            print(f"\n  {lab_label} PMMoV vs ToBRFV correlation:")
            print(f"    n = {n}")
            print(f"    R = {r:.4f}")
            print(f"    p = {p:.2e}")
            print(f"    slope (log-log) = {slope:.4f}")
            print(f"    intercept (log-log) = {intercept:.4f}")
            print(f"    10^intercept = {ratio:.4f}")
            print(f"    -> For every PMMoV read, there are typically ~{ratio:.2f} ToBRFV reads")
            fit_x = np.linspace(log_pmmov.min(), log_pmmov.max(), 100)
            fit_y = slope * fit_x + intercept
            ax_corr.plot(10**fit_x, 10**fit_y, color='black', linewidth=1.5,
                        linestyle='-', zorder=3)

            # Significance stars
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            else:
                stars = 'n.s.'

            ax_corr.text(0.05, 0.95, f'R = {r:.2f}{stars}',
                        transform=ax_corr.transAxes,
                        fontsize=FONT_SIZE_LARGE, va='top', ha='left')

        ax_corr.set_xscale('log')
        ax_corr.set_yscale('log')
        ax_corr.set_xlabel('PMMoV fraction', fontsize=FONT_SIZE_LARGE)
        ax_corr.tick_params(axis='both', labelsize=FONT_SIZE_LARGE)
        ax_corr.grid(True, alpha=0.3, zorder=1)
        ax_corr.set_axisbelow(True)
        ax_corr.spines['top'].set_visible(False)
        ax_corr.spines['right'].set_visible(False)

    # Shared y-limits and x-limits across both correlation panels (square axes)
    ylim_mu = ax_corr_mu.get_ylim()
    ylim_sb = ax_corr_sb.get_ylim()
    xlim_mu = ax_corr_mu.get_xlim()
    xlim_sb = ax_corr_sb.get_xlim()
    shared_lim = (min(ylim_mu[0], ylim_sb[0], xlim_mu[0], xlim_sb[0]),
                  max(ylim_mu[1], ylim_sb[1], xlim_mu[1], xlim_sb[1]) * 2)
    ax_corr_mu.set_ylim(shared_lim)
    ax_corr_sb.set_ylim(shared_lim)
    ax_corr_mu.set_xlim(shared_lim)
    ax_corr_sb.set_xlim(shared_lim)

    # Only bottom correlation panel gets x-axis label and bottom spine
    ax_corr_mu.tick_params(labelbottom=False, bottom=False)
    ax_corr_mu.set_xlabel('')
    ax_corr_mu.spines['bottom'].set_visible(False)
    ax_corr_mu.set_ylabel('ToBRFV fraction', fontsize=FONT_SIZE_LARGE)
    ax_corr_sb.set_ylabel('ToBRFV fraction', fontsize=FONT_SIZE_LARGE)

    # Shift correlation column right to add more space between columns 2 and 3
    corr_shift = 0.02
    for ax in [ax_corr_mu, ax_corr_sb]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + corr_shift, pos.y0, pos.width, pos.height])

    # Collect legend handles and labels from PMMoV row only
    handles_mu, labels_mu = ax_pmmov_mu.get_legend_handles_labels()
    handles_sb, labels_sb = ax_pmmov_sb.get_legend_handles_labels()

    # Add "Other sites" gray line entry
    from matplotlib.lines import Line2D
    other_sites_handle = Line2D([0], [0], color='#CCCCCC', linewidth=LINE_WIDTH * 1.5, alpha=0.6)

    # Add "(MU)" suffix to multi-lab sites in legend
    labels_mu = [f'{l} (MU)' if l in MULTI_LAB_SITES else l for l in labels_mu]

    # Combine all handles into one legend
    all_handles = list(handles_mu) + list(handles_sb) + [other_sites_handle]
    all_labels = list(labels_mu) + list(labels_sb) + ['Other sites']

    # Single combined legend above the PMMoV MU panel (with more vertical space)
    if all_handles:
        ax_pmmov_mu.legend(all_handles, all_labels, loc='upper left', ncol=3,
                          fontsize=FONT_SIZE_LARGE, frameon=False, bbox_to_anchor=(0, 1.50))


    # Add title boxes with gray backgrounds (on top row for all three columns)
    from matplotlib.patches import Rectangle

    bbox_pmmov_mu = ax_pmmov_mu.get_position()
    bbox_pmmov_sb = ax_pmmov_sb.get_position()
    bbox_corr_mu = ax_corr_mu.get_position()
    bbox_corr_sb = ax_corr_sb.get_position()

    title_height = 0.055

    # Top row title boxes - positioned AT the top of the PMMoV plot boxes
    box_bottom_top = bbox_pmmov_mu.y1

    # MU title box (over MU time series panel)
    rect_mu = Rectangle((bbox_pmmov_mu.x0, box_bottom_top), bbox_pmmov_mu.width, title_height,
                        transform=fig.transFigure,
                        facecolor='#CCCCCC', edgecolor='none',
                        zorder=1, alpha=0.5)
    fig.patches.append(rect_mu)

    # SB title box (over SB time series panel)
    rect_sb = Rectangle((bbox_pmmov_sb.x0, box_bottom_top), bbox_pmmov_sb.width, title_height,
                         transform=fig.transFigure,
                         facecolor='#CCCCCC', edgecolor='none',
                         zorder=1, alpha=0.5)
    fig.patches.append(rect_sb)

    # MU correlation title box (over top-right correlation panel)
    rect_corr_mu = Rectangle((bbox_corr_mu.x0, box_bottom_top), bbox_corr_mu.width, title_height,
                             transform=fig.transFigure,
                             facecolor='#CCCCCC', edgecolor='none',
                             zorder=1, alpha=0.5)
    fig.patches.append(rect_corr_mu)

    # SB correlation title box - between the two correlation panels
    box_bottom_bottom = bbox_corr_mu.y0 - title_height + 0.015
    rect_corr_sb = Rectangle((bbox_corr_sb.x0, box_bottom_bottom), bbox_corr_sb.width, title_height,
                             transform=fig.transFigure,
                             facecolor='#CCCCCC', edgecolor='none',
                             zorder=1, alpha=0.5)
    fig.patches.append(rect_corr_sb)

    # Add titles
    title_y_top = box_bottom_top + title_height / 2
    title_x_mu = bbox_pmmov_mu.x0 + bbox_pmmov_mu.width / 2
    fig.text(title_x_mu, title_y_top, 'MU-sequenced',
            fontsize=FONT_SIZE_LARGE, weight='normal',
            ha='center', va='center', zorder=10)

    title_x_sb = bbox_pmmov_sb.x0 + bbox_pmmov_sb.width / 2
    fig.text(title_x_sb, title_y_top, 'SB-sequenced',
            fontsize=FONT_SIZE_LARGE, weight='normal',
            ha='center', va='center', zorder=10)

    title_x_corr_mu = bbox_corr_mu.x0 + bbox_corr_mu.width / 2
    fig.text(title_x_corr_mu, title_y_top, 'MU-sequenced',
            fontsize=FONT_SIZE_LARGE, weight='normal',
            ha='center', va='center', zorder=10)

    title_y_bottom = box_bottom_bottom + title_height / 2
    title_x_corr_sb = bbox_corr_sb.x0 + bbox_corr_sb.width / 2
    fig.text(title_x_corr_sb, title_y_bottom, 'SB-sequenced',
            fontsize=FONT_SIZE_LARGE, weight='normal',
            ha='center', va='center', zorder=10)

    # Add panel labels (Nature style: lowercase bold)
    # a = MU time series, b = SB time series, c = correlation
    add_panel_labels(fig, [ax_pmmov_mu, ax_pmmov_sb, ax_corr_mu], ['a', 'b', 'c'], y_offset=0.05)

    if save_path:
        save_figure(fig, save_path)

    return fig, [[ax_pmmov_mu, ax_pmmov_sb, ax_corr_mu], [ax_tbrfv_mu, ax_tbrfv_sb, ax_corr_sb]]


def _plot_pmmov_lines_on_axis(ax, locations, df, highlight_sites=None):
    """
    Helper function to plot PMMoV fraction lines on a single axis.
    """
    if highlight_sites is None:
        highlight_sites = []

    # First plot all sites in gray (background)
    for site_name in locations:
        if site_name not in highlight_sites:
            loc_data = df[df['site_name'] == site_name].sort_values('date').reset_index(drop=True)

            if len(loc_data) >= 2:
                smoothed = calculate_mmwr_smoothed_trend(loc_data, 'date', 'pmmov_fraction')
                if not smoothed.empty:
                    ax.plot(smoothed['date'], smoothed['smoothed_values'],
                           color='#CCCCCC', linewidth=LINE_WIDTH,
                           alpha=0.6, zorder=1)

    # Then plot highlighted sites in color (foreground)
    for site_name in locations:
        if site_name in highlight_sites:
            loc_data = df[df['site_name'] == site_name].sort_values('date').reset_index(drop=True)

            if len(loc_data) > 0:
                color = get_location_color(site_name)

                # Plot raw scatter points in color
                ax.scatter(loc_data['date'], loc_data['pmmov_fraction'],
                          color=color, s=SCATTER_SIZE_SMALL * 2.5,
                          alpha=0.4, zorder=2, edgecolors='none')

                if len(loc_data) >= 2:
                    smoothed = calculate_mmwr_smoothed_trend(loc_data, 'date', 'pmmov_fraction')
                    if not smoothed.empty:
                        ax.plot(smoothed['date'], smoothed['smoothed_values'],
                               color=color, linewidth=LINE_WIDTH * 1.5,
                               alpha=LINE_ALPHA, label=site_name, zorder=3)
                else:
                    ax.plot(loc_data['date'], loc_data['pmmov_fraction'],
                           color=color, linewidth=LINE_WIDTH * 1.5,
                           alpha=LINE_ALPHA, label=site_name, zorder=3)


def _plot_tbrfv_lines_on_axis(ax, locations, df, highlight_sites=None):
    """
    Helper function to plot TBRFV fraction lines on a single axis.
    """
    if highlight_sites is None:
        highlight_sites = []

    # First plot all sites in gray (background)
    for site_name in locations:
        if site_name not in highlight_sites:
            loc_data = df[df['site_name'] == site_name].sort_values('date').reset_index(drop=True)

            if len(loc_data) >= 2:
                smoothed = calculate_mmwr_smoothed_trend(loc_data, 'date', 'tbrfv_fraction')
                if not smoothed.empty:
                    ax.plot(smoothed['date'], smoothed['smoothed_values'],
                           color='#CCCCCC', linewidth=LINE_WIDTH,
                           alpha=0.6, zorder=1)

    # Then plot highlighted sites in color (foreground)
    for site_name in locations:
        if site_name in highlight_sites:
            loc_data = df[df['site_name'] == site_name].sort_values('date').reset_index(drop=True)

            if len(loc_data) > 0:
                color = get_location_color(site_name)

                # Plot raw scatter points in color
                ax.scatter(loc_data['date'], loc_data['tbrfv_fraction'],
                          color=color, s=SCATTER_SIZE_SMALL * 2.5,
                          alpha=0.4, zorder=2, edgecolors='none')

                if len(loc_data) >= 2:
                    smoothed = calculate_mmwr_smoothed_trend(loc_data, 'date', 'tbrfv_fraction')
                    if not smoothed.empty:
                        ax.plot(smoothed['date'], smoothed['smoothed_values'],
                               color=color, linewidth=LINE_WIDTH * 1.5,
                               alpha=LINE_ALPHA, zorder=3)
                else:
                    ax.plot(loc_data['date'], loc_data['tbrfv_fraction'],
                           color=color, linewidth=LINE_WIDTH * 1.5,
                           alpha=LINE_ALPHA, zorder=3)


