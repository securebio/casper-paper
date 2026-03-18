#!/usr/bin/env python3
"""
Time series of mean raw read length per sample for ALL sites (supplementary figure).

Creates line/scatter plots showing mean raw read length over time for all sites,
arranged in 2 columns. Sites are ordered by sequencing lab (MU first, then SB),
then by state, then alphabetically within state.

Uses raw (pre-trimming) read length to show the sequencing configuration used.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Import plotting config
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *
from plot_config import _state_from_site_name

# Import data loading functions
from data_loaders import load_all_site_data, load_all_qc_data


def get_ordered_sites(site_data):
    """Get sites ordered by sequencing lab (MU first), then state, then alphabetically.

    Multi-lab sites (e.g. Boston DITP) appear in both lists.
    """
    site_labs = site_data.groupby(['site_name', 'sequencing_lab']).size().reset_index()[['site_name', 'sequencing_lab']]
    site_labs['state'] = site_labs['site_name'].apply(_state_from_site_name)
    site_labs = sort_locations_by_state_and_name(site_labs)

    mu_sites = site_labs[site_labs['sequencing_lab'] == 'MU']['site_name'].tolist()
    sb_sites = site_labs[site_labs['sequencing_lab'] == 'SB']['site_name'].tolist()

    return mu_sites, sb_sites


def plot_read_length_timeseries_all(qc_data, site_data, save_path=None):
    """
    Create time series scatter/line plots of raw read length for all sites.

    Sites are arranged in 2 columns, with MU-sequenced sites first (with gray header),
    then SB-sequenced sites (with gray header).

    Args:
        qc_data: DataFrame from load_all_qc_data()
        site_data: DataFrame from load_all_site_data() (for dates and sequencing_lab)
        save_path: Optional path to save figure

    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    # Merge QC data with site data to get dates and sequencing_lab
    merge_cols = ['site_name']
    if 'sra_accession' in qc_data.columns and 'sra_accession' in site_data.columns:
        merge_cols = ['sra_accession']

    site_cols = list(set(merge_cols + ['date', 'sequencing_lab']))
    site_cols = [c for c in site_cols if c in site_data.columns]
    merged = qc_data.merge(
        site_data[site_cols].drop_duplicates(),
        on=merge_cols,
        how='inner',
        suffixes=('', '_site')
    )
    if 'date_site' in merged.columns:
        merged['date'] = merged['date_site']
        merged = merged.drop(columns=['date_site'])

    # Get ordered sites
    mu_sites, sb_sites = get_ordered_sites(merged)

    n_mu = len(mu_sites)
    n_sb = len(sb_sites)

    # Get global date range for shared x-axis
    date_min = merged['date'].min()
    date_max = merged['date'].max()
    # Use fixed date_range_days to ensure quarterly tick labels across all SI figures
    date_range_days = 731  # Jan 2024 - Jan 2026

    # Layout: 2 columns
    ncols = 2

    # Calculate rows needed for each section
    nrows_mu = int(np.ceil(n_mu / ncols))
    nrows_sb = int(np.ceil(n_sb / ncols))

    # Create figure with gridspec for flexible layout
    import matplotlib.gridspec as gridspec

    row_height = 1.0
    header_height = 0.4
    legend_height = 0.3

    total_height = legend_height + header_height + (nrows_mu * row_height) + header_height + (nrows_sb * row_height) + 0.5
    fig = plt.figure(figsize=(14, total_height))

    gs = gridspec.GridSpec(
        nrows_mu + nrows_sb + 2,
        ncols,
        figure=fig,
        height_ratios=[header_height/row_height] + [1]*nrows_mu + [header_height/row_height] + [1]*nrows_sb,
        hspace=0.35,
        wspace=0.15,
        left=0.08,
        right=0.98,
        top=0.96,
        bottom=0.04
    )

    axes_mu = []
    for i in range(nrows_mu):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i + 1, j])
            axes_mu.append(ax)

    axes_sb = []
    sb_start_row = 1 + nrows_mu + 1
    for i in range(nrows_sb):
        for j in range(ncols):
            ax = fig.add_subplot(gs[sb_start_row + i, j])
            axes_sb.append(ax)

    # Filter data by sequencing lab for each section
    mu_data = merged[merged['sequencing_lab'] == 'MU']
    sb_data = merged[merged['sequencing_lab'] == 'SB']

    _plot_sites_on_axes(axes_mu, mu_sites, mu_data, date_min, date_max, date_range_days,
                        ncols, nrows_mu, show_date_labels=False)

    _plot_sites_on_axes(axes_sb, sb_sites, sb_data, date_min, date_max, date_range_days,
                        ncols, nrows_sb, show_date_labels=True)

    # Add section headers with gray backgrounds
    if len(axes_mu) > 0:
        bbox_mu_first = axes_mu[0].get_position()
        bbox_mu_last_col = axes_mu[min(1, len(axes_mu)-1)].get_position()

        header_y = bbox_mu_first.y1 + 0.018
        header_height_fig = 0.022

        rect_mu = Rectangle(
            (bbox_mu_first.x0, header_y),
            bbox_mu_last_col.x1 - bbox_mu_first.x0,
            header_height_fig,
            transform=fig.transFigure,
            facecolor='#CCCCCC',
            edgecolor='none',
            zorder=1,
            alpha=0.5
        )
        fig.patches.append(rect_mu)

        fig.text(
            (bbox_mu_first.x0 + bbox_mu_last_col.x1) / 2,
            header_y + header_height_fig / 2,
            'MU-sequenced',
            fontsize=FONT_SIZE_LARGE,
            weight='normal',
            ha='center',
            va='center',
            zorder=10
        )

    if len(axes_sb) > 0:
        bbox_sb_first = axes_sb[0].get_position()
        bbox_sb_last_col = axes_sb[min(1, len(axes_sb)-1)].get_position()

        header_y = bbox_sb_first.y1 + 0.018
        header_height_fig = 0.022

        rect_sb = Rectangle(
            (bbox_sb_first.x0, header_y),
            bbox_sb_last_col.x1 - bbox_sb_first.x0,
            header_height_fig,
            transform=fig.transFigure,
            facecolor='#CCCCCC',
            edgecolor='none',
            zorder=1,
            alpha=0.5
        )
        fig.patches.append(rect_sb)

        fig.text(
            (bbox_sb_first.x0 + bbox_sb_last_col.x1) / 2,
            header_y + header_height_fig / 2,
            'SB-sequenced',
            fontsize=FONT_SIZE_LARGE,
            weight='normal',
            ha='center',
            va='center',
            zorder=10
        )

    # Add shared y-axis label
    fig.text(0.02, 0.5, 'Mean read length (bp)', va='center', rotation='vertical',
             fontsize=FONT_SIZE_LARGE)

    if save_path:
        save_figure(fig, save_path)

    return fig, (axes_mu, axes_sb)


def _plot_sites_on_axes(axes, sites, merged_data, date_min, date_max, date_range_days,
                        ncols, nrows, show_date_labels=True):
    """
    Helper function to plot read length on a set of axes.

    Args:
        axes: List of matplotlib axes
        sites: List of site names to plot
        merged_data: DataFrame with read length and date data
        date_min, date_max: Global date range
        date_range_days: Total date range in days
        ncols, nrows: Grid dimensions
        show_date_labels: If True, show date labels on bottom row
    """
    buffer = pd.Timedelta(days=7)

    # Fixed y-axis range for read length (typically 100-151 bp for 2x100 or 2x150)
    y_min = 70
    y_max = 160

    for idx, site_name in enumerate(sites):
        ax = axes[idx]
        loc_data = merged_data[merged_data['site_name'] == site_name].sort_values('date')

        if len(loc_data) == 0:
            ax.set_visible(False)
            continue

        dates = loc_data['date']
        read_length = loc_data['raw_mean_mean_seq_len']

        # Get color for this site
        color = get_location_color(site_name)

        # Create scatter plot with line
        ax.plot(dates, read_length, color=color, linewidth=1.5, alpha=0.6, zorder=1)
        ax.scatter(dates, read_length, color=color, s=25, alpha=0.8, edgecolors='none', zorder=2)

        # Format subplot
        ax.set_ylim(y_min, y_max)
        ax.set_yticks([100, 125, 150])
        ax.set_yticklabels(['100', '', '150'])
        ax.tick_params(axis='both', labelsize=FONT_SIZE_SMALL)

        # Add site name as title (left-aligned)
        ax.set_title(site_name, fontsize=FONT_SIZE_SMALL, loc='left', pad=3)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set shared x-axis limits
        ax.set_xlim(date_min - buffer, date_max + buffer)

        # Only show y-axis labels on left column
        if idx % ncols != 0:
            ax.tick_params(labelleft=False)

        # Only show x-axis labels on bottom row if show_date_labels is True
        row = idx // ncols
        if row < nrows - 1:
            ax.tick_params(labelbottom=False)
        elif show_date_labels:
            format_date_axis(ax, date_range_days=date_range_days)
        else:
            ax.tick_params(labelbottom=False)

    # Hide unused axes, but for SB section ensure bottom row has date labels
    for idx in range(len(sites), len(axes)):
        ax = axes[idx]
        row = idx // ncols
        if row == nrows - 1 and show_date_labels:
            ax.set_xlim(date_min - buffer, date_max + buffer)
            ax.set_ylim(y_min, y_max)
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            format_date_axis(ax, date_range_days=date_range_days)
        else:
            ax.set_visible(False)
