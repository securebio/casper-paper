#!/usr/bin/env python3
"""
Time series of VV family composition for ALL sites (supplementary figure).

Creates stacked bar charts showing VV family relative abundance over time for
all sites, arranged in 2 columns. Sites are ordered by sequencing lab
(MU first, then SB), then by state, then alphabetically within state.

Top families by total abundance across all sites are shown separately,
with remaining families grouped into "Other".
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
from data_loaders import (
    load_all_vv_family_data,
    load_all_site_data,
    get_top_families,
    get_family_colors,
)


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


def plot_vv_family_timeseries_all(vv_family_data, save_path=None, n_top_families=9):
    """
    Create time series stacked bar charts of VV family composition for all sites.

    Sites are arranged in 2 columns, with MU-sequenced sites first (with gray header),
    then SB-sequenced sites (with gray header).

    Args:
        vv_family_data: DataFrame from load_all_vv_family_data()
        save_path: Optional path to save figure
        n_top_families: Number of top families to show separately (default 9)

    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    # Get ordered sites (need sequencing_lab; merge with site_data if not present)
    if 'sequencing_lab' not in vv_family_data.columns:
        site_data = load_all_site_data()
        merge_key = 'sra_accession' if 'sra_accession' in vv_family_data.columns else 'site_name'
        lab_cols = [merge_key, 'sequencing_lab']
        lab_cols = [c for c in lab_cols if c in site_data.columns]
        vv_family_data = vv_family_data.merge(
            site_data[lab_cols].drop_duplicates(),
            on=merge_key,
            how='left'
        )

    mu_sites, sb_sites = get_ordered_sites(vv_family_data)

    n_mu = len(mu_sites)
    n_sb = len(sb_sites)

    # Get top families
    top_families = get_top_families(vv_family_data, n=n_top_families)

    # Get family colors
    all_families = top_families + ['Other']
    family_colors = get_family_colors(all_families)

    # Prepare data with family fractions
    df = vv_family_data.copy()

    # Create "Other" category for remaining families
    df['family_display'] = df['name'].apply(
        lambda x: x if x in top_families else 'Other'
    )

    # Counts column name
    counts_col = 'clade_counts'

    # Calculate total VV counts per sample (sum across all families)
    group_cols = ['site_name', 'date', 'sequencing_lab']
    if 'sra_accession' in df.columns:
        group_cols = ['site_name', 'sra_accession', 'date', 'sequencing_lab']
    total_vv_per_lib = df.groupby(group_cols)[counts_col].sum().reset_index()
    total_vv_per_lib.rename(columns={counts_col: 'total_vv_counts'}, inplace=True)

    # Merge to get total VV counts
    df = df.merge(total_vv_per_lib, on=group_cols, how='left')

    # Calculate family fraction (fraction of all VV counts)
    df['family_fraction'] = df[counts_col] / df['total_vv_counts']
    df['family_fraction'] = df['family_fraction'].fillna(0)

    # Aggregate by family_display (to combine "Other" families)
    agg_group_cols = group_cols + ['family_display']
    df_agg = df.groupby(agg_group_cols).agg({
        'family_fraction': 'sum'
    }).reset_index()

    # Get global date range for shared x-axis
    date_min = df_agg['date'].min()
    date_max = df_agg['date'].max()
    # Use fixed date_range_days to ensure quarterly tick labels across all SI figures
    date_range_days = 731  # Jan 2024 - Jan 2026

    # Layout: 2 columns
    ncols = 2

    # Calculate rows needed for each section
    nrows_mu = int(np.ceil(n_mu / ncols))
    nrows_sb = int(np.ceil(n_sb / ncols))

    # Create figure with gridspec for flexible layout
    import matplotlib.gridspec as gridspec

    row_height = 1.0  # Height per subplot row
    header_height = 0.4  # Height for section headers
    legend_height = 0.6  # Height for legend at top

    total_height = legend_height + header_height + (nrows_mu * row_height) + header_height + (nrows_sb * row_height) + 0.5
    fig = plt.figure(figsize=(14, total_height))

    # Calculate relative heights for gridspec
    gs = gridspec.GridSpec(
        nrows_mu + nrows_sb + 2,  # +2 for headers
        ncols,
        figure=fig,
        height_ratios=[header_height/row_height] + [1]*nrows_mu + [header_height/row_height] + [1]*nrows_sb,
        hspace=0.35,
        wspace=0.15,
        left=0.08,
        right=0.98,
        top=0.92,
        bottom=0.04
    )

    # Create axes for MU section (skip row 0 which is for header)
    axes_mu = []
    for i in range(nrows_mu):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i + 1, j])  # +1 to skip header row
            axes_mu.append(ax)

    # Create axes for SB section (skip the SB header row)
    axes_sb = []
    sb_start_row = 1 + nrows_mu + 1  # After MU header + MU rows + SB header
    for i in range(nrows_sb):
        for j in range(ncols):
            ax = fig.add_subplot(gs[sb_start_row + i, j])
            axes_sb.append(ax)

    # Filter data by sequencing lab for each section
    mu_agg = df_agg[df_agg['sequencing_lab'] == 'MU']
    sb_agg = df_agg[df_agg['sequencing_lab'] == 'SB']

    # Plot MU sites (no date labels - they go on SB section only)
    _plot_sites_on_axes(axes_mu, mu_sites, mu_agg, all_families, family_colors,
                        date_min, date_max, date_range_days,
                        ncols, nrows_mu, show_date_labels=False)

    # Plot SB sites (show date labels on bottom row of both columns)
    _plot_sites_on_axes(axes_sb, sb_sites, sb_agg, all_families, family_colors,
                        date_min, date_max, date_range_days,
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

    # Add legend at top
    first_ax_with_data = axes_mu[0] if len(axes_mu) > 0 else axes_sb[0]
    handles, labels = first_ax_with_data.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=5, fontsize=FONT_SIZE_BASE, frameon=False)

    # Add shared y-axis label
    fig.text(0.02, 0.5, 'Relative abundance', va='center', rotation='vertical',
             fontsize=FONT_SIZE_LARGE)

    if save_path:
        save_figure(fig, save_path)

    return fig, (axes_mu, axes_sb)


def _plot_sites_on_axes(axes, sites, df_agg, all_families, family_colors,
                        date_min, date_max, date_range_days,
                        ncols, nrows, show_date_labels=True):
    """
    Helper function to plot VV family composition on a set of axes.

    Args:
        axes: List of matplotlib axes
        sites: List of site names to plot
        df_agg: DataFrame with aggregated family fractions
        all_families: List of all family names (including "Other")
        family_colors: Dict mapping family name to color
        date_min, date_max: Global date range
        date_range_days: Total date range in days
        ncols, nrows: Grid dimensions
        show_date_labels: If True, show date labels on bottom row
    """
    buffer = pd.Timedelta(days=7)
    max_bar_width = 10  # Maximum bar width in days

    for idx, site_name in enumerate(sites):
        ax = axes[idx]
        loc_data = df_agg[df_agg['site_name'] == site_name].copy()

        if len(loc_data) == 0:
            ax.set_visible(False)
            continue

        # Pivot to get families as columns
        pivot = loc_data.pivot_table(
            index='date',
            columns='family_display',
            values='family_fraction',
            aggfunc='sum'
        )
        pivot = pivot.fillna(0)

        # Ensure all families are present (even if zero)
        for fam in all_families:
            if fam not in pivot.columns:
                pivot[fam] = 0

        # Reorder columns: top families first, then Other
        pivot = pivot[all_families]
        pivot = pivot.sort_index()

        dates = pivot.index

        # Calculate bar width based on median sampling interval, with max limit
        if len(dates) > 1:
            dates_series = pd.Series(dates)
            intervals = dates_series.diff().dt.days.dropna()
            median_interval = np.median(intervals)
            bar_width = min(median_interval * 0.8, max_bar_width)
        else:
            bar_width = min(5, max_bar_width)

        # Create stacked bars
        bottom = np.zeros(len(dates))
        for fam in all_families:
            heights = pivot[fam].values
            # Only add label on first axis for legend
            label = fam if idx == 0 else ""
            ax.bar(dates, heights, bottom=bottom, width=bar_width,
                   color=family_colors[fam], edgecolor='none', label=label, alpha=0.8)
            bottom += heights

        # Format subplot
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: '0' if x == 0 else '1' if x == 1 else f'{x:.1f}'
        ))
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
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            format_date_axis(ax, date_range_days=date_range_days)
        else:
            ax.set_visible(False)
