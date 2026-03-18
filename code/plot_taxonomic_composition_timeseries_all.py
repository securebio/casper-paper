#!/usr/bin/env python3
"""
Time series of taxonomic composition for ALL sites (supplementary figure).

Creates stacked bar charts showing taxonomic composition over time for
all sites, arranged in 2 columns. Sites are ordered by sequencing lab
(MU first, then SB), then by state, then alphabetically within state.

Taxonomic categories:
- Ribosomal: All rRNA reads regardless of taxonomic domain
- Bacteria: Non-rRNA bacterial reads
- Archaea: Non-rRNA archaeal reads
- Eukaryota: Non-rRNA eukaryotic reads
- Viruses: Non-rRNA viral reads
- Unclassified: Non-rRNA unclassified reads + direct assignments to root/cellular organisms
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
    load_all_kraken_data,
    load_all_site_data,
    prepare_taxonomic_fractions_rrna_separated,
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


def plot_taxonomic_composition_timeseries_all(kraken_data, save_path=None):
    """
    Create time series stacked bar charts of taxonomic composition for all sites.

    Sites are arranged in 2 columns, with MU-sequenced sites first (with gray header),
    then SB-sequenced sites (with gray header).

    Args:
        kraken_data: DataFrame from load_all_kraken_data()
        save_path: Optional path to save figure

    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    # Ensure sequencing_lab is present (merge with site_data if needed)
    if 'sequencing_lab' not in kraken_data.columns:
        site_data = load_all_site_data()
        merge_key = 'sra_accession' if 'sra_accession' in kraken_data.columns else 'site_name'
        lab_cols = [merge_key, 'sequencing_lab']
        lab_cols = [c for c in lab_cols if c in site_data.columns]
        kraken_data = kraken_data.merge(
            site_data[lab_cols].drop_duplicates(),
            on=merge_key,
            how='left'
        )

    # Get ordered sites
    mu_sites, sb_sites = get_ordered_sites(kraken_data)

    n_mu = len(mu_sites)
    n_sb = len(sb_sites)

    # Prepare taxonomic fractions with rRNA as separate category
    df = prepare_taxonomic_fractions_rrna_separated(kraken_data)

    # Get colors
    colors = get_taxonomic_colors()

    # Taxonomic categories in order for stacking (bottom to top)
    categories = ['virus', 'eukaryota', 'bacteria', 'archaea', 'unclassified', 'ribosomal']

    category_labels = {
        'unclassified': 'Unclassified',
        'ribosomal': 'Ribosomal',
        'bacteria': 'Bacteria',
        'archaea': 'Archaea',
        'eukaryota': 'Eukaryota',
        'virus': 'Viruses'
    }

    # Map categories to fraction column names
    fraction_cols = {
        'unclassified': 'fraction_unclassified_non_rrna',
        'ribosomal': 'fraction_ribosomal',
        'bacteria': 'fraction_bacteria_non_rrna',
        'archaea': 'fraction_archaea_non_rrna',
        'eukaryota': 'fraction_eukaryota_non_rrna',
        'virus': 'fraction_virus_non_rrna'
    }

    # Get global date range for shared x-axis
    date_min = df['date'].min()
    date_max = df['date'].max()
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
    legend_height = 0.5  # Height for legend at top

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
        top=0.94,
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
    mu_df = df[df['sequencing_lab'] == 'MU']
    sb_df = df[df['sequencing_lab'] == 'SB']

    # Plot MU sites (no date labels - they go on SB section only)
    _plot_sites_on_axes(axes_mu, mu_sites, mu_df, categories, fraction_cols, colors,
                        category_labels, date_min, date_max, date_range_days,
                        ncols, nrows_mu, show_date_labels=False)

    # Plot SB sites (show date labels on bottom row of both columns)
    _plot_sites_on_axes(axes_sb, sb_sites, sb_df, categories, fraction_cols, colors,
                        category_labels, date_min, date_max, date_range_days,
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

    # Add legend at top (from first axis that has data), reversed order
    first_ax_with_data = axes_mu[0] if len(axes_mu) > 0 else axes_sb[0]
    handles, labels = first_ax_with_data.get_legend_handles_labels()
    # Reverse order so Ribosomal is first (matches visual stacking top-to-bottom)
    handles = handles[::-1]
    labels = labels[::-1]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=6, fontsize=FONT_SIZE_BASE, frameon=False)

    # Add shared y-axis label
    fig.text(0.02, 0.5, 'Relative abundance', va='center', rotation='vertical',
             fontsize=FONT_SIZE_LARGE)

    if save_path:
        save_figure(fig, save_path)

    return fig, (axes_mu, axes_sb)


def _plot_sites_on_axes(axes, sites, df, categories, fraction_cols, colors,
                        category_labels, date_min, date_max, date_range_days,
                        ncols, nrows, show_date_labels=True):
    """
    Helper function to plot taxonomic composition on a set of axes.

    Args:
        axes: List of matplotlib axes
        sites: List of site names to plot
        df: DataFrame with taxonomic fractions
        categories: List of taxonomic categories
        fraction_cols: Dict mapping category to column name
        colors: Dict mapping category to color
        category_labels: Dict mapping category to display label
        date_min, date_max: Global date range
        date_range_days: Total date range in days
        ncols, nrows: Grid dimensions
        show_date_labels: If True, show date labels on bottom row
    """
    buffer = pd.Timedelta(days=7)
    max_bar_width = 10  # Maximum bar width in days

    for idx, site_name in enumerate(sites):
        ax = axes[idx]
        loc_data = df[df['site_name'] == site_name].sort_values('date')

        if len(loc_data) == 0:
            ax.set_visible(False)
            continue

        dates = loc_data['date']

        # Calculate bar width based on median sampling interval, with max limit
        if len(dates) > 1:
            intervals = dates.diff().dt.days.dropna()
            median_interval = np.median(intervals)
            bar_width = min(median_interval * 0.8, max_bar_width)
        else:
            bar_width = min(5, max_bar_width)

        # Create stacked bars
        bottom = np.zeros(len(dates))
        for cat in categories:
            heights = loc_data[fraction_cols[cat]].values
            # Only add label on first axis for legend
            label = category_labels[cat] if idx == 0 else ""
            ax.bar(dates, heights, bottom=bottom, width=bar_width,
                   color=colors[cat], edgecolor='none', label=label)
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


def main():
    """Command-line interface for generating taxonomic composition time series for all sites."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate taxonomic composition time series for all sites (supplementary figure)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='figures/taxonomic_composition_timeseries_all.png',
        help='Output file path for the figure'
    )

    args = parser.parse_args()

    print("Loading kraken data...")
    kraken_data = load_all_kraken_data()

    print(f"Creating taxonomic composition time series for all sites...")
    print(f"  Output path: {args.output}")

    plot_taxonomic_composition_timeseries_all(
        kraken_data,
        save_path=args.output
    )

    print("\nComplete - taxonomic composition time series (all sites) saved")


if __name__ == '__main__':
    main()
