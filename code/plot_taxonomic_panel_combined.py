#!/usr/bin/env python3
"""
Combined taxonomic composition panel figure (Figure 3).

Creates a single large panel figure combining:
- (a) Aggregate composition across all samples (taxonomic + virus host)
- (b) Aggregate composition by site and sequencing partner (horizontal bars)
- (c) Temporal variation in composition for selected sites
- (d) Distribution of rRNA fraction by site (boxplots)
- (e) rRNA fraction and PMMoV relative abundance over time

Figure dimensions: 18x24 inches
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.lines import Line2D

# Import plotting config
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *
from plot_config import get_virus_host_colors, _state_from_site_name, MULTI_LAB_SITES, filter_timeseries_data

# Import data loading and aggregation functions
from data_loaders import (
    load_all_kraken_data,
    load_all_virus_host_data,
    prepare_host_fractions,
    load_all_site_data,
    load_sample_metadata,
    prepare_taxonomic_fractions_rrna_separated,
    aggregate_taxonomic_composition,
    aggregate_virus_host_composition,
    aggregate_taxonomic_composition_for_site,
)

# Default sites to show in time series panels (public site names)
DEFAULT_SITES = [
    'Columbia WWTP, MO',
    'Boston DITP North, MA',
    'Miami-Dade CDWWTP, FL',
    'Riverside WQCP, CA',
    'Chicago (CHI-A), IL',
]


# Consistent font sizes for the figure
LABEL_FONT = FONT_SIZE_LARGE + 5
TICK_FONT = FONT_SIZE_LARGE + 3
PANEL_LABEL_FONT = FONT_SIZE_LARGE + 8


def _get_sequencing_lab_column(df):
    """Add sequencing_lab column by merging with metadata if needed."""
    if 'sequencing_lab' in df.columns:
        return df
    meta = load_sample_metadata()
    if 'sra_accession' in df.columns and 'sra_accession' in meta.columns:
        df = df.merge(
            meta[['sra_accession', 'sequencing_lab']].drop_duplicates(),
            on='sra_accession', how='left',
        )
    return df


def _plot_rrna_lines_on_axis(ax, locations, df, highlight_sites=None):
    """Plot rRNA fraction lines on a single axis."""
    if highlight_sites is None:
        highlight_sites = []

    for site_name in locations:
        if site_name not in highlight_sites:
            loc_data = df[df['site_name'] == site_name].sort_values('date').reset_index(drop=True)
            if len(loc_data) >= 2:
                smoothed = calculate_mmwr_smoothed_trend(loc_data, 'date', 'rrna_fraction')
                if not smoothed.empty:
                    ax.plot(smoothed['date'], smoothed['smoothed_values'],
                           color='#CCCCCC', linewidth=LINE_WIDTH,
                           alpha=0.6, zorder=1)

    for site_name in locations:
        if site_name in highlight_sites:
            loc_data = df[df['site_name'] == site_name].sort_values('date').reset_index(drop=True)
            if len(loc_data) > 0:
                color = get_location_color(site_name)
                ax.scatter(loc_data['date'], loc_data['rrna_fraction'],
                          color=color, s=SCATTER_SIZE_SMALL * 2.5,
                          alpha=0.4, zorder=2, edgecolors='none')
                if len(loc_data) >= 2:
                    smoothed = calculate_mmwr_smoothed_trend(loc_data, 'date', 'rrna_fraction')
                    if not smoothed.empty:
                        ax.plot(smoothed['date'], smoothed['smoothed_values'],
                               color=color, linewidth=LINE_WIDTH * 1.5,
                               alpha=LINE_ALPHA, label=site_name, zorder=3)
                else:
                    ax.plot(loc_data['date'], loc_data['rrna_fraction'],
                           color=color, linewidth=LINE_WIDTH * 1.5,
                           alpha=LINE_ALPHA, label=site_name, zorder=3)


def _plot_pmmov_lines_on_axis(ax, locations, df, highlight_sites=None):
    """Plot PMMoV fraction lines on a single axis."""
    if highlight_sites is None:
        highlight_sites = []

    for site_name in locations:
        if site_name not in highlight_sites:
            loc_data = df[df['site_name'] == site_name].sort_values('date').reset_index(drop=True)
            if len(loc_data) >= 2:
                smoothed = calculate_mmwr_smoothed_trend(loc_data, 'date', 'pmmov_fraction')
                if not smoothed.empty:
                    ax.plot(smoothed['date'], smoothed['smoothed_values'],
                           color='#CCCCCC', linewidth=LINE_WIDTH,
                           alpha=0.6, zorder=1)

    for site_name in locations:
        if site_name in highlight_sites:
            loc_data = df[df['site_name'] == site_name].sort_values('date').reset_index(drop=True)
            if len(loc_data) > 0:
                color = get_location_color(site_name)
                ax.scatter(loc_data['date'], loc_data['pmmov_fraction'],
                          color=color, s=SCATTER_SIZE_SMALL * 2.5,
                          alpha=0.4, zorder=2, edgecolors='none')
                if len(loc_data) >= 2:
                    smoothed = calculate_mmwr_smoothed_trend(loc_data, 'date', 'pmmov_fraction')
                    if not smoothed.empty:
                        ax.plot(smoothed['date'], smoothed['smoothed_values'],
                               color=color, linewidth=LINE_WIDTH * 1.5,
                               alpha=LINE_ALPHA, zorder=3)
                else:
                    ax.plot(loc_data['date'], loc_data['pmmov_fraction'],
                           color=color, linewidth=LINE_WIDTH * 1.5,
                           alpha=LINE_ALPHA, zorder=3)


def _plot_bars_on_axis(ax, locations, df, location_stats, title):
    """Plot horizontal stacked bars on a single axis.

    Args:
        ax: Matplotlib axis
        locations: List of site names to plot
        df: DataFrame with merged kraken and virus data
        location_stats: DataFrame with location statistics
        title: Title for this panel (unused, kept for API compat)
    """
    categories = ['virus', 'eukaryota', 'bacteria', 'archaea', 'unclassified', 'ribosomal']
    colors = get_taxonomic_colors()

    compositions = []
    for site_name in locations:
        comp = aggregate_taxonomic_composition_for_site(df, site_name)
        if comp is None:
            comp = {cat: 0 for cat in categories}
        else:
            comp = comp['proportions']
        compositions.append(comp)

    y_positions = range(len(locations))
    left = np.zeros(len(locations))

    for cat in categories:
        widths = [comp[cat] for comp in compositions]
        ax.barh(y_positions, widths, left=left,
               color=colors[cat], edgecolor='none',
               height=0.8)
        left += widths

    ax.set_yticks(y_positions)
    ax.set_yticklabels(locations, fontsize=FONT_SIZE_LARGE)

    y_padding = 0.5
    ax.set_ylim(len(locations) - 1 + y_padding, -y_padding)

    current_state = None
    for idx, site_name in enumerate(locations):
        loc_stats = location_stats[location_stats['site_name'] == site_name]
        if len(loc_stats) > 0:
            state = loc_stats['state'].iloc[0]
            if state != current_state:
                if current_state is not None:
                    ax.axhline(idx - 0.5, color='gray', linestyle='--',
                              linewidth=0.8, alpha=0.4, zorder=1)
                current_state = state


def _plot_rrna_boxplot_on_axis(ax, locations, kraken_data, show_xlabel=True):
    """Plot horizontal rRNA fraction boxplots on a single axis.

    Args:
        ax: Matplotlib axis
        locations: List of site names to plot
        kraken_data: DataFrame with kraken data including rrna_fraction
        show_xlabel: If True, show x-axis label
    """
    df = kraken_data.copy()
    if 'state' not in df.columns:
        df['state'] = df['site_name'].apply(_state_from_site_name)

    data_to_plot = []
    positions = []
    colors_list = []

    for idx, site_name in enumerate(locations):
        loc_data = df[df['site_name'] == site_name]['rrna_fraction'].values
        if len(loc_data) > 0:
            data_to_plot.append(loc_data)
            positions.append(idx)
            colors_list.append(get_location_color(site_name))

    if len(data_to_plot) == 0:
        return

    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False, vert=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    boxprops=dict(linewidth=1),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1))

    for idx, (patch, color) in enumerate(zip(bp['boxes'], colors_list)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for idx, site_name in enumerate(locations):
        loc_data = df[df['site_name'] == site_name]['rrna_fraction'].values
        if len(loc_data) > 0:
            y_jitter = np.random.normal(0, 0.08, size=len(loc_data))
            y_positions = idx + y_jitter
            ax.scatter(loc_data, y_positions,
                      alpha=0.4, s=SCATTER_SIZE_SMALL,
                      color=get_location_color(site_name),
                      edgecolors='black', linewidths=0.3,
                      zorder=3)

    ax.set_yticks([])
    ax.set_ylim(len(locations) - 1 + 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: '0' if x == 0 else '1' if x == 1 else f'{x:.1f}'))

    if show_xlabel:
        ax.set_xlabel('rRNA fraction', fontsize=FONT_SIZE_LARGE)
    else:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False, bottom=False)

    ax.tick_params(axis='x', labelsize=FONT_SIZE_LARGE)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not show_xlabel:
        ax.spines['bottom'].set_visible(False)


def plot_panel_a(ax_tax, ax_virus, fig, kraken_data, virus_host_data, virus_data):
    """Plot panel (a): Total composition with virus host breakdown.

    Two vertical stacked bars with connecting lines.
    Returns legend handles and labels for use by other panels.
    """
    df_tax = prepare_taxonomic_fractions_rrna_separated(kraken_data)
    if 'total_read_pairs' not in df_tax.columns:
        df_tax = df_tax.merge(
            virus_data[['site_name', 'sra_accession', 'total_read_pairs']].drop_duplicates(),
            on=['site_name', 'sra_accession'],
            how='inner'
        )

    df_virus = prepare_host_fractions(virus_host_data)

    tax_result = aggregate_taxonomic_composition(df_tax)
    virus_result = aggregate_virus_host_composition(df_virus, virus_data)

    bar_width = 0.3

    # --- Left: Taxonomic composition ---
    tax_colors = get_taxonomic_colors()
    tax_categories = ['virus', 'eukaryota', 'bacteria', 'archaea', 'unclassified', 'ribosomal']
    tax_labels = {
        'unclassified': 'Unclassified',
        'ribosomal': 'Ribosomal',
        'bacteria': 'Bacteria',
        'archaea': 'Archaea',
        'eukaryota': 'Eukaryota',
        'virus': 'Viruses'
    }

    virus_bottom = 0
    virus_top = 0

    bottom = 0
    for cat in tax_categories:
        height = tax_result['proportions'][cat]
        ax_tax.bar(0, height, bottom=bottom, width=bar_width,
                   color=tax_colors[cat], edgecolor='none',
                   label=tax_labels[cat])
        if cat == 'virus':
            virus_bottom = bottom
            virus_top = bottom + height
        bottom += height

    ax_tax.set_ylim(0, 1)
    ax_tax.set_xlim(-0.25, 0.25)
    ax_tax.set_xticks([])
    ax_tax.set_ylabel('Relative abundance', fontsize=LABEL_FONT)
    ax_tax.set_title('All reads', fontsize=LABEL_FONT, pad=10)
    ax_tax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: '0' if x == 0 else '1' if x == 1 else f'{x:.1f}'
    ))
    ax_tax.tick_params(axis='y', labelsize=TICK_FONT)
    ax_tax.spines['top'].set_visible(False)
    ax_tax.spines['right'].set_visible(False)
    ax_tax.spines['bottom'].set_visible(False)

    # --- Right: Virus host composition ---
    virus_host_colors = get_virus_host_colors()
    virus_colors = {
        'vertebrate': virus_host_colors['vertebrate'],
        'invertebrate': virus_host_colors['invertebrate'],
        'bacteria': virus_host_colors['bacteriophage'],
        'plant': virus_host_colors['plant'],
        'other': virus_host_colors['other'],
        'unknown_ambiguous': virus_host_colors['unknown'],
    }

    virus_categories = ['vertebrate', 'invertebrate', 'bacteria', 'plant', 'other', 'unknown_ambiguous']
    virus_labels = {
        'vertebrate': 'Vertebrate',
        'invertebrate': 'Invertebrate',
        'bacteria': 'Bacteria',
        'plant': 'Plant',
        'other': 'Other',
        'unknown_ambiguous': 'Unknown'
    }

    bottom = 0
    for cat in virus_categories:
        height = virus_result['proportions'][cat]
        if height > 0:
            ax_virus.bar(0, height, bottom=bottom, width=bar_width,
                       color=virus_colors[cat], edgecolor='none',
                       label=virus_labels[cat])
            bottom += height

    ax_virus.set_ylim(0, 1)
    ax_virus.set_xlim(-0.25, 0.25)
    ax_virus.set_xticks([])
    ax_virus.set_yticks([])
    ax_virus.set_ylabel('')
    ax_virus.set_title('Viral reads', fontsize=LABEL_FONT, pad=10)
    ax_virus.spines['top'].set_visible(False)
    ax_virus.spines['right'].set_visible(False)
    ax_virus.spines['bottom'].set_visible(False)
    ax_virus.spines['left'].set_visible(False)

    # Draw connecting lines
    left_bar_right_x = bar_width / 2
    right_bar_left_x = -bar_width / 2

    con_top = ConnectionPatch(
        xyA=(left_bar_right_x, virus_top), coordsA=ax_tax.transData,
        xyB=(right_bar_left_x, 1.0), coordsB=ax_virus.transData,
        color='black', linewidth=1.0, linestyle='-', alpha=0.7
    )
    fig.add_artist(con_top)

    con_bottom = ConnectionPatch(
        xyA=(left_bar_right_x, virus_bottom), coordsA=ax_tax.transData,
        xyB=(right_bar_left_x, 0.0), coordsB=ax_virus.transData,
        color='black', linewidth=1.0, linestyle='-', alpha=0.7
    )
    fig.add_artist(con_bottom)

    handles_tax, labels_tax = ax_tax.get_legend_handles_labels()
    handles_virus, labels_virus = ax_virus.get_legend_handles_labels()

    return {
        'tax_handles': handles_tax[::-1],
        'tax_labels': labels_tax[::-1],
        'virus_handles': handles_virus[::-1],
        'virus_labels': labels_virus[::-1]
    }


def plot_panel_c(axes, kraken_data, sites=None):
    """Plot panel (c): Time-series taxonomic composition for selected sites.

    Args:
        axes: List of axes, one per site
        kraken_data: DataFrame from load_all_kraken_data()
        sites: List of site names to plot
    """
    if sites is None:
        sites = DEFAULT_SITES

    available_sites = kraken_data['site_name'].unique()
    sites = [s for s in sites if s in available_sites]

    df = prepare_taxonomic_fractions_rrna_separated(kraken_data)

    colors = get_taxonomic_colors()
    categories = ['virus', 'eukaryota', 'bacteria', 'archaea', 'unclassified', 'ribosomal']

    fraction_cols = {
        'unclassified': 'fraction_unclassified_non_rrna',
        'ribosomal': 'fraction_ribosomal',
        'bacteria': 'fraction_bacteria_non_rrna',
        'archaea': 'fraction_archaea_non_rrna',
        'eukaryota': 'fraction_eukaryota_non_rrna',
        'virus': 'fraction_virus_non_rrna'
    }

    site_data = df[df['site_name'].isin(sites)]
    date_min = site_data['date'].min()
    date_max = site_data['date'].max()

    # For multi-lab sites, use only MU data in main-text panels
    df = _get_sequencing_lab_column(df)

    for idx, site_name in enumerate(sites):
        ax = axes[idx]
        loc_data = df[df['site_name'] == site_name].sort_values('date')
        if site_name in MULTI_LAB_SITES:
            loc_data = loc_data[loc_data['sequencing_lab'] == 'MU']

        if len(loc_data) == 0:
            continue

        dates = loc_data['date']

        if len(dates) > 1:
            intervals = dates.diff().dt.days.dropna()
            median_interval = np.median(intervals)
            bar_width = median_interval * 0.8
        else:
            bar_width = 5

        bottom = np.zeros(len(dates))
        for cat in categories:
            heights = loc_data[fraction_cols[cat]].values
            ax.bar(dates, heights, bottom=bottom, width=bar_width,
                   color=colors[cat], edgecolor='none')
            bottom += heights

        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: '0' if x == 0 else '1' if x == 1 else f'{x:.1f}'
        ))
        ax.tick_params(axis='both', labelsize=TICK_FONT)

        title = f'{site_name} (MU)' if site_name in MULTI_LAB_SITES else site_name
        ax.set_title(title, fontsize=LABEL_FONT, loc='left', pad=3)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if idx < len(sites) - 1:
            ax.tick_params(labelbottom=False)

    import matplotlib.dates as mdates
    date_min_padded = pd.Timestamp('2024-01-01') - pd.Timedelta(days=14)
    date_max_padded = pd.Timestamp('2026-01-01') + pd.Timedelta(days=14)
    axes[0].set_xlim(date_min_padded, date_max_padded)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')


def plot_panels_b_d(ax_bars_mu, ax_bars_sb, ax_box_mu, ax_box_sb,
                    kraken_data, virus_data):
    """Plot panels (b) and (d): Aggregated bars and rRNA boxplots.

    Split by sequencing partner (MU vs SB).
    """
    df = prepare_taxonomic_fractions_rrna_separated(kraken_data)

    # Merge with virus data for total_read_pairs
    merge_key = 'sra_accession' if 'sra_accession' in virus_data.columns else 'site_name'
    merge_cols = [merge_key, 'total_read_pairs']
    if 'site_name' not in merge_cols:
        merge_cols.append('site_name')
    df = df.merge(
        virus_data[merge_cols].drop_duplicates(),
        on=[c for c in [merge_key] if c in df.columns],
        how='inner',
        suffixes=('', '_vd'),
    )
    if 'total_read_pairs_vd' in df.columns:
        df['total_read_pairs'] = df['total_read_pairs_vd']
        df.drop(columns=['total_read_pairs_vd'], inplace=True)

    # Add metadata
    df['state'] = df['site_name'].apply(_state_from_site_name)
    df = _get_sequencing_lab_column(df)

    # Multi-lab sites (e.g. Boston DITP) appear in both MU and SB lists
    site_labs = df.groupby(['site_name', 'sequencing_lab']).size().reset_index()[['site_name', 'sequencing_lab']]
    site_labs['state'] = site_labs['site_name'].apply(_state_from_site_name)
    site_labs = sort_locations_by_state_and_name(site_labs)

    mu_locations = site_labs[site_labs['sequencing_lab'] == 'MU']['site_name'].tolist()
    sb_locations = site_labs[site_labs['sequencing_lab'] != 'MU']['site_name'].tolist()

    # Filter data by sequencing lab for each section
    mu_df = df[df['sequencing_lab'] == 'MU']
    sb_df = df[df['sequencing_lab'] != 'MU']

    _plot_bars_on_axis(ax_bars_mu, mu_locations, mu_df, site_labs, None)
    _plot_bars_on_axis(ax_bars_sb, sb_locations, sb_df, site_labs, None)

    # Add sequencing_lab to kraken_data for boxplot
    kraken_with_lab = _get_sequencing_lab_column(kraken_data.copy())
    # Convert rrna_fraction to QC-passed denominator
    if 'fraction_qc_filtered' in kraken_with_lab.columns:
        kraken_with_lab['rrna_fraction'] = kraken_with_lab['rrna_fraction'] / (1 - kraken_with_lab['fraction_qc_filtered'])
    mu_kraken = kraken_with_lab[kraken_with_lab['sequencing_lab'] == 'MU']
    sb_kraken = kraken_with_lab[kraken_with_lab['sequencing_lab'] != 'MU']
    _plot_rrna_boxplot_on_axis(ax_box_mu, mu_locations, mu_kraken, show_xlabel=False)
    _plot_rrna_boxplot_on_axis(ax_box_sb, sb_locations, sb_kraken, show_xlabel=True)

    for ax in [ax_bars_mu, ax_bars_sb]:
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: '0' if x == 0 else '1' if x == 1 else f'{x:.1f}'
        ))
        ax.tick_params(axis='both', labelsize=TICK_FONT)
        for label in ax.get_yticklabels():
            label.set_fontsize(TICK_FONT)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax_bars_mu.set_xlabel('')
    ax_bars_mu.tick_params(labelbottom=False, bottom=False)
    ax_bars_mu.spines['bottom'].set_visible(False)
    ax_bars_sb.set_xlabel('Relative abundance', fontsize=LABEL_FONT)

    return mu_locations, sb_locations


def plot_panel_e(ax_rrna_mu, ax_rrna_sb, ax_pmmov_mu, ax_pmmov_sb, fig,
                 kraken_data, virus_data, highlight_sites=None):
    """Plot panel (e): rRNA and PMMoV fraction time series.

    2x2 grid: (rRNA | PMMoV) x (MU | SB)
    """
    if highlight_sites is None:
        highlight_sites = DEFAULT_SITES

    # Merge data
    merge_key = 'sra_accession' if 'sra_accession' in virus_data.columns else 'site_name'
    merge_cols = [merge_key, 'total_read_pairs']
    if 'site_name' not in merge_cols:
        merge_cols.append('site_name')
    df = kraken_data.merge(
        virus_data[merge_cols].drop_duplicates(),
        on=[c for c in [merge_key] if c in kraken_data.columns],
        how='inner',
        suffixes=('', '_vd'),
    )
    if 'total_read_pairs_vd' in df.columns:
        df['total_read_pairs'] = df['total_read_pairs_vd']
        df.drop(columns=['total_read_pairs_vd'], inplace=True)

    # Scale Kraken subset PMMoV counts up to total read pairs, then compute fraction.
    # n_reads_profiled is the pre-QC Kraken subset size.
    df['scaling_factor'] = df['total_read_pairs'] / df['n_reads_profiled']
    df['scaled_pmmov_counts'] = df['n_reads_pmmov_12239_clade_non_rrna'] * df['scaling_factor']
    df['pmmov_fraction'] = df['scaled_pmmov_counts'] / df['total_read_pairs']

    # Convert rrna_fraction to QC-passed denominator
    if 'fraction_qc_filtered' in df.columns:
        df['rrna_fraction'] = df['rrna_fraction'] / (1 - df['fraction_qc_filtered'])

    df['state'] = df['site_name'].apply(_state_from_site_name)
    df = _get_sequencing_lab_column(df)
    df = filter_timeseries_data(df)

    # Multi-lab sites (e.g. Boston DITP) appear in both MU and SB lists
    site_labs_e = df.groupby(['site_name', 'sequencing_lab']).size().reset_index()[['site_name', 'sequencing_lab']]
    site_labs_e['state'] = site_labs_e['site_name'].apply(_state_from_site_name)
    site_labs_e = sort_locations_by_state_and_name(site_labs_e)

    mu_locations = site_labs_e[site_labs_e['sequencing_lab'] == 'MU']['site_name'].tolist()
    sb_locations = site_labs_e[site_labs_e['sequencing_lab'] != 'MU']['site_name'].tolist()

    import matplotlib.dates as mdates
    mu_date_min = pd.Timestamp('2024-01-01') - pd.Timedelta(days=14)
    mu_date_max = pd.Timestamp('2026-01-01') + pd.Timedelta(days=14)

    # Filter data by sequencing lab for each column
    mu_ts = df[df['sequencing_lab'] == 'MU']
    sb_ts = df[df['sequencing_lab'] != 'MU']
    sb_date_min_data = sb_ts['date'].min() if len(sb_ts) > 0 else mu_date_min
    sb_date_min = sb_date_min_data - pd.Timedelta(days=14)
    sb_date_max = mu_date_max

    # Don't highlight multi-lab sites in SB panels (they are highlighted in MU only)
    sb_highlight = [s for s in highlight_sites if s not in MULTI_LAB_SITES]

    # Plot rRNA
    _plot_rrna_lines_on_axis(ax_rrna_mu, mu_locations, mu_ts, highlight_sites)
    _plot_rrna_lines_on_axis(ax_rrna_sb, sb_locations, sb_ts, sb_highlight)

    # Plot PMMoV
    _plot_pmmov_lines_on_axis(ax_pmmov_mu, mu_locations, mu_ts, highlight_sites)
    _plot_pmmov_lines_on_axis(ax_pmmov_sb, sb_locations, sb_ts, sb_highlight)

    # Format rRNA axes
    ax_rrna_mu.set_ylabel('rRNA fraction', fontsize=LABEL_FONT)
    ax_rrna_mu.set_ylim(0, 1)
    ax_rrna_mu.set_xlim(mu_date_min, mu_date_max)
    ax_rrna_mu.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: '0' if x == 0 else '1' if x == 1 else f'{x:.1f}'
    ))
    ax_rrna_mu.tick_params(axis='both', labelsize=TICK_FONT)
    ax_rrna_mu.tick_params(labelbottom=False)
    ax_rrna_mu.grid(True, alpha=0.3, zorder=1)
    ax_rrna_mu.set_axisbelow(True)
    ax_rrna_mu.spines['top'].set_visible(False)
    ax_rrna_mu.spines['right'].set_visible(False)

    ax_rrna_sb.set_ylim(0, 1)
    ax_rrna_sb.set_xlim(sb_date_min, sb_date_max)
    ax_rrna_sb.tick_params(axis='both', labelsize=TICK_FONT)
    ax_rrna_sb.tick_params(labelleft=False, labelbottom=False)
    ax_rrna_sb.grid(True, alpha=0.3, zorder=1)
    ax_rrna_sb.set_axisbelow(True)
    ax_rrna_sb.spines['top'].set_visible(False)
    ax_rrna_sb.spines['right'].set_visible(False)

    # Format PMMoV axes
    ax_pmmov_mu.set_ylabel('PMMoV fraction', fontsize=LABEL_FONT)
    ax_pmmov_mu.set_yscale('log')
    ax_pmmov_mu.set_ylim(1e-6, None)
    ax_pmmov_mu.set_xlim(mu_date_min, mu_date_max)
    ax_pmmov_mu.tick_params(axis='both', labelsize=TICK_FONT)
    ax_pmmov_mu.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax_pmmov_mu.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax_pmmov_mu.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_pmmov_mu.grid(True, alpha=0.3, zorder=1)
    ax_pmmov_mu.set_axisbelow(True)
    ax_pmmov_mu.spines['top'].set_visible(False)
    ax_pmmov_mu.spines['right'].set_visible(False)

    ax_pmmov_sb.set_xlim(sb_date_min, sb_date_max)
    ax_pmmov_sb.tick_params(axis='both', labelsize=TICK_FONT)
    ax_pmmov_sb.tick_params(labelleft=False)
    ax_pmmov_sb.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax_pmmov_sb.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax_pmmov_sb.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_pmmov_sb.grid(True, alpha=0.3, zorder=1)
    ax_pmmov_sb.set_axisbelow(True)
    ax_pmmov_sb.spines['top'].set_visible(False)
    ax_pmmov_sb.spines['right'].set_visible(False)

    handles_mu, labels_mu = ax_rrna_mu.get_legend_handles_labels()
    handles_sb, labels_sb = ax_rrna_sb.get_legend_handles_labels()

    return handles_mu, labels_mu, handles_sb, labels_sb


def add_panel_label(fig, ax, label, offset_x=-0.02, offset_y=1.05):
    """Add panel label a, b, etc. to a subplot."""
    bbox = ax.get_position()
    fig.text(bbox.x0 + offset_x, bbox.y1 + offset_y * (1 - bbox.y1) * 0.1,
             label, fontsize=PANEL_LABEL_FONT, fontweight='bold',
             ha='left', va='bottom')


def plot_taxonomic_panel_combined(kraken_data, virus_host_data, virus_data, virus_hits_data,
                                  save_path=None):
    """Create the combined taxonomic panel figure.

    Layout:
    - Top row: (a) Total composition + virus host | (c) Time-series composition
    - Middle row: (b) Per-site bars | (d) rRNA boxplots
    - Bottom row: (e) rRNA + PMMoV time series

    Args:
        kraken_data: DataFrame from load_all_kraken_data()
        virus_host_data: DataFrame from load_all_virus_host_data()
        virus_data: DataFrame from load_all_site_data()
        virus_hits_data: DataFrame from load_all_site_data()
        save_path: Optional path to save figure.

    Returns:
        fig: Matplotlib figure.
    """
    # Add sequencing_lab to kraken_data for splitting
    kraken_with_lab = _get_sequencing_lab_column(kraken_data.copy())
    n_mu = kraken_with_lab[kraken_with_lab['sequencing_lab'] == 'MU']['site_name'].nunique()
    n_sb = kraken_with_lab[kraken_with_lab['sequencing_lab'] != 'MU']['site_name'].nunique()
    n_sites_timeseries = len(DEFAULT_SITES)

    fig = plt.figure(figsize=(18, 26))

    top_height = 5
    middle_height_mu = n_mu * 0.4
    middle_height_sb = n_sb * 0.4
    middle_height = middle_height_mu + middle_height_sb + 1
    bottom_height = 6

    gs_main = fig.add_gridspec(3, 1, height_ratios=[top_height, middle_height, bottom_height],
                                hspace=0.38, top=0.97, bottom=0.03, left=0.06, right=0.94)

    # =========================================================================
    # TOP ROW: Panel (a) and Panel (c)
    # =========================================================================
    gs_top = gs_main[0].subgridspec(1, 2, width_ratios=[0.6, 1.8], wspace=0.65)

    gs_panel_a = gs_top[0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.15)
    ax_tax = fig.add_subplot(gs_panel_a[0])
    ax_virus = fig.add_subplot(gs_panel_a[1])

    gs_panel_c = gs_top[1].subgridspec(n_sites_timeseries, 1, hspace=1.2)
    axes_timeseries = []
    for i in range(n_sites_timeseries):
        if i == 0:
            ax = fig.add_subplot(gs_panel_c[i])
        else:
            ax = fig.add_subplot(gs_panel_c[i], sharex=axes_timeseries[0])
        axes_timeseries.append(ax)

    # =========================================================================
    # MIDDLE ROW: Panel (b) and Panel (d)
    # =========================================================================
    gs_middle = gs_main[1].subgridspec(2, 2, height_ratios=[n_mu, n_sb],
                                        width_ratios=[2, 1], hspace=0.12, wspace=0.05)

    ax_bars_mu = fig.add_subplot(gs_middle[0, 0])
    ax_box_mu = fig.add_subplot(gs_middle[0, 1])
    ax_bars_sb = fig.add_subplot(gs_middle[1, 0])
    ax_box_sb = fig.add_subplot(gs_middle[1, 1])

    # =========================================================================
    # BOTTOM ROW: Panel (e)
    # =========================================================================
    df_temp = kraken_data.merge(
        virus_hits_data[['sra_accession', 'total_read_pairs']].drop_duplicates(),
        on='sra_accession',
        how='inner',
        suffixes=('', '_vd'),
    )
    if 'total_read_pairs_vd' in df_temp.columns:
        df_temp['total_read_pairs'] = df_temp['total_read_pairs_vd']
    df_temp = _get_sequencing_lab_column(df_temp)

    date_min_global = df_temp['date'].min()
    date_max_global = df_temp['date'].max()
    sb_df = df_temp[df_temp['sequencing_lab'] != 'MU']
    sb_date_min = sb_df['date'].min() if len(sb_df) > 0 else date_min_global

    mu_days = (date_max_global - date_min_global).days + 14
    sb_days = (date_max_global - sb_date_min).days + 14
    width_ratio = mu_days / sb_days if sb_days > 0 else 2

    gs_bottom = gs_main[2].subgridspec(2, 2, width_ratios=[width_ratio, 1],
                                        hspace=0.35, wspace=0.08)

    ax_rrna_mu = fig.add_subplot(gs_bottom[0, 0])
    ax_rrna_sb = fig.add_subplot(gs_bottom[0, 1], sharey=ax_rrna_mu)
    ax_pmmov_mu = fig.add_subplot(gs_bottom[1, 0], sharex=ax_rrna_mu)
    ax_pmmov_sb = fig.add_subplot(gs_bottom[1, 1], sharex=ax_rrna_sb, sharey=ax_pmmov_mu)

    # =========================================================================
    # PLOT ALL PANELS
    # =========================================================================

    legend_info = plot_panel_a(ax_tax, ax_virus, fig, kraken_data, virus_host_data, virus_data)

    plot_panel_c(axes_timeseries, kraken_data, sites=DEFAULT_SITES)

    mu_locs, sb_locs = plot_panels_b_d(ax_bars_mu, ax_bars_sb, ax_box_mu, ax_box_sb,
                                         kraken_data, virus_data)

    e_handles_mu, e_labels_mu, e_handles_sb, e_labels_sb = plot_panel_e(
        ax_rrna_mu, ax_rrna_sb, ax_pmmov_mu, ax_pmmov_sb, fig,
        kraken_data, virus_hits_data, highlight_sites=DEFAULT_SITES,
    )

    # =========================================================================
    # ADD LEGENDS
    # =========================================================================

    leg_tax = ax_tax.legend(legend_info['tax_handles'], legend_info['tax_labels'],
                            loc='upper right', bbox_to_anchor=(-0.5, 1.0), ncol=1,
                            fontsize=LABEL_FONT, frameon=False,
                            handletextpad=0.4, title='Taxonomic\ncomposition',
                            title_fontsize=LABEL_FONT)
    leg_tax._legend_box.align = 'left'

    leg_virus = ax_virus.legend(legend_info['virus_handles'], legend_info['virus_labels'],
                                loc='upper left', bbox_to_anchor=(0.9, 1.0), ncol=1,
                                fontsize=LABEL_FONT, frameon=False,
                                handletextpad=0.4, title='Host\ncomposition',
                                title_fontsize=LABEL_FONT)
    leg_virus._legend_box.align = 'left'

    # Panel (e) legend — add "(MU)" suffix to multi-lab sites
    e_labels_mu = [f'{l} (MU)' if l in MULTI_LAB_SITES else l for l in e_labels_mu]
    other_sites_handle = Line2D([0], [0], color='#CCCCCC', linewidth=LINE_WIDTH * 1.5, alpha=0.6)
    all_handles = list(e_handles_mu) + list(e_handles_sb) + [other_sites_handle]
    all_labels = list(e_labels_mu) + list(e_labels_sb) + ['Other sites']

    if all_handles:
        ax_rrna_mu.legend(all_handles, all_labels, loc='lower left', ncol=3,
                          fontsize=LABEL_FONT, frameon=False, bbox_to_anchor=(0, 1.20))

    # =========================================================================
    # ADD TITLE BOXES FOR MIDDLE ROW
    # =========================================================================

    fig.canvas.draw()

    for ax in [ax_box_mu, ax_box_sb]:
        ax.tick_params(axis='x', labelsize=TICK_FONT)
        for label in ax.get_xticklabels():
            label.set_fontsize(TICK_FONT)
    ax_box_sb.set_xlabel('rRNA fraction', fontsize=LABEL_FONT)

    bbox_bars_mu = ax_bars_mu.get_position()
    bbox_box_mu = ax_box_mu.get_position()
    bbox_bars_sb = ax_bars_sb.get_position()
    bbox_box_sb = ax_box_sb.get_position()

    title_height = 0.018
    title_height_mu = 0.022

    # MU title boxes
    rect_mu_bars = Rectangle((bbox_bars_mu.x0, bbox_bars_mu.y1), bbox_bars_mu.width, title_height_mu,
                              transform=fig.transFigure, facecolor='#CCCCCC', edgecolor='none',
                              zorder=1, alpha=0.5)
    fig.patches.append(rect_mu_bars)

    rect_mu_box = Rectangle((bbox_box_mu.x0, bbox_box_mu.y1), bbox_box_mu.width, title_height_mu,
                             transform=fig.transFigure, facecolor='#CCCCCC', edgecolor='none',
                             zorder=1, alpha=0.5)
    fig.patches.append(rect_mu_box)

    fig.text(bbox_bars_mu.x0 + bbox_bars_mu.width / 2, bbox_bars_mu.y1 + title_height_mu / 2,
             'MU-sequenced', fontsize=LABEL_FONT, ha='center', va='center', zorder=10)
    fig.text(bbox_box_mu.x0 + bbox_box_mu.width / 2, bbox_box_mu.y1 + title_height_mu / 2,
             'MU-sequenced', fontsize=LABEL_FONT, ha='center', va='center', zorder=10)

    # SB title boxes (in the gap)
    gap_bottom = bbox_bars_sb.y1
    gap_height = bbox_bars_mu.y0 - bbox_bars_sb.y1

    rect_sb_bars = Rectangle((bbox_bars_sb.x0, gap_bottom), bbox_bars_sb.width, gap_height,
                               transform=fig.transFigure, facecolor='#CCCCCC', edgecolor='none',
                               zorder=1, alpha=0.5)
    fig.patches.append(rect_sb_bars)

    rect_sb_box = Rectangle((bbox_box_sb.x0, gap_bottom), bbox_box_sb.width, gap_height,
                              transform=fig.transFigure, facecolor='#CCCCCC', edgecolor='none',
                              zorder=1, alpha=0.5)
    fig.patches.append(rect_sb_box)

    fig.text(bbox_bars_sb.x0 + bbox_bars_sb.width / 2, gap_bottom + gap_height / 2,
             'SB-sequenced', fontsize=LABEL_FONT, ha='center', va='center', zorder=10)
    fig.text(bbox_box_sb.x0 + bbox_box_sb.width / 2, gap_bottom + gap_height / 2,
             'SB-sequenced', fontsize=LABEL_FONT, ha='center', va='center', zorder=10)

    # Title boxes for panel (e)
    bbox_rrna_mu = ax_rrna_mu.get_position()
    bbox_rrna_sb = ax_rrna_sb.get_position()

    rect_e_mu = Rectangle((bbox_rrna_mu.x0, bbox_rrna_mu.y1), bbox_rrna_mu.width, title_height_mu,
                           transform=fig.transFigure, facecolor='#CCCCCC', edgecolor='none',
                           zorder=1, alpha=0.5)
    fig.patches.append(rect_e_mu)

    rect_e_sb = Rectangle((bbox_rrna_sb.x0, bbox_rrna_sb.y1), bbox_rrna_sb.width, title_height_mu,
                            transform=fig.transFigure, facecolor='#CCCCCC', edgecolor='none',
                            zorder=1, alpha=0.5)
    fig.patches.append(rect_e_sb)

    fig.text(bbox_rrna_mu.x0 + bbox_rrna_mu.width / 2, bbox_rrna_mu.y1 + title_height_mu / 2,
             'MU-sequenced', fontsize=LABEL_FONT, ha='center', va='center', zorder=10)
    fig.text(bbox_rrna_sb.x0 + bbox_rrna_sb.width / 2, bbox_rrna_sb.y1 + title_height_mu / 2,
             'SB-sequenced', fontsize=LABEL_FONT, ha='center', va='center', zorder=10)

    # =========================================================================
    # ADD PANEL LABELS
    # =========================================================================

    fig.text(ax_tax.get_position().x0 - 0.02, ax_tax.get_position().y1 + 0.02,
             'a', fontsize=PANEL_LABEL_FONT, fontweight='bold', ha='left', va='bottom')

    fig.text(ax_bars_mu.get_position().x0 - 0.02, bbox_bars_mu.y1 + title_height_mu + 0.005,
             'b', fontsize=PANEL_LABEL_FONT, fontweight='bold', ha='left', va='bottom')

    fig.text(axes_timeseries[0].get_position().x0 - 0.02, axes_timeseries[0].get_position().y1 + 0.02,
             'c', fontsize=PANEL_LABEL_FONT, fontweight='bold', ha='left', va='bottom')

    fig.text(ax_box_mu.get_position().x0 - 0.02, bbox_box_mu.y1 + title_height_mu + 0.005,
             'd', fontsize=PANEL_LABEL_FONT, fontweight='bold', ha='left', va='bottom')

    legend_y = bbox_rrna_mu.y0 + (bbox_rrna_mu.height * 1.20)
    fig.text(ax_rrna_mu.get_position().x0 - 0.02, legend_y + 0.04,
             'e', fontsize=PANEL_LABEL_FONT, fontweight='bold', ha='left', va='bottom')

    # =========================================================================
    # ADD SHARED AXIS LABELS
    # =========================================================================

    fig.text(axes_timeseries[0].get_position().x0 - 0.04,
             (axes_timeseries[0].get_position().y1 + axes_timeseries[-1].get_position().y0) / 2,
             'Relative abundance', fontsize=LABEL_FONT,
             ha='center', va='center', rotation='vertical')

    if save_path:
        save_figure(fig, save_path)

    return fig


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate combined taxonomic panel figure'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='figures/taxonomic_panel_combined.png',
        help='Output file path'
    )

    args = parser.parse_args()

    print("Loading data...")
    print("  Loading kraken data...")
    kraken_data = load_all_kraken_data()

    print("  Loading virus host data...")
    virus_host_data = load_all_virus_host_data()

    print("  Loading site data...")
    virus_data = load_all_site_data()

    print("  Loading virus hits data...")
    virus_hits_data = load_all_site_data()

    print(f"Creating combined taxonomic panel figure...")
    print(f"  Output path: {args.output}")

    plot_taxonomic_panel_combined(
        kraken_data,
        virus_host_data,
        virus_data,
        virus_hits_data,
        save_path=args.output,
    )

    print("\nDone - combined taxonomic panel figure saved")


if __name__ == '__main__':
    main()
