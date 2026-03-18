#!/usr/bin/env python3
"""
Combined visualization: sampling timeline + read counts + SRA comparison + cost.

Creates a 2-row figure:
- Top row (a, b): Sample collection timeline and read count distributions
- Bottom row (c, d, e): CASPER cumulative reads, SRA comparison, and median depth with cost

Figure 2 in the CASPER manuscript.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# Import plotting config
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *
from plot_config import _state_from_site_name

# Import data loading functions
from data_loaders import (
    load_all_site_data, load_sample_metadata,
    load_sra_timeline, load_sra_summary_statistics,
    print_sample_site_stats, print_sra_statistics,
    print_lit_review_statistics, print_cost_statistics,
    CASPER_BIOPROJECTS,
)

# Paths
_CODE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CODE_DIR.parent
_SUPP_DIR = _REPO_ROOT / "tables"


def _get_sequencing_lab_column(df):
    """Add sequencing_lab column to a DataFrame by merging with metadata.

    If the column already exists, return as-is. Otherwise merge on
    sra_accession with libraries_metadata.
    """
    if 'sequencing_lab' in df.columns:
        return df

    meta = load_sample_metadata()
    merge_cols = ['sra_accession', 'sequencing_lab']
    merge_cols = [c for c in merge_cols if c in meta.columns]
    if 'sra_accession' in df.columns and 'sra_accession' in meta.columns:
        df = df.merge(
            meta[['sra_accession', 'sequencing_lab']].drop_duplicates(),
            on='sra_accession', how='left',
        )
    return df


def plot_casper_cumulative_on_axis(ax, site_data):
    """Plot CASPER cumulative reads on the given axis."""
    df = site_data.copy()
    df['state'] = df['site_name'].apply(_state_from_site_name)
    if df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=FONT_SIZE_LARGE)
        return
    daily_by_state = df.groupby(['date', 'state'])['total_read_pairs'].sum().reset_index()
    pivot = daily_by_state.pivot(index='date', columns='state', values='total_read_pairs')
    pivot = pivot.fillna(0)
    pivot_cumsum = pivot.cumsum()
    state_totals = pivot_cumsum.iloc[-1].sort_values(ascending=False)
    pivot_cumsum = pivot_cumsum[state_totals.index]
    state_abbrev_labels = [STATE_ABBREVIATIONS.get(state, state) for state in pivot_cumsum.columns]
    ax.stackplot(pivot_cumsum.index, *[pivot_cumsum[col] for col in pivot_cumsum.columns],
                 labels=state_abbrev_labels,
                 colors=[get_state_color(state) for state in pivot_cumsum.columns],
                 alpha=0.8, edgecolor='none', linewidth=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: '0' if x == 0 else f'{x/1e9:.0f}B' if x >= 1e9 else f'{x/1e6:.0f}M'
    ))
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left',
             fontsize=FONT_SIZE_BASE, frameon=False, title='State',
             title_fontsize=FONT_SIZE_BASE)


def plot_sra_panel_on_axis(ax, timeline_data, by_collection_date=False, category='untargeted'):
    """Plot SRA cumulative data on a single axis."""
    if category == 'all':
        total_cat = 'wastewater_metagenome'
        total_label = 'All wastewater sequencing'
    else:
        total_cat = 'wastewater_shotgun_metagenomic'
        total_label = 'Untargeted wastewater sequencing'
    pivot_data = {}
    for cat in [total_cat, 'casper_PRJNA1247874', 'casper_rothman_PRJNA1198001']:
        cat_data = timeline_data[timeline_data['category'] == cat].copy()
        if not cat_data.empty:
            cat_data = cat_data.sort_values('date')
            cat_data['cumulative_mbases'] = cat_data['mbases'].cumsum()
            pivot_data[cat] = cat_data[['date', 'cumulative_mbases']].set_index('date')
    total_data = pivot_data.get(total_cat)
    casper_data = pivot_data.get('casper_PRJNA1247874')
    rothman_data = pivot_data.get('casper_rothman_PRJNA1198001')
    casper_label = 'CASPER'
    if total_data is None or total_data.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=FONT_SIZE_LARGE)
        ax.set_title(total_label, fontsize=FONT_SIZE_LARGE, weight='normal')
        return
    other_color = OTHER_COLOR
    casper_color = CASPER_COLOR
    all_dates = total_data.index
    if casper_data is not None and not casper_data.empty:
        casper_aligned = casper_data.reindex(all_dates)
        casper_aligned = casper_aligned.ffill().fillna(0)
        if rothman_data is not None and not rothman_data.empty:
            rothman_aligned = rothman_data.reindex(all_dates)
            rothman_aligned = rothman_aligned.ffill().fillna(0)
            combined_casper = casper_aligned['cumulative_mbases'] + rothman_aligned['cumulative_mbases']
        else:
            combined_casper = casper_aligned['cumulative_mbases']
        other_data = total_data['cumulative_mbases'] - combined_casper
        other_data = other_data.clip(lower=0)
        ax.fill_between(all_dates, 0, other_data,
                       label='Other', color=other_color, alpha=0.8)
        casper_start_date = casper_data.index.min()
        if rothman_data is not None and not rothman_data.empty:
            rothman_start_date = rothman_data.index.min()
            combined_start_date = min(casper_start_date, rothman_start_date)
        else:
            combined_start_date = casper_start_date
        casper_dates = all_dates[all_dates >= combined_start_date]
        casper_other = other_data[all_dates >= combined_start_date]
        casper_total = total_data['cumulative_mbases'][all_dates >= combined_start_date]
        ax.fill_between(casper_dates, casper_other, casper_total,
                       label=casper_label, color=casper_color, alpha=0.8)
        final_total = total_data['cumulative_mbases'].iloc[-1]
        final_combined_casper = combined_casper.iloc[-1]
        casper_pct = 100 * final_combined_casper / final_total if final_total > 0 else 0
        final_other = final_total - final_combined_casper
        bracket_bottom = final_other
        bracket_top = final_total
        bracket_mid = (bracket_bottom + bracket_top) / 2
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        bracket_x = 0.99
        tick_len = 0.02
        line_color = 'black'
        lw = 1.5
        ax.plot([bracket_x, bracket_x], [bracket_bottom, bracket_top],
               color=line_color, linewidth=lw, clip_on=False, zorder=10, transform=trans)
        ax.plot([bracket_x - tick_len, bracket_x], [bracket_bottom, bracket_bottom],
               color=line_color, linewidth=lw, clip_on=False, zorder=10, transform=trans)
        ax.plot([bracket_x - tick_len, bracket_x], [bracket_top, bracket_top],
               color=line_color, linewidth=lw, clip_on=False, zorder=10, transform=trans)
        ax.text(bracket_x + 0.02, bracket_mid, f'{casper_pct:.0f}%',
               fontsize=FONT_SIZE_BASE, ha='left', va='center',
               fontweight='bold', transform=trans, zorder=10)
    else:
        ax.fill_between(all_dates, 0, total_data['cumulative_mbases'],
                       label=total_label, color=other_color, alpha=0.8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: '0' if x == 0 else f'{x/1e6:.0f}'
    ))
    ax.set_title(total_label, fontsize=FONT_SIZE_LARGE, weight='normal')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left',
             fontsize=FONT_SIZE_LARGE, frameon=False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_xlim(left=pd.Timestamp('2013-01-01'))
    ax.set_ylim(bottom=0)


def plot_lit_review_scatter(ax, lit_review_path, y_column, y_label,
                            y_scale_divisor=1, annotate_casper=False,
                            annotation_offsets=None):
    """Plot scatter of study-level sequencing metrics from lit review data."""
    df = pd.read_csv(lit_review_path, sep='\t').copy()
    df['date'] = pd.to_datetime(df['date_paper_published'])
    df['y_val'] = pd.to_numeric(df[y_column], errors='coerce') / y_scale_divisor
    df = df.dropna(subset=['date', 'y_val'])
    casper_mask = df['bioproject'].isin(CASPER_BIOPROJECTS)
    other = df[~casper_mask]
    casper = df[casper_mask]
    ax.scatter(other['date'], other['y_val'],
               c=OTHER_COLOR, s=SCATTER_SIZE_LARGE, alpha=0.8,
               edgecolors='none', linewidths=0, zorder=5,
               label='Other')
    ax.scatter(casper['date'], casper['y_val'],
               c=CASPER_COLOR, s=SCATTER_SIZE_LARGE, alpha=0.9,
               edgecolors='none', linewidths=0, zorder=10,
               label='CASPER')
    if annotate_casper:
        default_config = {
            'PRJNA1198001': ((-35, -12), 'top'),
            'PRJNA1247874': ((-35, 8), 'bottom'),
        }
        if annotation_offsets:
            default_config.update(annotation_offsets)
        label_names = {
            'PRJNA1198001': 'Grimm et al.',
            'PRJNA1247874': 'Justen et al.',
        }
        for _, row in casper.iterrows():
            bp = row['bioproject']
            if bp in label_names and bp in default_config:
                offset, va = default_config[bp]
                ax.annotate(label_names[bp],
                            xy=(row['date'], row['y_val']),
                            xytext=offset, textcoords='offset points',
                            fontsize=FONT_SIZE_BASE - 1, color='black',
                            ha='center', va=va, fontstyle='italic')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'$10^{{{int(np.log10(x))}}}$' if x > 0 else '0'
    ))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              loc='upper left', fontsize=FONT_SIZE_LARGE, frameon=False,
              handletextpad=0.05, markerscale=1.5,
              bbox_to_anchor=(-0.02, 1.0))


def _plot_timeline_panel(ax, locations, df, location_stats,
                         date_range=None, show_xlabel=True, show_bottom_spine=True):
    """Plot sampling timeline on left panel.

    Args:
        ax: Matplotlib axis
        locations: List of site names to plot
        df: DataFrame with site data
        location_stats: DataFrame with location statistics
        date_range: Tuple of (min_date, max_date, date_range_days)
        show_xlabel: If True, show x-axis label
        show_bottom_spine: If True, show bottom spine
    """
    for idx, site_name in enumerate(locations):
        loc_data = df[df['site_name'] == site_name].sort_values('date')
        color = get_location_color(site_name)

        y_vals = [idx] * len(loc_data)
        ax.scatter(loc_data['date'], y_vals,
                  alpha=0.8, s=SCATTER_SIZE_LARGE * 0.9,
                  color=color, zorder=3)

    ax.set_yticks(range(len(locations)))
    ax.set_yticklabels(locations, fontsize=FONT_SIZE_LARGE + 2)

    y_padding = 0.5
    ax.set_ylim(len(locations) - 1 + y_padding, -y_padding)

    if date_range:
        date_min_padded, date_max_padded, date_range_days = date_range
    else:
        date_min = df['date'].min()
        date_max = df['date'].max()
        date_range_days = (date_max - date_min).days
        x_padding = pd.Timedelta(days=7)
        date_min_padded = date_min - x_padding
        date_max_padded = date_max + x_padding

    ax.set_xlim(date_min_padded, date_max_padded)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not show_bottom_spine:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(labelbottom=False, bottom=False)

    if show_xlabel:
        ax.set_xlabel('Date', fontsize=FONT_SIZE_LARGE + 2)
    else:
        ax.set_xlabel('')

    ax.tick_params(axis='both', labelsize=FONT_SIZE_LARGE + 2)

    # Add state separators
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


def _plot_boxplot_panel(ax, locations, df, show_xlabel=True):
    """Plot horizontal boxplots on right panel.

    Args:
        ax: Matplotlib axis
        locations: List of site names to plot
        df: DataFrame with site data
        show_xlabel: If True, show x-axis label
    """
    data_to_plot = []
    positions = []
    colors_list = []

    for idx, site_name in enumerate(locations):
        loc_data = df[df['site_name'] == site_name]['total_read_pairs'].values
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
        loc_data = df[df['site_name'] == site_name]['total_read_pairs'].values
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

    if show_xlabel:
        ax.set_xlabel('Read pairs', fontsize=FONT_SIZE_LARGE + 2)
    else:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False, bottom=False)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: '0' if x == 0 else (f'{int(x/1e9)}B' if x/1e9 == int(x/1e9) else f'{x/1e9:.1f}B') if x >= 1e9 else f'{x/1e6:.0f}M' if x >= 1e6 else f'{x/1e3:.0f}K'
    ))
    ax.tick_params(axis='x', labelsize=FONT_SIZE_LARGE + 2)

    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not show_xlabel:
        ax.spines['bottom'].set_visible(False)


def compute_sample_site_stats_by_seq_lab(site_data):
    """Compute sample and site statistics grouped by sequencing lab.

    Args:
        site_data: DataFrame with site_name and sequencing_lab columns.

    Returns:
        dict with keys for each lab, each containing n_samples, n_sites,
        total_read_pairs, and site_list.
    """
    df = _get_sequencing_lab_column(site_data.copy())

    stats = {}
    for lab in df['sequencing_lab'].unique():
        lab_data = df[df['sequencing_lab'] == lab]
        stats[lab] = {
            'n_samples': len(lab_data),
            'n_sites': lab_data['site_name'].nunique(),
            'total_read_pairs': lab_data['total_read_pairs'].sum() if 'total_read_pairs' in lab_data.columns else 0,
            'site_list': sorted(lab_data['site_name'].unique().tolist())
        }

    return stats


def plot_sampling_timeline_sra_combined(save_path=None):
    """Create combined 2-row figure with sampling timeline, read counts, and SRA comparison.

    Top row (a, b): Sample collection timeline and read count distributions
    Bottom row (c, d, e): CASPER cumulative reads, SRA comparison, median depth with cost

    Args:
        save_path: Optional path to save figure.

    Returns:
        fig: Matplotlib figure.
    """
    # Load data
    site_data = load_all_site_data()
    site_data = _get_sequencing_lab_column(site_data)
    sra_timeline_collection = load_sra_timeline(by_collection_date=True)

    # Paths
    lit_review_path = _SUPP_DIR / 'lit_review_table.tsv'

    # =========================================================================
    # Prepare site data for top row panels
    # =========================================================================

    df = site_data.copy()
    df['state'] = df['site_name'].apply(_state_from_site_name)

    # Get unique (site_name, sequencing_lab) pairs so multi-lab sites appear in both sections
    location_stats = df.groupby(['site_name', 'sequencing_lab']).agg({
        'date': ['min', 'max', 'count'],
        'state': 'first',
        'total_read_pairs': 'median'
    }).reset_index()
    location_stats.columns = ['site_name', 'sequencing_lab', 'start_date', 'end_date', 'n_samples',
                               'state', 'median_reads']

    location_stats = sort_locations_by_state_and_name(location_stats)

    # Split into MU and SB sites
    mu_locations = location_stats[location_stats['sequencing_lab'] == 'MU']['site_name'].tolist()
    sb_locations = location_stats[location_stats['sequencing_lab'] != 'MU']['site_name'].tolist()

    n_mu = len(mu_locations)
    n_sb = len(sb_locations)

    # =========================================================================
    # Create figure with complex gridspec layout
    # =========================================================================

    top_row_height = (n_mu + n_sb) * 0.38
    bottom_row_height = 3.8 * (18 / 14)

    fig = plt.figure(figsize=(18, top_row_height + bottom_row_height + 2.5))

    gs_main = fig.add_gridspec(2, 1, height_ratios=[top_row_height, bottom_row_height],
                                hspace=0.38)

    gs_top = gs_main[0].subgridspec(2, 2, height_ratios=[n_mu, n_sb],
                                     width_ratios=[2, 1], hspace=0.10, wspace=0.05)

    gs_bottom = gs_main[1].subgridspec(1, 3, wspace=0.25)

    ax_timeline_mu = fig.add_subplot(gs_top[0, 0])
    ax_boxplot_mu = fig.add_subplot(gs_top[0, 1])
    ax_timeline_sb = fig.add_subplot(gs_top[1, 0])
    ax_boxplot_sb = fig.add_subplot(gs_top[1, 1])

    ax_casper_reads = fig.add_subplot(gs_bottom[0])
    ax_sra_all = fig.add_subplot(gs_bottom[1])
    ax_median_bp = fig.add_subplot(gs_bottom[2])

    # =========================================================================
    # Top row (a, b): Timeline and read counts
    # =========================================================================

    date_min_padded = pd.Timestamp('2024-01-01') - pd.Timedelta(days=14)
    date_max_padded = pd.Timestamp('2026-01-01') + pd.Timedelta(days=14)
    date_range = (date_max_padded - date_min_padded).days

    # Filter data by sequencing lab for each section
    mu_df = df[df['sequencing_lab'] == 'MU']
    sb_df = df[df['sequencing_lab'] != 'MU']

    _plot_timeline_panel(ax_timeline_mu, mu_locations, mu_df, location_stats,
                        date_range=(date_min_padded, date_max_padded, date_range),
                        show_xlabel=False, show_bottom_spine=False)
    _plot_boxplot_panel(ax_boxplot_mu, mu_locations, mu_df, show_xlabel=False)

    _plot_timeline_panel(ax_timeline_sb, sb_locations, sb_df, location_stats,
                        date_range=(date_min_padded, date_max_padded, date_range),
                        show_xlabel=False, show_bottom_spine=True)
    _plot_boxplot_panel(ax_boxplot_sb, sb_locations, sb_df, show_xlabel=True)

    # Add title boxes with gray backgrounds
    from matplotlib.patches import Rectangle

    bbox_timeline_mu = ax_timeline_mu.get_position()
    bbox_boxplot_mu = ax_boxplot_mu.get_position()
    bbox_timeline_sb = ax_timeline_sb.get_position()
    bbox_boxplot_sb = ax_boxplot_sb.get_position()

    title_height = 0.022

    # MU row
    box_bottom_mu = bbox_timeline_mu.y1

    rect_mu_timeline = Rectangle((bbox_timeline_mu.x0, box_bottom_mu), bbox_timeline_mu.width, title_height,
                                 transform=fig.transFigure,
                                 facecolor='#CCCCCC', edgecolor='none',
                                 zorder=1, alpha=0.5)
    fig.patches.append(rect_mu_timeline)

    rect_mu_boxplot = Rectangle((bbox_boxplot_mu.x0, box_bottom_mu), bbox_boxplot_mu.width, title_height,
                               transform=fig.transFigure,
                               facecolor='#CCCCCC', edgecolor='none',
                               zorder=1, alpha=0.5)
    fig.patches.append(rect_mu_boxplot)

    title_y_mu = box_bottom_mu + title_height / 2
    title_x_mu_timeline = bbox_timeline_mu.x0 + bbox_timeline_mu.width / 2
    title_x_mu_boxplot = bbox_boxplot_mu.x0 + bbox_boxplot_mu.width / 2

    fig.text(title_x_mu_timeline, title_y_mu, 'MU-sequenced',
            fontsize=FONT_SIZE_LARGE + 2, weight='normal',
            ha='center', va='center', zorder=10)
    fig.text(title_x_mu_boxplot, title_y_mu, 'MU-sequenced',
            fontsize=FONT_SIZE_LARGE + 2, weight='normal',
            ha='center', va='center', zorder=10)

    # SB row
    other_bbox = ax_timeline_mu.get_position()
    gap_size = other_bbox.y0 - bbox_timeline_sb.y1
    title_height_sb = gap_size
    box_bottom_sb = bbox_timeline_sb.y1

    rect_sb_timeline = Rectangle((bbox_timeline_sb.x0, box_bottom_sb), bbox_timeline_sb.width, title_height_sb,
                                  transform=fig.transFigure,
                                  facecolor='#CCCCCC', edgecolor='none',
                                  zorder=1, alpha=0.5)
    fig.patches.append(rect_sb_timeline)

    rect_sb_boxplot = Rectangle((bbox_boxplot_sb.x0, box_bottom_sb), bbox_boxplot_sb.width, title_height_sb,
                                transform=fig.transFigure,
                                facecolor='#CCCCCC', edgecolor='none',
                                zorder=1, alpha=0.5)
    fig.patches.append(rect_sb_boxplot)

    title_y_sb = box_bottom_sb + title_height_sb / 2
    title_x_sb_timeline = bbox_timeline_sb.x0 + bbox_timeline_sb.width / 2
    title_x_sb_boxplot = bbox_boxplot_sb.x0 + bbox_boxplot_sb.width / 2

    fig.text(title_x_sb_timeline, title_y_sb, 'SB-sequenced',
            fontsize=FONT_SIZE_LARGE + 2, weight='normal',
            ha='center', va='center', zorder=10)
    fig.text(title_x_sb_boxplot, title_y_sb, 'SB-sequenced',
            fontsize=FONT_SIZE_LARGE + 2, weight='normal',
            ha='center', va='center', zorder=10)

    # =========================================================================
    # Bottom row (c): CASPER cumulative read pairs
    # =========================================================================

    plot_casper_cumulative_on_axis(ax_casper_reads, site_data)
    ax_casper_reads.set_ylabel('Cumulative read pairs', fontsize=FONT_SIZE_LARGE + 3)
    ax_casper_reads.set_xlabel('')
    ax_casper_reads.tick_params(axis='both', labelsize=FONT_SIZE_LARGE + 3)
    ax_casper_reads.spines['top'].set_visible(False)
    ax_casper_reads.spines['right'].set_visible(False)
    ax_casper_reads.set_title('CASPER sequencing', fontsize=FONT_SIZE_LARGE + 3, weight='normal', pad=24)

    legend = ax_casper_reads.get_legend()
    if legend is not None:
        legend.set_bbox_to_anchor((0.0, 1.08))
        legend._loc = 2
        for text in legend.get_texts():
            text.set_fontsize(FONT_SIZE_LARGE + 1)
        legend.set_title('State')
        legend.get_title().set_fontsize(FONT_SIZE_LARGE + 1)

    ax_casper_reads.set_yticks([0, 0.25e12, 0.5e12, 0.75e12, 1e12])
    ax_casper_reads.set_yticklabels(['0', '', '500B', '', '1T'])

    # =========================================================================
    # Bottom row (d): All wastewater sequencing
    # =========================================================================

    plot_sra_panel_on_axis(ax_sra_all, sra_timeline_collection,
                           by_collection_date=True, category='all')
    ax_sra_all.set_xlabel('')
    ax_sra_all.set_ylabel('Cumulative bases (Tb)', fontsize=FONT_SIZE_LARGE + 3)
    ax_sra_all.tick_params(axis='both', labelsize=FONT_SIZE_LARGE + 3)
    ax_sra_all.spines['top'].set_visible(False)
    ax_sra_all.spines['right'].set_visible(False)
    ax_sra_all.set_title(ax_sra_all.get_title(), fontsize=FONT_SIZE_LARGE + 3, weight='normal', pad=24)

    mid_legend = ax_sra_all.get_legend()
    if mid_legend:
        for text in mid_legend.get_texts():
            text.set_fontsize(FONT_SIZE_LARGE + 1)
        mid_legend.handletextpad = 0.4

    for child in ax_sra_all.texts:
        child.set_fontsize(FONT_SIZE_LARGE + 3)

    ax_sra_all.set_yticks([0, 100e6, 200e6, 300e6, 400e6, 500e6, 600e6])
    ax_sra_all.set_yticklabels(['0', '', '200', '', '400', '', '600'])

    xlim_mid_left = pd.Timestamp('2011-01-01')
    xlim_mid_right = pd.Timestamp('2026-06-01')
    ax_sra_all.set_xlim(left=xlim_mid_left, right=xlim_mid_right)
    ax_sra_all.xaxis.set_major_locator(mdates.YearLocator(3))
    ax_sra_all.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_sra_all.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # =========================================================================
    # Bottom row (e): Median sequencing depth by study + cost overlay
    # =========================================================================

    # Use median_gb_untargeted (public column name) converted to bp for axis scale
    _median_col = 'median_gb_untargeted'
    _gb_to_bp = 1e9  # convert Gb to bases to match internal axis scale
    plot_lit_review_scatter(ax_median_bp, lit_review_path,
                           y_column=_median_col,
                           y_label='Median bases per sample',
                           y_scale_divisor=1.0 / _gb_to_bp,
                           annotate_casper=False)

    # Remove Grimm (PRJNA1198001) from CASPER scatter points
    _lit_df = pd.read_csv(lit_review_path, sep='\t').copy()
    _lit_df['date'] = pd.to_datetime(_lit_df['date_paper_published'])
    _lit_df['y_val'] = pd.to_numeric(_lit_df[_median_col], errors='coerce') * _gb_to_bp
    _casper_df = _lit_df[_lit_df['bioproject'].isin(CASPER_BIOPROJECTS)].dropna(subset=['date', 'y_val'])
    _justen = _casper_df[_casper_df['bioproject'] == 'PRJNA1247874']
    if len(ax_median_bp.collections) >= 2:
        casper_coll = ax_median_bp.collections[1]
        casper_coll.set_offsets(np.column_stack([
            mdates.date2num(_justen['date'].values),
            _justen['y_val'].values
        ]))
    for coll in ax_median_bp.collections:
        coll.set_sizes([coll.get_sizes()[0] * 2.5])

    existing_handles, existing_labels = ax_median_bp.get_legend_handles_labels()
    for h in existing_handles:
        if hasattr(h, 'set_sizes'):
            h.set_sizes([80])

    ax_median_bp.set_ylabel('Bases', fontsize=FONT_SIZE_LARGE + 3, labelpad=-2)
    ax_median_bp.set_xlabel('')
    ax_median_bp.set_title('Sequencing depth and cost', fontsize=FONT_SIZE_LARGE + 3, weight='normal', pad=28)
    ax_median_bp.tick_params(axis='both', labelsize=FONT_SIZE_LARGE + 3)
    ax_median_bp.spines['top'].set_visible(False)
    ax_median_bp.spines['right'].set_visible(False)
    ax_median_bp.set_ylim(top=ax_median_bp.get_ylim()[1] * 3)

    ax_median_bp.set_xlim(left=xlim_mid_left, right=pd.Timestamp('2027-01-01'))
    ax_median_bp.xaxis.set_major_locator(mdates.YearLocator(3))
    ax_median_bp.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_median_bp.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ---- Cost overlay (bases_per_1k mode) ----
    cost_path = _SUPP_DIR / 'sequencing_cost_table.tsv'
    cost_df = pd.read_csv(cost_path, sep='\t')

    cost_df['parsed_date'] = pd.to_datetime(cost_df['Date'], format='%b %Y')
    # Parse 'Cost per Mb' column: values like '$522.71', '$0.006'
    def parse_cost(s):
        s = str(s).strip().replace('$', '').replace(',', '')
        return float(s)
    cost_df['cost_per_mb'] = cost_df['Cost per Mb'].apply(parse_cost)

    cost_filtered = cost_df[cost_df['parsed_date'] >= xlim_mid_left].sort_values('parsed_date')

    y_values = (100.0 / cost_filtered['cost_per_mb'].values) * 1e6  # bases per $100

    cost_filtered = cost_filtered.copy()
    cost_filtered['is_nhgri'] = cost_filtered['Source'].str.strip() == 'NHGRI'
    cost_filtered['is_sb'] = cost_filtered['Source'].str.strip() == 'estimated'
    cost_filtered['y_values'] = y_values

    cost_color = '#666666'
    nhgri_mask = cost_filtered['is_nhgri']
    sb_mask = cost_filtered['is_sb']
    nhgri_data = cost_filtered[nhgri_mask]
    sb_data = cost_filtered[sb_mask]

    ax_median_bp.plot(nhgri_data['parsed_date'], nhgri_data['y_values'],
                     color=cost_color, linewidth=1.5, linestyle='-', zorder=1)
    if len(nhgri_data) > 0 and len(sb_data) > 0:
        bridge_dates = [nhgri_data['parsed_date'].iloc[-1], sb_data['parsed_date'].iloc[0]]
        bridge_y = [nhgri_data['y_values'].iloc[-1], sb_data['y_values'].iloc[0]]
        ax_median_bp.plot(bridge_dates, bridge_y,
                         color=cost_color, linewidth=1.5, linestyle='--', zorder=1)
    ax_median_bp.plot(sb_data['parsed_date'], sb_data['y_values'],
                     color=cost_color, linewidth=1.5, linestyle='--', zorder=1)

    scatter_labels = [(l if l != 'Other' else 'Other studies') for l in existing_labels[::-1]]
    all_handles = existing_handles[::-1]
    all_labels = scatter_labels
    leg = ax_median_bp.legend(handles=all_handles, labels=all_labels,
                              fontsize=FONT_SIZE_LARGE + 1, frameon=False,
                              loc='upper left', bbox_to_anchor=(0.06, 1.0),
                              handlelength=0.8, handletextpad=0.4,
                              labelspacing=0.6,
                              markerscale=1.5)
    ax_median_bp.annotate('Median sample depth', xy=(0.06, 1.005),
                          xycoords='axes fraction', fontsize=FONT_SIZE_LARGE + 1,
                          ha='left', va='center')
    ax_median_bp.annotate('Bases per $100', xy=(0.5, 0.52),
                          xycoords='axes fraction', fontsize=FONT_SIZE_LARGE + 1,
                          ha='center', va='bottom', rotation=14,
                          color='black')

    def y1_formatter(x, pos):
        if x >= 1e12:
            val = x / 1e12
            if val == int(val):
                return f'{int(val)} Tb'
            return f'{val:.1f} Tb'
        elif x >= 1e9:
            val = x / 1e9
            if val == int(val):
                return f'{int(val)} Gb'
            return f'{val:.1f} Gb'
        elif x >= 1e6:
            val = x / 1e6
            if val == int(val):
                return f'{int(val)} Mb'
            return f'{val:.1f} Mb'
        return f'{x:.0f}'
    ax_median_bp.yaxis.set_major_formatter(mticker.FuncFormatter(y1_formatter))

    # =========================================================================
    # Layout and panel labels
    # =========================================================================

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    bbox_timeline_sb = ax_timeline_sb.get_position()
    bbox_boxplot_sb = ax_boxplot_sb.get_position()

    left_edge = 0.0
    right_edge = bbox_boxplot_sb.x1

    total_width = right_edge - left_edge

    pos_c = ax_casper_reads.get_position()
    pos_d = ax_sra_all.get_position()
    pos_e = ax_median_bp.get_position()

    gap_left = 0.065
    gap_right = 0.105
    panel_width = (total_width - gap_left - gap_right) / 3

    ax_casper_reads.set_position([left_edge, pos_c.y0, panel_width, pos_c.height])
    ax_sra_all.set_position([left_edge + panel_width + gap_left, pos_d.y0, panel_width, pos_d.height])
    ax_median_bp.set_position([left_edge + 2 * panel_width + gap_left + gap_right, pos_e.y0, panel_width, pos_e.height])

    # Panel labels a-e
    bbox_timeline = ax_timeline_mu.get_position()
    bbox_boxplot = ax_boxplot_mu.get_position()
    fig.text(bbox_timeline.x0 - 0.02, bbox_timeline.y1 + 0.045, 'a',
            fontsize=FONT_SIZE_LARGE + 8, fontweight='bold',
            ha='right', va='bottom')
    fig.text(bbox_boxplot.x0 - 0.01, bbox_boxplot.y1 + 0.045, 'b',
            fontsize=FONT_SIZE_LARGE + 8, fontweight='bold',
            ha='right', va='bottom')

    panel_labels_bottom = ['c', 'd', 'e']
    panel_axes_bottom = [ax_casper_reads, ax_sra_all, ax_median_bp]
    for ax, label in zip(panel_axes_bottom, panel_labels_bottom):
        bbox = ax.get_position()
        fig.text(bbox.x0 - 0.03, bbox.y1 + 0.03, label,
                fontsize=FONT_SIZE_LARGE + 8, fontweight='bold',
                ha='right', va='bottom')

    if save_path:
        save_figure(fig, save_path)

    return fig
