#!/usr/bin/env python3
"""
Supplemental figure: CASPER as fraction of untargeted SRA data + total sequencing effort by study.

Creates a 1x2 figure:
- Left (a): CASPER as fraction of untargeted wastewater sequencing on SRA
- Right (b): Total untargeted sequencing effort by study (scatter)

Figure S2 in the CASPER manuscript.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# Import plotting config
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *

# Import data loading functions
from data_loaders import load_sra_timeline, CASPER_BIOPROJECTS

# Paths
_CODE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CODE_DIR.parent
_SUPP_DIR = _REPO_ROOT / "tables"


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
        last_date = all_dates[-1]
        import matplotlib.dates as mdates
        last_date_num = mdates.date2num(last_date)
        tick_len_days = (mdates.date2num(all_dates[-1]) - mdates.date2num(all_dates[0])) * 0.02
        line_color = 'black'
        lw = 1.5
        ax.plot([last_date_num, last_date_num], [bracket_bottom, bracket_top],
               color=line_color, linewidth=lw, clip_on=False, zorder=10)
        ax.plot([last_date_num - tick_len_days, last_date_num], [bracket_bottom, bracket_bottom],
               color=line_color, linewidth=lw, clip_on=False, zorder=10)
        ax.plot([last_date_num - tick_len_days, last_date_num], [bracket_top, bracket_top],
               color=line_color, linewidth=lw, clip_on=False, zorder=10)
        ax.text(last_date_num + tick_len_days * 0.5, bracket_mid, f'{casper_pct:.0f}%',
               fontsize=FONT_SIZE_BASE, ha='left', va='center',
               fontweight='bold', zorder=10)
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
    df = pd.read_csv(lit_review_path, sep='\t')
    df = df.copy()
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


def plot_sra_untargeted_supplement(save_path=None):
    """Create 1x2 supplemental figure with untargeted SRA and total bp scatter.

    Args:
        save_path: Optional path to save figure.

    Returns:
        fig: Matplotlib figure.
    """
    sra_timeline_collection = load_sra_timeline(by_collection_date=True)

    lit_review_path = _SUPP_DIR / 'lit_review_table.tsv'

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8),
                              gridspec_kw={'wspace': 0.38})

    ax_sra_untargeted = axes[0]
    ax_total_bp = axes[1]

    xlim_left = pd.Timestamp('2011-01-01')
    xlim_right = pd.Timestamp('2027-01-01')

    # =========================================================================
    # Left (a): Untargeted wastewater sequencing on SRA
    # =========================================================================

    plot_sra_panel_on_axis(ax_sra_untargeted, sra_timeline_collection,
                           by_collection_date=True, category='untargeted')
    ax_sra_untargeted.set_xlabel('')
    ax_sra_untargeted.set_ylabel('Cumulative bases (Tb)', fontsize=FONT_SIZE_LARGE)
    ax_sra_untargeted.tick_params(axis='both', labelsize=FONT_SIZE_LARGE)
    ax_sra_untargeted.spines['top'].set_visible(False)
    ax_sra_untargeted.spines['right'].set_visible(False)
    ax_sra_untargeted.set_title('')

    ax_sra_untargeted.set_xlim(left=xlim_left, right=xlim_right)
    ax_sra_untargeted.xaxis.set_major_locator(mdates.YearLocator(3))
    ax_sra_untargeted.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_sra_untargeted.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # =========================================================================
    # Right (b): Total untargeted sequencing effort by study
    # =========================================================================

    plot_lit_review_scatter(ax_total_bp, lit_review_path,
                           y_column='total_gb_untargeted',
                           y_label='Total untargeted bases',
                           y_scale_divisor=1e-9,
                           annotate_casper=True,
                           annotation_offsets={
                               'PRJNA1198001': ((10, -8), 'top'),
                               'PRJNA1247874': ((-25, 6), 'bottom'),
                           })

    for coll in ax_total_bp.collections:
        coll.set_sizes([coll.get_sizes()[0] * 2.5])

    existing_handles, existing_labels = ax_total_bp.get_legend_handles_labels()
    for h in existing_handles:
        if hasattr(h, 'set_sizes'):
            h.set_sizes([80])

    all_handles = existing_handles[::-1]
    all_labels = [l if l != 'Other' else 'Other studies' for l in existing_labels[::-1]]
    ax_total_bp.legend(handles=all_handles, labels=all_labels,
                       fontsize=FONT_SIZE_BASE, frameon=False,
                       loc='upper left', bbox_to_anchor=(0.02, 1.07),
                       handletextpad=0.4, markerscale=1.5)

    ax_total_bp.set_ylabel('Total bases sequenced by study', fontsize=FONT_SIZE_LARGE)
    ax_total_bp.set_xlabel('')
    ax_total_bp.set_title('')
    ax_total_bp.tick_params(axis='both', labelsize=FONT_SIZE_LARGE)
    ax_total_bp.spines['top'].set_visible(False)
    ax_total_bp.spines['right'].set_visible(False)
    ax_total_bp.set_ylim(top=ax_total_bp.get_ylim()[1] * 3)

    def y1_formatter(x, pos):
        if x in (1e9, 1e11, 1e13):
            return ''
        exp = np.log10(x)
        if exp == int(exp):
            return f'$10^{{{int(exp)}}}$'
        return ''
    ax_total_bp.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=20))
    ax_total_bp.yaxis.set_major_formatter(mticker.FuncFormatter(y1_formatter))

    ax_total_bp.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    ax_total_bp.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_total_bp.tick_params(axis='y', which='minor', length=3, width=0.8)

    ax_total_bp.set_xlim(left=xlim_left, right=xlim_right)
    ax_total_bp.xaxis.set_major_locator(mdates.YearLocator(3))
    ax_total_bp.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_total_bp.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # =========================================================================
    # Panel labels and layout
    # =========================================================================

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    shift = 0.03
    pos0 = axes[0].get_position()
    axes[0].set_position([pos0.x0 - shift, pos0.y0, pos0.width, pos0.height])

    panel_labels = ['a', 'b']
    panel_axes = [ax_sra_untargeted, ax_total_bp]
    for ax, label in zip(panel_axes, panel_labels):
        bbox = ax.get_position()
        fig.text(bbox.x0 - 0.03, bbox.y1 + 0.09, label,
                fontsize=FONT_SIZE_LARGE, fontweight='bold',
                ha='right', va='bottom')

    if save_path:
        save_figure(fig, save_path)

    return fig
