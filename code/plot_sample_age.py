#!/usr/bin/env python3
"""
Plot sample age distribution (time from collection to sequencing results).

Shows histograms of sample age for MU-sequenced and SB-sequenced samples
separately, along with summary statistics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Import plotting config and data loaders
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *
from data_loaders import load_sample_age_data


def plot_sample_age_histogram(sample_age_data, save_path=None, figsize=(12, 5)):
    """
    Plot histograms of sample age by sequencing lab.

    Args:
        sample_age_data: DataFrame with sample_age and sequencing_lab columns
        save_path: Optional path to save figure
        figsize: Figure size tuple

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    df = sample_age_data.copy()

    # Filter to valid sample ages
    df = df[df['sample_age'].notna() & (df['sample_age'] >= 0)]

    # Separate by sequencing lab
    mu_data = df[df['sequencing_lab'] == 'MU']['sample_age']
    sb_data = df[df['sequencing_lab'].isin(['BCL', 'SB'])]['sample_age']

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Use 5-day bins, with tick labels every 10 days
    bin_width = 5
    tick_interval = 10
    max_regular_value = 60  # Values >60 go in overflow bin
    gap_width = 3

    # Regular bins from 0 to 60
    regular_bins = list(range(0, max_regular_value + 1, bin_width))

    # Overflow bin position (after gap)
    overflow_bin_start = max_regular_value + gap_width
    overflow_bin_end = overflow_bin_start + bin_width
    overflow_center = overflow_bin_start + bin_width / 2

    # Split data into regular (<=60) and overflow (>60)
    mu_regular = mu_data[mu_data <= max_regular_value]
    mu_overflow_count = (mu_data > max_regular_value).sum()
    sb_regular = sb_data[sb_data <= max_regular_value]
    sb_overflow_count = (sb_data > max_regular_value).sum()

    # Colors
    mu_color = '#1f77b4'
    sb_color = '#ff7f0e'

    # MU histogram
    ax_mu = axes[0]
    if len(mu_data) > 0:
        ax_mu.hist(mu_regular, bins=regular_bins, color=mu_color, alpha=0.7,
                   edgecolor='none')
        if mu_overflow_count > 0:
            ax_mu.bar(overflow_center, mu_overflow_count, width=bin_width,
                     color=mu_color, alpha=0.7, edgecolor='none')

        median_mu = mu_data.median()
        legend_text = f'n = {len(mu_data):,}\nMedian: {median_mu:.0f} days'
        ax_mu.text(0.97, 0.97, legend_text,
                   transform=ax_mu.transAxes,
                   fontsize=FONT_SIZE_BASE,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    ax_mu.set_ylabel('Samples', fontsize=FONT_SIZE_LARGE)
    ax_mu.tick_params(axis='both', labelsize=FONT_SIZE_BASE)
    ax_mu.spines['top'].set_visible(False)
    ax_mu.spines['right'].set_visible(False)
    ax_mu.grid(axis='y', alpha=0.3)

    tick_positions = list(range(0, max_regular_value, tick_interval)) + [overflow_center]
    tick_labels = [str(i) for i in range(0, max_regular_value, tick_interval)] + ['>60']
    ax_mu.set_xticks(tick_positions)
    ax_mu.set_xticklabels(tick_labels)

    # SB histogram
    ax_sb = axes[1]
    if len(sb_data) > 0:
        ax_sb.hist(sb_regular, bins=regular_bins, color=sb_color, alpha=0.7,
                   edgecolor='none')
        if sb_overflow_count > 0:
            ax_sb.bar(overflow_center, sb_overflow_count, width=bin_width,
                     color=sb_color, alpha=0.7, edgecolor='none')

        median_sb = sb_data.median()
        legend_text = f'n = {len(sb_data):,}\nMedian: {median_sb:.0f} days'
        ax_sb.text(0.97, 0.97, legend_text,
                   transform=ax_sb.transAxes,
                   fontsize=FONT_SIZE_BASE,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    ax_sb.tick_params(axis='both', labelsize=FONT_SIZE_BASE)
    ax_sb.spines['top'].set_visible(False)
    ax_sb.spines['right'].set_visible(False)
    ax_sb.grid(axis='y', alpha=0.3)

    ax_sb.set_xticks(tick_positions)
    ax_sb.set_xticklabels(tick_labels)

    # Set x-axis limits
    for ax in axes:
        ax.set_xlim(0, overflow_bin_end)

    plt.tight_layout()

    # Add gray title boxes after tight_layout
    bbox_mu = ax_mu.get_position()
    bbox_sb = ax_sb.get_position()

    title_height = 0.08

    rect_mu = Rectangle((bbox_mu.x0, bbox_mu.y1), bbox_mu.width, title_height,
                         transform=fig.transFigure,
                         facecolor='#CCCCCC', edgecolor='none',
                         zorder=1, alpha=0.5)
    fig.patches.append(rect_mu)

    rect_sb = Rectangle((bbox_sb.x0, bbox_sb.y1), bbox_sb.width, title_height,
                         transform=fig.transFigure,
                         facecolor='#CCCCCC', edgecolor='none',
                         zorder=1, alpha=0.5)
    fig.patches.append(rect_sb)

    title_y = bbox_mu.y1 + title_height / 2
    fig.text(bbox_mu.x0 + bbox_mu.width / 2, title_y, 'MU-sequenced',
             fontsize=FONT_SIZE_LARGE, weight='normal',
             ha='center', va='center', zorder=10)
    fig.text(bbox_sb.x0 + bbox_sb.width / 2, title_y, 'SB-sequenced',
             fontsize=FONT_SIZE_LARGE, weight='normal',
             ha='center', va='center', zorder=10)

    # Single centered x-axis label
    fig.text(0.5, 0.02, 'Sample collection to sequencing lag (days)',
             fontsize=FONT_SIZE_LARGE, ha='center', va='bottom')

    plt.subplots_adjust(bottom=0.15)

    if save_path:
        save_figure(fig, save_path)

    return fig, axes


def print_summary_stats(sample_age_data):
    """
    Print summary statistics for sample age by sequencing lab.

    Args:
        sample_age_data: DataFrame with sample_age and sequencing_lab columns
    """
    df = sample_age_data.copy()
    df = df[df['sample_age'].notna() & (df['sample_age'] >= 0)]

    all_ages = df['sample_age']
    overall_median = all_ages.median()
    overall_min = all_ages.min()
    overall_max = all_ages.max()

    print("Turnaround time stats:")
    print(f"  Overall: median {overall_median:.0f} days, range {overall_min:.0f}-{overall_max:.0f} (n={len(all_ages)})")

    for lab in ['MU', 'BCL', 'SB']:
        lab_data = df[df['sequencing_lab'] == lab]['sample_age']
        if len(lab_data) > 0:
            lab_label = 'SB' if lab == 'BCL' else lab
            print(f"  {lab_label}: median {lab_data.median():.0f} days, "
                  f"IQR {lab_data.quantile(0.25):.0f}-{lab_data.quantile(0.75):.0f}, "
                  f"range {lab_data.min():.0f}-{lab_data.max():.0f} (n={len(lab_data)})")

    nyc_mask = df['site_name'].str.contains('NYC', case=False, na=False)
    nyc_data = df[nyc_mask]['sample_age']
    non_nyc_mu_data = df[(df['sequencing_lab'] == 'MU') & ~nyc_mask]['sample_age']
    if len(nyc_data) > 0:
        print(f"  NYC: median {nyc_data.median():.0f} days, "
              f"range {nyc_data.min():.0f}-{nyc_data.max():.0f} (n={len(nyc_data)})")
    if len(non_nyc_mu_data) > 0:
        print(f"  MU (excl NYC): median {non_nyc_mu_data.median():.0f} days, "
              f"range {non_nyc_mu_data.min():.0f}-{non_nyc_mu_data.max():.0f} (n={len(non_nyc_mu_data)})")
