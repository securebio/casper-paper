#!/usr/bin/env python3
"""
Combined visualization: VV family composition + pathogen abundance panel.

Creates a tall 2-section figure:
- Top section (a, b): VV family composition bars and VV fraction boxplots
- Bottom section (c): 4x4 pathogen abundance time series
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Import plotting config
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *
from plot_config import _state_from_site_name

# Import data loading functions
from data_loaders import (
    load_all_vv_family_data,
    get_top_families,
    get_family_colors,
    load_all_site_data,
    load_all_relative_abundance,
    calculate_vv_fraction_per_library,
)

# Import panel functions from combined script
from plot_vv_pathogen_combined import (
    aggregate_vv_family_composition_for_site,
    _plot_bars_on_axis,
    _plot_boxplot_panel,
    print_vv_fraction_statistics,
    plot_pathogen_panel_on_axis,
    add_seasonal_shading,
    PATHOGENS,
    N_COLS,
    N_ROWS,
    DEFAULT_HIGHLIGHT_SITES,
    MULTI_LAB_SITES,
)

# Manuscript font sizes (increased for publication readability)
FONT_MS = 24
FONT_MS_PANEL = 38


def plot_vv_pathogen_combined(save_path=None, log_scale=True,
                               normalize_by_site_median=False):
    """
    Create combined figure with VV composition (top) and pathogen panel (bottom).

    Args:
        save_path: Optional path to save figure
        log_scale: If True, use symlog scale for pathogen y-axis
        normalize_by_site_median: If True, normalize pathogen values by site median

    Returns:
        fig: Matplotlib figure
    """
    # Load data
    vv_data = load_all_vv_family_data()
    virus_data = load_all_site_data()
    mj_data = load_all_relative_abundance()

    # =========================================================================
    # Prepare VV data (top section)
    # =========================================================================

    n_families = 9
    top_families = get_top_families(vv_data, n=n_families)

    # Merge VV data with virus data
    vv_per_library = vv_data.groupby(['site_name', 'sra_accession', 'name']).agg({
        'clade_counts': 'sum'
    }).reset_index()

    df_vv = vv_per_library.merge(
        virus_data[['site_name', 'sra_accession', 'total_read_pairs', 'sequencing_lab']],
        on=['site_name', 'sra_accession'],
        how='inner'
    )

    df_vv['state'] = df_vv['site_name'].apply(_state_from_site_name)

    # Multi-lab sites (e.g. Boston DITP) appear in both MU and SB lists
    site_labs = df_vv.groupby(['site_name', 'sequencing_lab']).size().reset_index()[['site_name', 'sequencing_lab']]
    site_labs['state'] = site_labs['site_name'].apply(_state_from_site_name)
    site_labs = sort_locations_by_state_and_name(site_labs)

    mu_locations = site_labs[site_labs['sequencing_lab'] == 'MU']['site_name'].tolist()
    sb_locations = site_labs[site_labs['sequencing_lab'] == 'SB']['site_name'].tolist()

    n_mu = len(mu_locations)
    n_sb = len(sb_locations)

    # =========================================================================
    # Prepare pathogen data (bottom section)
    # =========================================================================

    df_pathogen = mj_data.copy()
    df_pathogen['state'] = df_pathogen['site_name'].apply(_state_from_site_name)
    if 'sequencing_lab' not in df_pathogen.columns:
        from data_loaders import load_sample_metadata
        meta = load_sample_metadata()
        df_pathogen = df_pathogen.merge(
            meta[['sra_accession', 'sequencing_lab']].drop_duplicates(),
            on='sra_accession', how='left'
        )
    df_pathogen = filter_timeseries_data(df_pathogen)

    pathogen_location_order = df_pathogen.groupby('site_name').agg({
        'state': 'first',
        'date': 'min'
    }).reset_index()
    pathogen_location_order = sort_locations_by_state_and_name(pathogen_location_order)
    pathogen_locations = pathogen_location_order['site_name'].tolist()

    date_min = df_pathogen['date'].min()
    date_max = df_pathogen['date'].max()
    date_range_days = (date_max - date_min).days
    buffer_days = 7

    # =========================================================================
    # Create figure with complex gridspec layout
    # =========================================================================

    # Calculate heights
    top_section_height = (n_mu + n_sb) * 0.42
    bottom_section_height = 18  # Increased for 4x4 pathogen panel

    fig = plt.figure(figsize=(20, top_section_height + bottom_section_height + 3))

    # Main gridspec: 2 rows (top for VV, bottom for pathogens)
    gs_main = fig.add_gridspec(2, 1, height_ratios=[top_section_height, bottom_section_height],
                                hspace=0.25)

    # Top section: nested gridspec for VV bars (2 rows for MU/SB) x (bars, boxplots)
    gs_top = gs_main[0].subgridspec(2, 2, height_ratios=[n_mu, n_sb],
                                     width_ratios=[2, 1], hspace=0.12, wspace=0.05)

    # Bottom section: 4x4 grid for pathogens
    gs_bottom = gs_main[1].subgridspec(N_ROWS, N_COLS, hspace=0.25, wspace=0.15)

    # Create top section axes
    ax_bars_mu = fig.add_subplot(gs_top[0, 0])
    ax_boxplot_mu = fig.add_subplot(gs_top[0, 1])
    ax_bars_sb = fig.add_subplot(gs_top[1, 0])
    ax_boxplot_sb = fig.add_subplot(gs_top[1, 1])

    # Create bottom section axes (4x4 grid)
    axes_pathogen = []
    for row in range(N_ROWS):
        row_axes = []
        for col in range(N_COLS):
            ax = fig.add_subplot(gs_bottom[row, col])
            row_axes.append(ax)
        axes_pathogen.append(row_axes)
    axes_pathogen = np.array(axes_pathogen)

    # =========================================================================
    # Top section (a, b): VV family composition
    # =========================================================================

    # Filter data by sequencing lab for each section
    mu_vv = df_vv[df_vv['sequencing_lab'] == 'MU']
    sb_vv = df_vv[df_vv['sequencing_lab'] == 'SB']

    _plot_bars_on_axis(ax_bars_mu, mu_locations, mu_vv, site_labs, 'VV Family Composition',
                       top_families)
    _plot_bars_on_axis(ax_bars_sb, sb_locations, sb_vv, site_labs, 'VV Family Composition',
                       top_families)

    # Calculate VV fractions for boxplots, split by lab
    vv_fractions = calculate_vv_fraction_per_library(vv_data, virus_data)
    if 'sequencing_lab' not in vv_fractions.columns:
        from data_loaders import load_sample_metadata
        _meta = load_sample_metadata()
        vv_fractions = vv_fractions.merge(
            _meta[['sra_accession', 'sequencing_lab']].drop_duplicates(),
            on='sra_accession', how='left')
    mu_vv_frac = vv_fractions[vv_fractions['sequencing_lab'] == 'MU']
    sb_vv_frac = vv_fractions[vv_fractions['sequencing_lab'] == 'SB']

    _plot_boxplot_panel(ax_boxplot_mu, mu_locations, mu_vv_frac, show_xlabel=False)
    _plot_boxplot_panel(ax_boxplot_sb, sb_locations, sb_vv_frac, show_xlabel=True)

    # Override helper font sizes for manuscript
    for ax in [ax_boxplot_mu, ax_boxplot_sb]:
        ax.tick_params(axis='x', labelsize=FONT_MS + 2)
    ax_boxplot_sb.set_xlabel('VV fraction', fontsize=FONT_MS + 2)

    # Format bar axes
    for ax in [ax_bars_mu, ax_bars_sb]:
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: '0' if x == 0 else '1' if x == 1 else f'{x:.1f}'))
        ax.tick_params(axis='both', labelsize=FONT_MS + 2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax_bars_mu.set_xlabel('')
    ax_bars_mu.tick_params(labelbottom=False, bottom=False)
    ax_bars_mu.spines['bottom'].set_visible(False)
    ax_bars_sb.set_xlabel('Relative abundance', fontsize=FONT_MS + 2)

    # Add VV family legend above left column
    families_with_other = top_families + ['Other']
    family_colors = get_family_colors(families_with_other)

    bbox_bars_mu = ax_bars_mu.get_position()
    left_column_center_x = bbox_bars_mu.x0 + bbox_bars_mu.width / 2

    ordered_families = top_families[:9] + ['Other', '', '']
    handles_vv = []
    legend_labels_vv = []
    for fam in ordered_families:
        if fam == '':
            handles_vv.append(plt.Rectangle((0, 0), 1, 1, fc='none', edgecolor='none'))
            legend_labels_vv.append('')
        else:
            handles_vv.append(plt.Rectangle((0, 0), 1, 1, fc=family_colors[fam]))
            legend_labels_vv.append(fam)

    # Position legend above top section (left-aligned)
    fig.legend(handles_vv, legend_labels_vv, loc='upper left', ncol=4,
              fontsize=FONT_MS, frameon=False, handletextpad=0.4,
              bbox_to_anchor=(-0.05, 0.955))

    # Add title boxes for MU/SB rows
    from matplotlib.patches import Rectangle

    bbox_bars_mu = ax_bars_mu.get_position()
    bbox_boxplot_mu = ax_boxplot_mu.get_position()
    bbox_bars_sb = ax_bars_sb.get_position()
    bbox_boxplot_sb = ax_boxplot_sb.get_position()

    title_height = 0.018

    # MU row boxes
    box_bottom_mu = bbox_bars_mu.y1
    rect_mu_bars = Rectangle((bbox_bars_mu.x0, box_bottom_mu), bbox_bars_mu.width, title_height,
                             transform=fig.transFigure, facecolor='#CCCCCC', edgecolor='none',
                             zorder=1, alpha=0.5)
    fig.patches.append(rect_mu_bars)
    rect_mu_boxplot = Rectangle((bbox_boxplot_mu.x0, box_bottom_mu), bbox_boxplot_mu.width, title_height,
                               transform=fig.transFigure, facecolor='#CCCCCC', edgecolor='none',
                               zorder=1, alpha=0.5)
    fig.patches.append(rect_mu_boxplot)

    title_y_mu = box_bottom_mu + title_height / 2
    fig.text(bbox_bars_mu.x0 + bbox_bars_mu.width / 2, title_y_mu, 'MU-sequenced',
            fontsize=FONT_MS + 2, weight='normal', ha='center', va='center', zorder=10)
    fig.text(bbox_boxplot_mu.x0 + bbox_boxplot_mu.width / 2, title_y_mu, 'MU-sequenced',
            fontsize=FONT_MS + 2, weight='normal', ha='center', va='center', zorder=10)

    # SB row boxes
    gap_size = bbox_bars_mu.y0 - bbox_bars_sb.y1
    title_height_sb = gap_size
    box_bottom_sb = bbox_bars_sb.y1
    rect_sb_bars = Rectangle((bbox_bars_sb.x0, box_bottom_sb), bbox_bars_sb.width, title_height_sb,
                              transform=fig.transFigure, facecolor='#CCCCCC', edgecolor='none',
                              zorder=1, alpha=0.5)
    fig.patches.append(rect_sb_bars)
    rect_sb_boxplot = Rectangle((bbox_boxplot_sb.x0, box_bottom_sb), bbox_boxplot_sb.width, title_height_sb,
                                transform=fig.transFigure, facecolor='#CCCCCC', edgecolor='none',
                                zorder=1, alpha=0.5)
    fig.patches.append(rect_sb_boxplot)

    title_y_sb = box_bottom_sb + title_height_sb / 2
    fig.text(bbox_bars_sb.x0 + bbox_bars_sb.width / 2, title_y_sb, 'SB-sequenced',
            fontsize=FONT_MS + 2, weight='normal', ha='center', va='center', zorder=10)
    fig.text(bbox_boxplot_sb.x0 + bbox_boxplot_sb.width / 2, title_y_sb, 'SB-sequenced',
            fontsize=FONT_MS + 2, weight='normal', ha='center', va='center', zorder=10)

    # =========================================================================
    # Bottom section (c): Pathogen abundance panel
    # =========================================================================

    highlight_sites = DEFAULT_HIGHLIGHT_SITES

    # Plot each pathogen
    for idx, (taxid, label) in enumerate(PATHOGENS):
        row = idx // N_COLS
        col = idx % N_COLS
        ax = axes_pathogen[row, col]

        # Add seasonal shading
        add_seasonal_shading(ax, date_min, date_max)

        plot_pathogen_panel_on_axis(
            ax, df_pathogen, taxid, label, pathogen_locations, highlight_sites,
            log_scale=log_scale,
            normalize_by_site_median=normalize_by_site_median
        )

    # Set ylim for pathogen panels
    # For multi-lab sites, split by sequencing_lab to match plot behavior
    # (MU highlighted, SB always gray)
    has_lab_col = 'sequencing_lab' in df_pathogen.columns
    if log_scale:
        # Shared ylim per row for log scale
        for row_idx in range(N_ROWS):
            row_taxids = [PATHOGENS[row_idx * N_COLS + col][0] for col in range(N_COLS)]
            row_data = df_pathogen[df_pathogen['taxid'].isin(row_taxids)]
            if not row_data.empty:
                y_values_for_ylim = []
                for site_name in pathogen_locations:
                    site_data = row_data[row_data['site_name'] == site_name]
                    is_multi = site_name in MULTI_LAB_SITES and has_lab_col
                    if is_multi:
                        for lab in ['MU', 'SB']:
                            lab_data = site_data[site_data['sequencing_lab'] == lab].sort_values('date').reset_index(drop=True)
                            if len(lab_data) == 0:
                                continue
                            is_highlighted = (lab == 'MU' and site_name in highlight_sites)
                            if is_highlighted:
                                y_values_for_ylim.extend(lab_data['ra_clade_pmmov_norm'].tolist())
                            elif len(lab_data) >= 2:
                                smoothed_df = calculate_mmwr_smoothed_trend(lab_data, 'date', 'ra_clade_pmmov_norm')
                                if not smoothed_df.empty:
                                    y_values_for_ylim.extend(smoothed_df['smoothed_values'].tolist())
                    else:
                        loc_data = site_data.sort_values('date').reset_index(drop=True)
                        if len(loc_data) == 0:
                            continue
                        if site_name in highlight_sites:
                            y_values_for_ylim.extend(loc_data['ra_clade_pmmov_norm'].tolist())
                        elif len(loc_data) >= 2:
                            smoothed_df = calculate_mmwr_smoothed_trend(loc_data, 'date', 'ra_clade_pmmov_norm')
                            if not smoothed_df.empty:
                                y_values_for_ylim.extend(smoothed_df['smoothed_values'].tolist())
                if y_values_for_ylim:
                    y_min = min(y_values_for_ylim)
                    y_max = max(y_values_for_ylim)
                    y_max_padded = y_max * 2
                    for col in range(N_COLS):
                        axes_pathogen[row_idx, col].set_ylim(y_min, y_max_padded)
    elif normalize_by_site_median:
        # Per-subplot ylim for site-median normalized (linear scale)
        # Scale so the highest colored (highlighted) line peak is at 75% of the y-height
        from matplotlib.ticker import MaxNLocator
        for idx, (taxid, label) in enumerate(PATHOGENS):
            row = idx // N_COLS
            col = idx % N_COLS
            ax = axes_pathogen[row, col]

            taxid_data = df_pathogen[df_pathogen['taxid'] == taxid].copy()
            if taxid_data.empty:
                continue

            # Compute fold-change for ylim
            site_medians = taxid_data[taxid_data['ra_clade_pmmov_norm'] > 0].groupby('site_name')['ra_clade_pmmov_norm'].median()

            def normalize_value(row):
                site_name = row['site_name']
                if site_name in site_medians and site_medians[site_name] > 0:
                    return row['ra_clade_pmmov_norm'] / site_medians[site_name]
                return np.nan

            taxid_data['plot_value'] = taxid_data.apply(normalize_value, axis=1)

            highlight_smoothed_max = 0
            for site_name in pathogen_locations:
                if site_name not in highlight_sites:
                    continue
                site_data = taxid_data[taxid_data['site_name'] == site_name]
                is_multi = site_name in MULTI_LAB_SITES and has_lab_col
                # For multi-lab highlighted sites, use only MU data (matches plot behavior)
                if is_multi:
                    site_data = site_data[site_data['sequencing_lab'] == 'MU']
                loc_data = site_data.sort_values('date').reset_index(drop=True)
                if len(loc_data) == 0 or loc_data['plot_value'].isna().all():
                    continue
                if len(loc_data) >= 2:
                    smoothed_df = calculate_mmwr_smoothed_trend(loc_data, 'date', 'plot_value')
                    if not smoothed_df.empty:
                        peak = smoothed_df['smoothed_values'].max()
                        if peak > highlight_smoothed_max:
                            highlight_smoothed_max = peak

            if highlight_smoothed_max > 0:
                # Set ylim so the highest colored line peak sits at 75% of the y-height
                y_max = highlight_smoothed_max / 0.75
                ax.set_ylim(0, y_max)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    # Format x-axis - set xlim on ALL axes (since we're not using sharex)
    xlim_left = date_min - pd.Timedelta(days=buffer_days)
    xlim_right = date_max + pd.Timedelta(days=buffer_days)

    for row in range(N_ROWS):
        for col in range(N_COLS):
            ax = axes_pathogen[row, col]
            ax.set_xlim(xlim_left, xlim_right)

            # Increase font sizes to match top figure
            ax.tick_params(axis='x', labelsize=FONT_MS + 2)
            ax.tick_params(axis='y', labelsize=FONT_MS + 4)
            ax.set_title(ax.get_title(), fontsize=FONT_MS + 2)

            # Reduce y-tick frequency for log/symlog scale
            from matplotlib.ticker import SymmetricalLogLocator, MaxNLocator
            if log_scale:
                # symlog uses linthresh=1e-5; show only major decades
                locator = SymmetricalLogLocator(base=10, linthresh=1e-5)
                locator.set_params(numticks=5)
                ax.yaxis.set_major_locator(locator)
                ax.yaxis.set_minor_locator(plt.NullLocator())
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

            # Only bottom row shows tick labels
            if row == N_ROWS - 1:
                # Use longer date range to get fewer ticks (semi-annual)
                format_date_axis(ax, date_range_days=max(date_range_days, 851))
            else:
                ax.tick_params(labelbottom=False)

    # Add pathogen legend
    handles, labels = axes_pathogen[0, 0].get_legend_handles_labels()

    mu_items = []
    sb_items = []
    for h, label in zip(handles, labels):
        if 'Miami' in label or 'FL' in label or 'South Florida' in label:
            sb_items.append((h, label))
        elif 'Boston' in label or 'DITP' in label:
            # Boston DITP is sequenced by both MU and SB -- keep "(MU)" to distinguish
            mu_items.append((h, f'{label} (MU)'))
        else:
            mu_items.append((h, label))

    mu_items.sort(key=lambda x: x[1])

    if len(mu_items) >= 4 and len(sb_items) >= 1:
        ordered_items = [mu_items[0], mu_items[1], sb_items[0], mu_items[2], mu_items[3]]
    else:
        ordered_items = mu_items + sb_items

    ordered_handles = [item[0] for item in ordered_items]
    ordered_labels = [item[1] for item in ordered_items]

    other_sites_handle = Line2D([0], [0], color='#CCCCCC', linewidth=LINE_WIDTH * 1.5, alpha=0.6)
    ordered_handles.append(other_sites_handle)
    ordered_labels.append('Other sites')

    # Position pathogen legend below VV section
    pathogen_top_y = axes_pathogen[0, 0].get_position().y1
    legend_top_y = pathogen_top_y + 0.05
    fig.legend(ordered_handles, ordered_labels, loc='upper left', ncol=3,
              bbox_to_anchor=(-0.08, legend_top_y), frameon=False, fontsize=FONT_MS,
              handlelength=1.5, columnspacing=1.0, handletextpad=0.5)

    # Add seasonal shading legend (top aligned with site legend, moved inward)
    winter_patch = mpatches.Patch(color='#E6F2FF', alpha=0.5, label='Winter (Dec-Feb)')
    summer_patch = mpatches.Patch(color='#FFF3B0', alpha=0.5, label='Summer (Jun-Aug)')
    fig.legend(handles=[winter_patch, summer_patch], loc='upper right', ncol=1,
              bbox_to_anchor=(0.92, legend_top_y), frameon=False, fontsize=FONT_MS)

    # =========================================================================
    # Panel labels and layout adjustments
    # =========================================================================

    plt.tight_layout(rect=[0.02, 0.02, 1, 0.99])

    # Adjust bottom section (pathogen panel) to extend leftward only
    # Get positions from top row panels
    bbox_bars_mu = ax_bars_mu.get_position()
    bbox_boxplot_mu = ax_boxplot_mu.get_position()

    # Left edge extends to figure edge, right edge aligned with top row
    left_edge = 0.0
    right_edge = bbox_boxplot_mu.x1  # Align with top row right edge

    # Calculate total width available for pathogen panels
    total_width = right_edge - left_edge

    # Get current pathogen panel dimensions for reference
    current_pos = axes_pathogen[0, 0].get_position()
    panel_height = current_pos.height

    # Calculate panel widths with gaps
    # Extend panels as far left as possible to minimize whitespace
    effective_left = -0.07
    effective_width = right_edge - effective_left

    # Calculate spacing between columns (wider for log scale tick labels)
    hspace_fraction = 0.06  # gap between columns as fraction of effective width
    total_hspace = hspace_fraction * (N_COLS - 1)
    panel_width = (effective_width - total_hspace * effective_width) / N_COLS
    col_gap = hspace_fraction * effective_width

    # Get vertical positions from current layout (preserve row spacing)
    row_positions = []
    for row in range(N_ROWS):
        row_positions.append(axes_pathogen[row, 0].get_position().y0)

    # Reposition all pathogen axes
    for row in range(N_ROWS):
        for col in range(N_COLS):
            ax = axes_pathogen[row, col]
            new_x = effective_left + col * (panel_width + col_gap)
            new_y = row_positions[row]
            ax.set_position([new_x, new_y, panel_width, panel_height])

    # Add pathogen section y-label AFTER repositioning (so we get correct positions)
    pathogen_center_y = (axes_pathogen[0, 0].get_position().y1 + axes_pathogen[-1, 0].get_position().y0) / 2
    # Position label to the left of the leftmost panel
    leftmost_panel_x = axes_pathogen[0, 0].get_position().x0
    ylabel_x = leftmost_panel_x - 0.08
    if normalize_by_site_median:
        fig.text(ylabel_x, pathogen_center_y, 'Fold change from site median\n(pathogen / PMMoV) / median',
                 ha='center', va='center', rotation='vertical', fontsize=FONT_MS + 2)
    else:
        fig.text(ylabel_x, pathogen_center_y, 'Wastewater sequencing signal\n(pathogen / PMMoV abundance)',
                 ha='center', va='center', rotation='vertical', fontsize=FONT_MS + 2)

    # Refresh positions after tight_layout
    bbox_bars_mu = ax_bars_mu.get_position()
    bbox_boxplot_mu = ax_boxplot_mu.get_position()

    # Bottom section label (c)
    bbox_pathogen_topleft = axes_pathogen[0, 0].get_position()
    panel_c_label_x = bbox_pathogen_topleft.x0 - 0.02

    # Top section labels (a, b) -- "a" aligned on x with "c"
    panel_label_y = bbox_bars_mu.y1 + 0.025
    fig.text(panel_c_label_x, panel_label_y, 'a',
            fontsize=FONT_MS_PANEL, fontweight='bold', ha='right', va='bottom')
    fig.text(bbox_boxplot_mu.x0 - 0.01, panel_label_y, 'b',
            fontsize=FONT_MS_PANEL, fontweight='bold', ha='right', va='bottom')

    # Bottom section label (c)
    fig.text(panel_c_label_x, bbox_pathogen_topleft.y1 + 0.025, 'c',
            fontsize=FONT_MS_PANEL, fontweight='bold', ha='right', va='bottom')

    if save_path:
        save_figure(fig, save_path)

    return fig


