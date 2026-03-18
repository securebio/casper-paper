#!/usr/bin/env python3
"""
Supplementary figure showing Boston clinical testing volume and positivity rate
alongside MGS vs clinical comparison.

Creates a 3-row x 3-column figure:
- Row 1: MU-sequenced MGS vs clinical time series
- Row 2: SB-sequenced MGS vs clinical time series
- Row 3: Stacked bar plots showing test volume (gray=negative, colored=positive)

Columns: SARS-CoV-2, Influenza A, RSV

Clinical data source:
Clinical respiratory virus testing data from the Massachusetts General Hospital
(MGH) Clinical Diagnostic Microbiology Laboratory. Data include aggregated weekly
test counts from MGH and affiliated Mass General Brigham hospitals serving the
greater Boston metropolitan area, which overlaps geographically with the sewershed
served by Boston's Deer Island Treatment Plant.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import plotting config
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *

# Import data loading functions from centralized module
from data_loaders import (
    load_mgs_pathogen_data,
    load_clinical_positives_tests_data,
    load_sample_metadata,
    NWSS_PATHOGEN_MAP
)

# Import correlation formatting function
from plot_mgs_nwss_panel import format_r_with_stars

# Import ticker for custom formatting
from matplotlib.ticker import FuncFormatter, ScalarFormatter


class CleanScalarFormatter(ScalarFormatter):
    """ScalarFormatter that shows '0' instead of '0.00' for zero values."""

    def __call__(self, x, pos=None):
        if x == 0:
            return '0'
        return super().__call__(x, pos)


def clean_zero_formatter(x, pos):
    """Format tick labels to show '0' instead of '0.0' or '0.00'."""
    if x == 0:
        return '0'
    # For other values, use default formatting
    if abs(x) >= 1000:
        return f'{x:.0f}'
    elif abs(x) >= 1:
        return f'{x:.0f}' if x == int(x) else f'{x:.1f}'.rstrip('0').rstrip('.')
    else:
        # Small values - format without trailing zeros
        formatted = f'{x:.2g}'
        return formatted

# MMWR smoothing is now imported from plot_config via wildcard import above

# Pathogens to plot (columns)
PATHOGENS_TO_PLOT = ['sars-cov-2', 'influenza_a', 'rsv']

# Clinical data column names for each pathogen
CLINICAL_PATHOGEN_MAP = {
    'sars-cov-2': 'SARS-CoV-2',
    'influenza_a': 'Influenza A',
    'rsv': 'RSV',
}

# Sites with temporal outliers to skip (first N samples) — MU only
SITE_SKIP_FIRST_N = {
    'Boston DITP North, MA': 1,
    'Boston DITP South, MA': 1,
}
SITE_SKIP_FIRST_N_LAB = 'MU'  # Only skip for this sequencing lab

# Clinical comparison sites (Boston DITP North and South)
CLINICAL_SITES = ['Boston DITP North, MA', 'Boston DITP South, MA']

# Sequencing lab keys for each row
# Row 1: MU-sequenced, Row 2: SB-sequenced
SEQ_LABS = ['MU', 'SB']

# Display labels for sequencing labs
SEQ_LAB_LABELS = {
    'MU': 'MU',
    'SB': 'SB',
}

# Color for positive bars (red across all pathogens)
POSITIVE_BAR_COLOR = '#E41A1C'  # Red


def _get_lab_for_sra(metadata, sra_accession):
    """Look up sequencing lab for an SRA accession."""
    row = metadata[metadata['sra_accession'] == sra_accession]
    if not row.empty:
        return row['sequencing_lab'].iloc[0]
    return None


def add_seasonal_shading(ax, date_min, date_max, alpha=0.3):
    """Add faint blue shading for winter and yellow for summer."""
    start_year = date_min.year
    end_year = date_max.year + 1

    winter_color = '#E6F2FF'
    summer_color = '#FFF3B0'

    for year in range(start_year - 1, end_year + 1):
        # Winter: Dec-Feb
        winter_start = pd.Timestamp(f'{year}-12-01')
        winter_end = pd.Timestamp(f'{year + 1}-03-01')
        shade_start = max(winter_start, date_min)
        shade_end = min(winter_end, date_max)
        if shade_start < shade_end:
            ax.axvspan(shade_start, shade_end, color=winter_color, alpha=alpha, zorder=0)

        # Summer: Jun-Aug
        summer_start = pd.Timestamp(f'{year}-06-01')
        summer_end = pd.Timestamp(f'{year}-09-01')
        shade_start = max(summer_start, date_min)
        shade_end = min(summer_end, date_max)
        if shade_start < shade_end:
            ax.axvspan(shade_start, shade_end, color=summer_color, alpha=alpha, zorder=0)


def load_clinical_testing_data(drop_last_n_weeks=1):
    """
    Load clinical testing data with Tests, Positives, Negatives per pathogen
    per week from the public clinical_testing_data.csv.

    Args:
        drop_last_n_weeks: Number of weeks to drop from the end (default 1,
            as last week often has incomplete data)

    Returns DataFrame with columns:
        date, Target, Tests, Positives, Negatives
    """
    from data_loaders import _DATA_DIR

    path = _DATA_DIR / "clinical_testing_data.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])

    # Rename columns to match internal format expected by plotting code
    df = df.rename(columns={
        'pathogen': 'Target',
        'positives': 'Positives',
        'tests': 'Tests',
    })

    # Compute Negatives
    df['Negatives'] = df['Tests'] - df['Positives']

    # Drop last N weeks (often incomplete data)
    if drop_last_n_weeks > 0 and not df.empty:
        unique_dates = sorted(df['date'].unique())
        if len(unique_dates) > drop_last_n_weeks:
            dates_to_drop = unique_dates[-drop_last_n_weeks:]
            df = df[~df['date'].isin(dates_to_drop)]

    return df


def _filter_mgs_by_lab(mgs_data, metadata, seq_lab):
    """
    Filter MGS data to only include samples from a specific sequencing lab.

    Args:
        mgs_data: DataFrame with sra_accession column
        metadata: Sample metadata with sra_accession and sequencing_lab
        seq_lab: Sequencing lab to filter to (e.g. 'MU' or 'SB')

    Returns:
        Filtered DataFrame
    """
    lab_accessions = metadata[
        metadata['sequencing_lab'] == seq_lab
    ]['sra_accession'].unique()
    return mgs_data[mgs_data['sra_accession'].isin(lab_accessions)]


def plot_mgs_row(
    axes_row,
    mgs_data_dict,
    clinical_time_series,
    seq_lab,
    metadata,
    date_min,
    date_max,
    global_xlim_min,
    global_xlim_max,
    is_first_row=False,
    is_last_mgs_row=False
):
    """
    Plot a single row of MGS vs clinical time series.

    Args:
        axes_row: Row of axes (3 columns)
        mgs_data_dict: Dictionary of MGS data by pathogen
        clinical_time_series: Clinical time series data
        seq_lab: Sequencing lab key ('MU' or 'SB')
        metadata: Sample metadata DataFrame
        date_min: Minimum date for data filtering
        date_max: Maximum date for data filtering
        global_xlim_min: Global x-axis minimum (shared across all rows)
        global_xlim_max: Global x-axis maximum (shared across all rows)
        is_first_row: Whether this is the first row (for column titles)
        is_last_mgs_row: Whether this is the last MGS row (for hiding x-tick labels)

    Returns:
        Dictionary with site_colors for legend
    """
    from scipy.stats import spearmanr

    n_cols = len(PATHOGENS_TO_PLOT)
    row_info = {'site_colors': {}}

    # Plot each pathogen column
    for col_idx, pathogen_key in enumerate(PATHOGENS_TO_PLOT):
        pathogen_config = NWSS_PATHOGEN_MAP[pathogen_key]
        pathogen_label = pathogen_config['label']
        taxids = pathogen_config['taxids']
        clinical_col_name = CLINICAL_PATHOGEN_MAP.get(pathogen_key)

        ax1 = axes_row[col_idx]
        ax2 = ax1.twinx()

        mgs_data = mgs_data_dict.get(pathogen_key)
        site_smoothed_data = {}
        mgs_max_y = 0

        # Plot MGS for both Boston sites
        for site_name in CLINICAL_SITES:
            if mgs_data is not None:
                # Filter to this site and sequencing lab
                site_mgs = mgs_data[
                    (mgs_data['site_name'] == site_name) &
                    (mgs_data['taxid'].isin(taxids))
                ].copy()
                site_mgs = _filter_mgs_by_lab(site_mgs, metadata, seq_lab)

                if not site_mgs.empty:
                    site_mgs = site_mgs.sort_values('date')

                    # Smooth on full data to avoid edge effects
                    mgs_smoothed = calculate_mmwr_smoothed_trend(
                        site_mgs, 'date', 'ra_clade_pmmov_norm'
                    )

                    color = get_location_color(site_name)
                    row_info['site_colors'][site_name] = color

                    if not mgs_smoothed.empty:
                        # For display: clip to date range and skip first sample (MU only)
                        smoothed_dates = pd.to_datetime(mgs_smoothed['date'])
                        display_min = pd.Timestamp(date_min)
                        if site_name in SITE_SKIP_FIRST_N and seq_lab == SITE_SKIP_FIRST_N_LAB:
                            first_date = site_mgs['date'].min()
                            display_min = max(display_min, pd.Timestamp(first_date) + pd.Timedelta(days=1))
                        mgs_display = mgs_smoothed[
                            (smoothed_dates >= display_min) &
                            (smoothed_dates <= pd.Timestamp(date_max))
                        ]

                        site_label = 'DITPN' if 'North' in site_name else 'DITPS'
                        if not mgs_display.empty:
                            ax1.plot(mgs_display['date'], mgs_display['smoothed_values'],
                                    color=color, linewidth=LINE_WIDTH * 1.5,
                                    alpha=LINE_ALPHA, label=site_label, zorder=4)

                            site_max = mgs_display['smoothed_values'].max()
                            if site_max > mgs_max_y:
                                mgs_max_y = site_max

                        # Store full smoothed data for correlation (inner merge handles overlap)
                        site_smoothed_data[site_name] = {
                            'smoothed': mgs_smoothed,
                            'color': color,
                            'label': site_label
                        }

        # Plot clinical positives (smooth full series, clip for display)
        clinical_filtered = clinical_time_series[['date', clinical_col_name]].copy()
        clinical_filtered = clinical_filtered[
            (clinical_filtered[clinical_col_name].notna())
        ]
        clinical_color = '#666666'
        if not clinical_filtered.empty:
            clinical_smoothed = calculate_mmwr_smoothed_trend(
                clinical_filtered, 'date', clinical_col_name
            )

            if not clinical_smoothed.empty:
                # Clip to display range for plotting
                clinical_smoothed_dates = pd.to_datetime(clinical_smoothed['date'])
                clinical_display = clinical_smoothed[
                    (clinical_smoothed_dates >= pd.Timestamp(global_xlim_min)) &
                    (clinical_smoothed_dates <= pd.Timestamp(global_xlim_max))
                ]

                if not clinical_display.empty:
                    ax2.plot(clinical_display['date'], clinical_display['smoothed_values'],
                            color=clinical_color, linewidth=LINE_WIDTH * 1.5,
                            linestyle=':', alpha=LINE_ALPHA, label='Clinical', zorder=2)

                # Calculate correlation using full smoothed data (inner merge handles overlap)
                if site_smoothed_data:
                    corr_items = []
                    for site_id, site_info in site_smoothed_data.items():
                        mgs_sm = site_info['smoothed']

                        merged = pd.merge(
                            mgs_sm[['mmwr_year', 'mmwr_week', 'smoothed_values']].rename(
                                columns={'smoothed_values': 'mgs_value'}),
                            clinical_smoothed[['mmwr_year', 'mmwr_week', 'smoothed_values']].rename(
                                columns={'smoothed_values': 'clinical_value'}),
                            on=['mmwr_year', 'mmwr_week'],
                            how='inner'
                        )
                        if len(merged) >= 3:
                            r, p = spearmanr(merged['mgs_value'], merged['clinical_value'])
                            corr_items.append((format_r_with_stars(r, p), site_info['color']))

                    # Display correlations
                    # SARS-CoV-2: top right; RSV: top right; others: top left
                    if corr_items:
                        if pathogen_key == 'sars-cov-2':
                            y_pos = 0.96
                            use_right = True
                        elif pathogen_key == 'rsv':
                            y_pos = 0.96
                            use_right = True
                        else:
                            y_pos = 0.96
                            use_right = False
                        for i, (text, color) in enumerate(corr_items):
                            y = y_pos - i * 0.12
                            if use_right:
                                marker_x = 0.64
                                ax1.plot(marker_x, y, 'o', transform=ax1.transAxes,
                                        color=color, markersize=6, zorder=10)
                                ax1.text(marker_x + 0.03, y, text, transform=ax1.transAxes,
                                        ha='left', va='center', fontsize=FONT_SIZE_BASE,
                                        color='black', zorder=10)
                            else:
                                ax1.plot(0.03, y, 'o', transform=ax1.transAxes,
                                        color=color, markersize=6, zorder=10)
                                ax1.text(0.06, y, text, transform=ax1.transAxes,
                                        ha='left', va='center', fontsize=FONT_SIZE_BASE,
                                        color='black', zorder=10)

                # Set y limits (use display-range data for clinical axis)
                if mgs_max_y > 0:
                    ax1.set_ylim(bottom=-mgs_max_y * 0.05, top=mgs_max_y * 1.1)
                if not clinical_display.empty:
                    clinical_max = clinical_display['smoothed_values'].max() * 1.1
                else:
                    clinical_max = clinical_smoothed['smoothed_values'].max() * 1.1
                ax2.set_ylim(bottom=-clinical_max * 0.05, top=clinical_max)

        # Formatting - use global xlim for consistent x-axis across all rows
        buffer = pd.Timedelta(days=7)

        # Add seasonal shading using global xlim
        add_seasonal_shading(ax1, global_xlim_min, global_xlim_max)

        ax1.set_xlim(global_xlim_min - buffer, global_xlim_max + buffer)
        ax1.set_xticklabels([])  # Always hide x-tick labels for MGS rows
        ax1.grid(True, alpha=0.3, zorder=1)
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.tick_params(axis='both', labelsize=FONT_SIZE_BASE)
        ax2.tick_params(axis='both', labelsize=FONT_SIZE_BASE)

        # Column titles (only for first row)
        if is_first_row:
            ax1.set_title(pathogen_label, fontsize=FONT_SIZE_LARGE + 2, pad=10)

        # Scientific notation for MGS axis with clean zeros (1e notation)
        formatter = CleanScalarFormatter(useMathText=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        ax1.yaxis.set_major_formatter(formatter)
        ax1.yaxis.get_offset_text().set_fontsize(FONT_SIZE_BASE)

        # Clean zero formatting for clinical axis (ax2)
        ax2.yaxis.set_major_formatter(FuncFormatter(clean_zero_formatter))

        # Y-axis labels
        if col_idx == 0:
            seq_lab_label = SEQ_LAB_LABELS.get(seq_lab, seq_lab)
            ax1.set_ylabel(f'{seq_lab_label}-sequenced\nWW-MGS signal', fontsize=FONT_SIZE_LARGE)
        if col_idx == n_cols - 1:
            ax2.set_ylabel('Positive tests', fontsize=FONT_SIZE_LARGE)

    return row_info


def plot_clinical_testing_supplement(
    mgs_data_dict,
    clinical_time_series,
    clinical_testing_data,
    metadata,
    save_path=None,
):
    """
    Create 3x3 supplementary figure with MU-sequenced MGS, SB-sequenced MGS, and testing volume.

    Row 1: MU-sequenced MGS (colored lines) vs Clinical positives (gray dotted line)
    Row 2: SB-sequenced MGS (colored lines) vs Clinical positives (gray dotted line)
    Row 3: Stacked bars showing test volume (gray=negative, red=positive) with positivity rate on y2
    """
    import matplotlib.dates as mdates

    n_rows = 3
    n_cols = len(PATHOGENS_TO_PLOT)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10),
                             gridspec_kw={'height_ratios': [1.2, 1.2, 1], 'hspace': 0.18, 'wspace': 0.35})

    # Get overall date range from MGS data (both sequencing labs)
    # Apply skip logic so the date range reflects what's actually plotted
    all_mgs_dates = []
    for seq_lab in SEQ_LABS:
        for pathogen_key in PATHOGENS_TO_PLOT:
            mgs_data = mgs_data_dict.get(pathogen_key)
            if mgs_data is None:
                continue
            taxids = NWSS_PATHOGEN_MAP[pathogen_key]['taxids']
            for site_name in CLINICAL_SITES:
                site_mgs = mgs_data[
                    (mgs_data['site_name'] == site_name) &
                    (mgs_data['taxid'].isin(taxids))
                ].copy()
                site_mgs = _filter_mgs_by_lab(site_mgs, metadata, seq_lab)
                if not site_mgs.empty:
                    # Apply same skip logic used in plot_mgs_row (MU only)
                    if site_name in SITE_SKIP_FIRST_N and seq_lab == SITE_SKIP_FIRST_N_LAB:
                        n_skip = SITE_SKIP_FIRST_N[site_name]
                        site_mgs = site_mgs.sort_values('date')
                        unique_dates = site_mgs['date'].unique()
                        if len(unique_dates) > n_skip:
                            skip_dates = unique_dates[:n_skip]
                            site_mgs = site_mgs[~site_mgs['date'].isin(skip_dates)]
                    all_mgs_dates.extend(site_mgs['date'].tolist())

    if not all_mgs_dates:
        print("No MGS data found for Boston sites")
        return fig, axes

    # Global x-axis limits: use the overlap period between MGS and clinical data
    mgs_min = min(all_mgs_dates)
    mgs_max = max(all_mgs_dates)

    # Get the clinical data date range
    clinical_dates = []
    for pathogen_key in PATHOGENS_TO_PLOT:
        clinical_col_name = CLINICAL_PATHOGEN_MAP.get(pathogen_key)
        if clinical_col_name and clinical_col_name in clinical_time_series.columns:
            clinical_filtered = clinical_time_series[
                clinical_time_series[clinical_col_name].notna()
            ]
            if not clinical_filtered.empty:
                clinical_dates.extend(clinical_filtered['date'].tolist())

    if clinical_dates:
        clinical_min = min(clinical_dates)
        clinical_max = max(clinical_dates)
        # Use overlap: latest start, earliest end
        global_xlim_min = max(mgs_min, clinical_min)
        global_xlim_max = min(mgs_max, clinical_max)
    else:
        # Fallback to MGS range if no clinical data
        global_xlim_min = mgs_min
        global_xlim_max = mgs_max

    # For data filtering, use the overlap range
    date_min = global_xlim_min
    date_max = global_xlim_max

    buffer = pd.Timedelta(days=7)

    # ===== Row 1: MU-sequenced MGS vs Clinical =====
    mu_row_info = plot_mgs_row(
        axes[0, :], mgs_data_dict, clinical_time_series,
        seq_lab='MU', metadata=metadata,
        date_min=date_min, date_max=date_max,
        global_xlim_min=global_xlim_min, global_xlim_max=global_xlim_max,
        is_first_row=True, is_last_mgs_row=False
    )

    # ===== Row 2: SB-sequenced MGS vs Clinical =====
    sb_row_info = plot_mgs_row(
        axes[1, :], mgs_data_dict, clinical_time_series,
        seq_lab='SB', metadata=metadata,
        date_min=date_min, date_max=date_max,
        global_xlim_min=global_xlim_min, global_xlim_max=global_xlim_max,
        is_first_row=False, is_last_mgs_row=True
    )

    # ===== Row 3: Testing volume bars =====
    for col_idx, pathogen_key in enumerate(PATHOGENS_TO_PLOT):
        clinical_col_name = CLINICAL_PATHOGEN_MAP.get(pathogen_key)

        ax_bar = axes[2, col_idx]
        ax_bar2 = ax_bar.twinx()

        # Add seasonal shading using global xlim
        add_seasonal_shading(ax_bar, global_xlim_min, global_xlim_max)

        # Get testing data for this pathogen
        pathogen_testing = clinical_testing_data[
            (clinical_testing_data['Target'] == clinical_col_name) &
            (clinical_testing_data['date'] >= global_xlim_min) &
            (clinical_testing_data['date'] <= global_xlim_max)
        ].copy()

        if not pathogen_testing.empty:
            pathogen_testing = pathogen_testing.sort_values('date')
            bar_width = 5

            dates = pathogen_testing['date']
            negatives = pathogen_testing['Negatives'].fillna(0)
            positives = pathogen_testing['Positives'].fillna(0)
            tests = pathogen_testing['Tests'].fillna(0)

            ax_bar.bar(dates, negatives, width=bar_width, color='#CCCCCC',
                      label='Negative', zorder=2)
            ax_bar.bar(dates, positives, width=bar_width, bottom=negatives,
                      color=POSITIVE_BAR_COLOR, label='Positive', zorder=3)

            # Calculate positivity rate and apply MMWR smoothing
            pathogen_testing['positivity_rate'] = (positives / tests * 100).replace([np.inf, -np.inf], np.nan)
            positivity_smoothed = calculate_mmwr_smoothed_trend(
                pathogen_testing, 'date', 'positivity_rate'
            )
            if not positivity_smoothed.empty:
                ax_bar2.plot(positivity_smoothed['date'], positivity_smoothed['smoothed_values'],
                            color=POSITIVE_BAR_COLOR, linewidth=LINE_WIDTH * 1.5,
                            linestyle='-', alpha=0.7, zorder=4)

            # Set y limits
            max_total = (negatives + positives).max()
            ax_bar.set_ylim(0, max_total * 1.1)

            if not positivity_smoothed.empty:
                max_positivity = positivity_smoothed['smoothed_values'].max()
                if not np.isnan(max_positivity):
                    ax_bar2.set_ylim(0, min(max_positivity * 1.3, 100))
                else:
                    ax_bar2.set_ylim(0, 50)
            else:
                ax_bar2.set_ylim(0, 50)

        # Row 3 formatting - use global xlim for consistent x-axis
        ax_bar.set_xlim(global_xlim_min - buffer, global_xlim_max + buffer)
        ax_bar.grid(True, alpha=0.3, zorder=1, axis='y')
        ax_bar.spines['top'].set_visible(False)
        ax_bar2.spines['top'].set_visible(False)
        ax_bar.tick_params(axis='both', labelsize=FONT_SIZE_BASE)
        ax_bar2.tick_params(axis='both', labelsize=FONT_SIZE_BASE)

        # Clean zero formatting for bar chart axes
        ax_bar.yaxis.set_major_formatter(FuncFormatter(clean_zero_formatter))
        ax_bar2.yaxis.set_major_formatter(FuncFormatter(clean_zero_formatter))

        # X-axis date formatting
        ax_bar.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax_bar.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax_bar.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Y-axis labels
        if col_idx == 0:
            ax_bar.set_ylabel('Tests per week', fontsize=FONT_SIZE_LARGE)
        if col_idx == n_cols - 1:
            ax_bar2.set_ylabel('Positivity rate (%)', fontsize=FONT_SIZE_LARGE)

    # ===== Panel labels (a, b, c) =====
    panel_labels = ['a', 'b', 'c']
    for row_idx, label in enumerate(panel_labels):
        ax = axes[row_idx, 0]
        # Add panel label in top-left corner of first column
        ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                fontsize=FONT_SIZE_LARGE + 2, fontweight='bold',
                va='bottom', ha='right')

    # ===== Legend =====
    from matplotlib.lines import Line2D

    plt.tight_layout()
    plt.subplots_adjust(top=0.78, bottom=0.10, left=0.08)

    ax_top = axes[0, 0].get_position().y1
    legend_y = ax_top + 0.20

    # Column 1: CASPER wastewater sequencing (WW-MGS sites) - both MU and SB
    col1_handles = [
        Line2D([0], [0], color=get_location_color('Boston DITP North, MA'),
               linewidth=LINE_WIDTH * 1.5),
        Line2D([0], [0], color=get_location_color('Boston DITP South, MA'),
               linewidth=LINE_WIDTH * 1.5),
    ]
    col1_labels = [
        'Boston DITP North',
        'Boston DITP South',
    ]

    # Column 2: Clinical testing
    col2_handles = [
        Line2D([0], [0], color='#666666', linewidth=LINE_WIDTH * 1.5,
               linestyle=':'),
        Line2D([0], [0], color=POSITIVE_BAR_COLOR, linewidth=LINE_WIDTH * 1.5,
               linestyle='-', alpha=0.7),
        mpatches.Patch(color='#CCCCCC'),
        mpatches.Patch(color=POSITIVE_BAR_COLOR),
    ]
    col2_labels = [
        'Positive tests',
        'Positivity rate',
        'Negative tests',
        'Positive tests',
    ]

    # Column 3: Seasonal shading
    col3_handles = [
        mpatches.Patch(color='#E6F2FF', alpha=0.5),
        mpatches.Patch(color='#FFF3B0', alpha=0.5),
    ]
    col3_labels = ['Winter (Dec-Feb)', 'Summer (Jun-Aug)']

    title_offset = 0.005

    # Add title for Column 1 legend
    fig.text(0.08, legend_y + title_offset, 'CASPER wastewater sequencing',
             fontsize=FONT_SIZE_LARGE, weight='bold', ha='left', va='bottom')

    legend1 = fig.legend(col1_handles, col1_labels, loc='upper left',
                        bbox_to_anchor=(0.08, legend_y), frameon=False,
                        fontsize=FONT_SIZE_LARGE, ncol=1,
                        handletextpad=0.7, labelspacing=0.3)

    # Column 2 legend
    fig.text(0.38, legend_y + title_offset, 'MGH clinical testing (Boston, MA)',
             fontsize=FONT_SIZE_LARGE, weight='bold', ha='left', va='bottom')
    legend2 = fig.legend(col2_handles, col2_labels, loc='upper left',
                        bbox_to_anchor=(0.38, legend_y), frameon=False,
                        fontsize=FONT_SIZE_LARGE, ncol=1,
                        handletextpad=0.7, labelspacing=0.3)

    # Column 3 legend
    legend3 = fig.legend(col3_handles, col3_labels, loc='upper left',
                        bbox_to_anchor=(0.72, legend_y), frameon=False,
                        fontsize=FONT_SIZE_LARGE, ncol=1,
                        handletextpad=0.7, labelspacing=0.3)

    fig.add_artist(legend1)
    fig.add_artist(legend2)

    if save_path:
        save_figure(fig, save_path)

    return fig, axes
