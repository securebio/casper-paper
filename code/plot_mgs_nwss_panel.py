#!/usr/bin/env python3
"""
Polished panel figure comparing MGS pathogen abundance with NWSS PCR data
and clinical testing data.

Creates a 3-row (pathogens) x 5-column (4 sites + clinical) figure showing:
- Row 1: SARS-CoV-2
- Row 2: Influenza A
- Row 3: RSV

Site columns (1-4):
- Chicago (CHI-A), IL
- Columbia WWTP, MO
- Riverside WQCP, CA
- Boston DITP North, MA

Clinical column (5):
- Boston DITP MGS data compared with Massachusetts General Hospital (MGH)
  clinical testing data

Each site panel shows MGS abundance (left y-axis) alongside NWSS ddPCR
measurements (right y-axis), restricted to overlapping date ranges.
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
from plot_config import _state_from_site_name

# Import data loading functions from centralized module
from data_loaders import (
    load_nwss_data,
    load_mgs_nwss_matches,
    load_mgs_pathogen_data,
    load_clinical_positives_tests_data,
    filter_to_dominant_source,
    NWSS_PATHOGEN_MAP
)

# Sites to include (in order for columns)
SITES_TO_PLOT = [
    'Chicago (CHI-A), IL',
    'Columbia WWTP, MO',
    'Riverside WQCP, CA',
    'Boston DITP North, MA',
]

# Additional MGS sites to plot in the same column (for Boston DITP)
SITES_SAME_COLUMN = {
    'Boston DITP North, MA': ['Boston DITP South, MA'],
}

# Sites with temporal outliers to skip (first N samples) — visualization only
SITE_SKIP_FIRST_N = {
    'Boston DITP North, MA': 1,
    'Boston DITP South, MA': 1,
}

# Sites where only MU-sequenced data should be used in NWSS/clinical comparisons
MU_ONLY_SITES = ['Boston DITP North, MA', 'Boston DITP South, MA']


def _filter_to_mu(df, site_name):
    """Filter MGS data to MU-only for multi-lab sites."""
    if site_name in MU_ONLY_SITES and 'sequencing_lab' in df.columns:
        return df[df['sequencing_lab'] == 'MU']
    return df

# Pathogens to plot (rows)
PATHOGENS_TO_PLOT = ['sars-cov-2', 'influenza_a', 'rsv']

# Consistent font sizes for the figure
LABEL_FONT = FONT_SIZE_LARGE + 4
LABEL2_FONT = 19
TICK_FONT = 16
CORR_FONT = 14

# Clinical data column names for each pathogen
CLINICAL_PATHOGEN_MAP = {
    'sars-cov-2': 'SARS-CoV-2',
    'influenza_a': 'Influenza A',
    'rsv': 'RSV',
}

# Clinical comparison sites (Boston DITP North and South)
CLINICAL_SITES = ['Boston DITP North, MA', 'Boston DITP South, MA']

# Normalization method to column suffix mapping
NORMALIZATION_COLUMNS = {
    'pmmov': 'ra_clade_pmmov_norm',
    'raw': 'ra_clade',
}

# Normalization display names for y-axis labels
NORMALIZATION_LABELS = {
    'pmmov': 'Wastewater sequencing signal',
    'raw': 'Relative abundance',
}


def format_r_with_stars(r, p):
    """
    Format correlation R value with significance stars.

    Args:
        r: Spearman correlation coefficient
        p: p-value

    Returns:
        Formatted string like "R = 0.85***" for p < 0.001
    """
    if np.isnan(r) or np.isnan(p):
        return 'R = nan'

    r_str = f'R = {r:.2f}'
    if p < 0.001:
        return f'{r_str}***'
    elif p < 0.01:
        return f'{r_str}**'
    elif p < 0.05:
        return f'{r_str}*'
    else:
        return r_str


def print_correlation_summary(correlations, save_tsv_path=None, verbose=False):
    """
    Print a summary table of all correlations with R and p-values.

    Args:
        correlations: list of dicts with keys: pathogen, mgs_site, comparison_type,
                      comparison_source, r_value, p_value, n_points
        save_tsv_path: Optional path to save correlations as TSV file
        verbose: If True, print full correlation tables to stdout.
    """
    if not correlations:
        if verbose:
            print("\nNo correlations calculated.")
        return

    if save_tsv_path:
        corr_df = pd.DataFrame(correlations)
        sort_cols = [c for c in ['comparison_type', 'pathogen', 'mgs_site', 'site_name'] if c in corr_df.columns]
        corr_df = corr_df.sort_values(sort_cols)

        def format_r_with_asterisks(row):
            r = row['r_value']
            p = row['p_value']
            if np.isnan(r) or np.isnan(p):
                return 'nan'
            r_str = f"{r:.2f}"
            if p < 0.001:
                return f"{r_str}***"
            elif p < 0.01:
                return f"{r_str}**"
            elif p < 0.05:
                return f"{r_str}*"
            else:
                return r_str

        corr_df['r_formatted'] = corr_df.apply(format_r_with_asterisks, axis=1)

        base_cols = ['pathogen', 'mgs_site']
        if 'site_name' in corr_df.columns:
            base_cols.append('site_name')
        if 'seq_lab' in corr_df.columns:
            base_cols.append('seq_lab')
        base_cols.extend(['comparison_type', 'comparison_source',
                          'r_value', 'p_value', 'n_points', 'r_formatted'])
        cols = [c for c in base_cols if c in corr_df.columns]
        corr_df = corr_df[cols]

        corr_df.to_csv(save_tsv_path, sep='\t', index=False)

    if not verbose:
        return

    print("\n" + "="*90)
    print("CORRELATION SUMMARY (Spearman R and p-values)")
    print("="*90)

    nwss_corrs = [c for c in correlations if c['comparison_type'] == 'NWSS']
    clinical_corrs = [c for c in correlations if c['comparison_type'] == 'Clinical']

    if nwss_corrs:
        print("\n--- MGS vs NWSS ddPCR ---")
        print(f"{'Pathogen':<15} {'MGS Site':<35} {'R':<8} {'p-value':<12} {'N':<5}")
        print("-" * 80)
        for c in sorted(nwss_corrs, key=lambda x: (x['pathogen'], x['mgs_site'])):
            p_str = f"{c['p_value']:.2e}" if c['p_value'] < 0.01 else f"{c['p_value']:.3f}"
            print(f"{c['pathogen']:<15} {c['mgs_site']:<35} {c['r_value']:<8.2f} {p_str:<12} {c['n_points']:<5}")

    if clinical_corrs:
        print("\n--- MGS vs MGH Clinical Testing ---")
        print(f"{'Pathogen':<15} {'MGS Site':<35} {'R':<8} {'p-value':<12} {'N':<5}")
        print("-" * 80)
        for c in sorted(clinical_corrs, key=lambda x: (x['pathogen'], x['mgs_site'])):
            p_str = f"{c['p_value']:.2e}" if c['p_value'] < 0.01 else f"{c['p_value']:.3f}"
            print(f"{c['pathogen']:<15} {c['mgs_site']:<35} {c['r_value']:<8.2f} {p_str:<12} {c['n_points']:<5}")

    print("="*90 + "\n")


def add_seasonal_shading(ax, date_min, date_max, alpha=0.3):
    """Add faint blue shading for winter and yellow for summer."""
    start_year = date_min.year
    end_year = date_max.year + 1

    winter_color = '#E6F2FF'
    summer_color = '#FFF3B0'

    for year in range(start_year - 1, end_year + 1):
        winter_start = pd.Timestamp(f'{year}-12-01')
        winter_end = pd.Timestamp(f'{year + 1}-03-01')
        shade_start = max(winter_start, date_min)
        shade_end = min(winter_end, date_max)
        if shade_start < shade_end:
            ax.axvspan(shade_start, shade_end, color=winter_color, alpha=alpha, zorder=0)

        summer_start = pd.Timestamp(f'{year}-06-01')
        summer_end = pd.Timestamp(f'{year}-09-01')
        shade_start = max(summer_start, date_min)
        shade_end = min(summer_end, date_max)
        if shade_start < shade_end:
            ax.axvspan(shade_start, shade_end, color=summer_color, alpha=alpha, zorder=0)


def get_overlapping_date_range(mgs_data, nwss_data, site_name, sewershed_ids, taxids):
    """
    Calculate the overlapping date range between MGS and NWSS data for a site.

    Returns:
        tuple: (date_min, date_max) or (None, None) if no overlap
    """
    mgs_filtered = _filter_to_mu(mgs_data[
        (mgs_data['site_name'] == site_name) &
        (mgs_data['taxid'].isin(taxids))
    ], site_name).copy()

    if mgs_filtered.empty:
        return None, None

    # Skip first N samples if configured (temporal outliers)
    if site_name in SITE_SKIP_FIRST_N:
        mgs_filtered = mgs_filtered.sort_values('date')
        unique_dates = sorted(mgs_filtered['date'].unique())
        if len(unique_dates) > SITE_SKIP_FIRST_N[site_name]:
            skip_dates = unique_dates[:SITE_SKIP_FIRST_N[site_name]]
            mgs_filtered = mgs_filtered[~mgs_filtered['date'].isin(skip_dates)]

    if mgs_filtered.empty:
        return None, None

    nwss_filtered = nwss_data[nwss_data['sewershed_id'].isin(sewershed_ids)]

    if nwss_filtered.empty:
        return None, None

    mgs_min, mgs_max = mgs_filtered['date'].min(), mgs_filtered['date'].max()
    nwss_min, nwss_max = nwss_filtered['date'].min(), nwss_filtered['date'].max()

    overlap_min = max(mgs_min, nwss_min)
    overlap_max = min(mgs_max, nwss_max)

    if overlap_min <= overlap_max:
        return overlap_min, overlap_max

    return None, None


def _build_site_info_dict(matches):
    """
    Build site_info_dict from matches DataFrame returned by load_mgs_nwss_matches().

    Returns:
        dict mapping site_name -> {sewershed_ids: [...], state: str}
    """
    site_info_dict = {}
    for site_name in matches['site_name'].unique():
        site_matches = matches[matches['site_name'] == site_name]
        sewershed_ids = []
        for _, row in site_matches.iterrows():
            sewershed_ids.extend(row['sewershed_ids'])
        state = site_matches['state'].iloc[0] if 'state' in site_matches.columns else _state_from_site_name(site_name)
        site_info_dict[site_name] = {
            'sewershed_ids': sewershed_ids,
            'state': state,
        }
    return site_info_dict


def plot_mgs_nwss_panel(
    mgs_data_dict,
    nwss_data_dict,
    site_info_dict,
    clinical_data=None,
    save_path=None,
    standardize_xlim=True,
    clip_comparison_to_mgs=True,
    normalization='pmmov',
    verbose=False
):
    """
    Create polished 3x5 panel figure comparing MGS and NWSS data, plus clinical.

    Args:
        mgs_data_dict: dict mapping pathogen_key -> DataFrame
        nwss_data_dict: dict mapping pathogen_key -> DataFrame
        site_info_dict: dict mapping site_name -> {sewershed_ids: [...]}
        clinical_data: DataFrame with clinical positive test data (optional)
        save_path: Optional path to save figure
        standardize_xlim: If True, use global MGS date range for x-axis limits
        clip_comparison_to_mgs: If True, only show NWSS/clinical data within MGS date range
        normalization: 'pmmov' (default) or 'raw'
        verbose: If True, print detailed correlation tables

    Returns:
        fig, axes
    """
    n_rows = len(PATHOGENS_TO_PLOT)
    n_cols = len(SITES_TO_PLOT) + (1 if clinical_data is not None else 0)
    has_clinical = clinical_data is not None

    if normalization not in NORMALIZATION_COLUMNS:
        raise ValueError(f"Unknown normalization method: {normalization}. "
                        f"Options: {list(NORMALIZATION_COLUMNS.keys())}")
    mgs_col = NORMALIZATION_COLUMNS[normalization]
    mgs_y_label = NORMALIZATION_LABELS[normalization]

    all_correlations = []

    fig_width = 18 if has_clinical else 14
    fig_height = 11

    if has_clinical:
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                      width_ratios=[1, 1, 1, 1, 0.01, 1],
                      wspace=0.35, hspace=0.35)
        axes = np.empty((n_rows, n_cols), dtype=object)
        for row in range(n_rows):
            for col in range(len(SITES_TO_PLOT)):
                axes[row, col] = fig.add_subplot(gs[row, col])
            axes[row, len(SITES_TO_PLOT)] = fig.add_subplot(gs[row, 5])
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Calculate global MGS date range
    global_mgs_min = None
    global_mgs_max = None
    if standardize_xlim:
        for pathogen_key in PATHOGENS_TO_PLOT:
            mgs_data = mgs_data_dict.get(pathogen_key)
            if mgs_data is None:
                continue
            all_plot_sites = list(SITES_TO_PLOT) + [s for s in CLINICAL_SITES if s not in SITES_TO_PLOT]
            for site_name in all_plot_sites:
                sites_to_check = [site_name]
                if site_name in SITES_SAME_COLUMN:
                    sites_to_check.extend(SITES_SAME_COLUMN[site_name])
                for check_site in sites_to_check:
                    site_mgs = _filter_to_mu(mgs_data[mgs_data['site_name'] == check_site], check_site)
                    if not site_mgs.empty:
                        if check_site in SITE_SKIP_FIRST_N:
                            site_mgs = site_mgs.sort_values('date')
                            unique_dates = sorted(site_mgs['date'].unique())
                            if len(unique_dates) > SITE_SKIP_FIRST_N[check_site]:
                                skip_dates = unique_dates[:SITE_SKIP_FIRST_N[check_site]]
                                site_mgs = site_mgs[~site_mgs['date'].isin(skip_dates)]
                        if not site_mgs.empty:
                            mgs_min = site_mgs['date'].min()
                            mgs_max = site_mgs['date'].max()
                            if global_mgs_min is None or mgs_min < global_mgs_min:
                                global_mgs_min = mgs_min
                            if global_mgs_max is None or mgs_max > global_mgs_max:
                                global_mgs_max = mgs_max

    # Calculate Boston MGS-only date range
    boston_mgs_only_min, boston_mgs_only_max = None, None
    for pathogen_key in PATHOGENS_TO_PLOT:
        mgs_data = mgs_data_dict.get(pathogen_key)
        if mgs_data is None:
            continue
        taxids = NWSS_PATHOGEN_MAP[pathogen_key]['taxids']
        for site_name in CLINICAL_SITES:
            site_mgs = _filter_to_mu(mgs_data[
                (mgs_data['site_name'] == site_name) &
                (mgs_data['taxid'].isin(taxids))
            ], site_name)
            if not site_mgs.empty:
                if site_name in SITE_SKIP_FIRST_N:
                    site_mgs = site_mgs.sort_values('date')
                    unique_dates = sorted(site_mgs['date'].unique())
                    if len(unique_dates) > SITE_SKIP_FIRST_N[site_name]:
                        skip_dates = unique_dates[:SITE_SKIP_FIRST_N[site_name]]
                        site_mgs = site_mgs[~site_mgs['date'].isin(skip_dates)]
                if not site_mgs.empty:
                    mgs_min = site_mgs['date'].min()
                    mgs_max = site_mgs['date'].max()
                    if boston_mgs_only_min is None or mgs_min < boston_mgs_only_min:
                        boston_mgs_only_min = mgs_min
                    if boston_mgs_only_max is None or mgs_max > boston_mgs_only_max:
                        boston_mgs_only_max = mgs_max

    # Track date ranges per site
    all_sites_for_date_ranges = list(SITES_TO_PLOT)
    for site in CLINICAL_SITES:
        if site not in all_sites_for_date_ranges:
            all_sites_for_date_ranges.append(site)

    site_date_ranges = {}
    for site_name in all_sites_for_date_ranges:
        if site_name not in site_info_dict:
            if site_name in CLINICAL_SITES:
                all_mgs_min = None
                all_mgs_max = None
                for pathogen_key in PATHOGENS_TO_PLOT:
                    mgs_data = mgs_data_dict.get(pathogen_key)
                    if mgs_data is None:
                        continue
                    taxids = NWSS_PATHOGEN_MAP[pathogen_key]['taxids']
                    site_mgs = mgs_data[
                        (mgs_data['site_name'] == site_name) &
                        (mgs_data['taxid'].isin(taxids))
                    ]
                    if not site_mgs.empty:
                        mgs_min = site_mgs['date'].min()
                        mgs_max = site_mgs['date'].max()
                        if all_mgs_min is None or mgs_min < all_mgs_min:
                            all_mgs_min = mgs_min
                        if all_mgs_max is None or mgs_max > all_mgs_max:
                            all_mgs_max = mgs_max
                if all_mgs_min is not None:
                    site_date_ranges[site_name] = (all_mgs_min, all_mgs_max)
            continue

        sewershed_ids = site_info_dict[site_name]['sewershed_ids']
        all_overlap_min = None
        all_overlap_max = None

        for pathogen_key in PATHOGENS_TO_PLOT:
            taxids = NWSS_PATHOGEN_MAP[pathogen_key]['taxids']
            mgs_data = mgs_data_dict.get(pathogen_key)
            nwss_data = nwss_data_dict.get(pathogen_key)

            if mgs_data is None or nwss_data is None:
                continue

            overlap_min, overlap_max = get_overlapping_date_range(
                mgs_data, nwss_data, site_name, sewershed_ids, taxids
            )

            if overlap_min is not None:
                if all_overlap_min is None or overlap_min < all_overlap_min:
                    all_overlap_min = overlap_min
                if all_overlap_max is None or overlap_max > all_overlap_max:
                    all_overlap_max = overlap_max

        if all_overlap_min is not None:
            site_date_ranges[site_name] = (all_overlap_min, all_overlap_max)

    # Compute per-site NWSS end date (used to clip MGS display curves)
    nwss_end_by_site = {}
    for site_name in SITES_TO_PLOT:
        if site_name not in site_info_dict:
            continue
        sewershed_ids = site_info_dict[site_name]['sewershed_ids']
        nwss_max_date = None
        for pathogen_key in PATHOGENS_TO_PLOT:
            nwss_data = nwss_data_dict.get(pathogen_key)
            if nwss_data is None:
                continue
            site_nwss = nwss_data[nwss_data['sewershed_id'].isin(sewershed_ids)]
            if not site_nwss.empty:
                pmax = site_nwss['date'].max()
                if nwss_max_date is None or pmax > nwss_max_date:
                    nwss_max_date = pmax
        if nwss_max_date is not None:
            nwss_end_by_site[site_name] = nwss_max_date

    # First pass: determine which sites have PMMoV-normalized NWSS data and collect y ranges
    site_has_pmmov = {}
    subplot_mgs_max = {}
    subplot_nwss_max = {}

    for row_idx, pathogen_key in enumerate(PATHOGENS_TO_PLOT):
        taxids = NWSS_PATHOGEN_MAP[pathogen_key]['taxids']
        mgs_data = mgs_data_dict.get(pathogen_key)
        nwss_data = nwss_data_dict.get(pathogen_key)

        for col_idx, site_name in enumerate(SITES_TO_PLOT):
            if site_name not in site_info_dict or site_name not in site_date_ranges:
                continue

            sewershed_ids = site_info_dict[site_name]['sewershed_ids']
            if clip_comparison_to_mgs:
                if site_name in CLINICAL_SITES or site_name in SITES_SAME_COLUMN.get('Boston DITP North, MA', []):
                    date_min = boston_mgs_only_min if boston_mgs_only_min else site_date_ranges[site_name][0]
                    date_max = boston_mgs_only_max if boston_mgs_only_max else site_date_ranges[site_name][1]
                else:
                    date_min, date_max = site_date_ranges[site_name]
            elif standardize_xlim and global_mgs_min is not None:
                date_min, date_max = global_mgs_min, global_mgs_max
            else:
                date_min, date_max = site_date_ranges[site_name]
            key = (row_idx, col_idx)

            if site_name not in site_has_pmmov and nwss_data is not None:
                site_nwss_check = nwss_data[nwss_data['sewershed_id'].isin(sewershed_ids)]
                site_has_pmmov[site_name] = (site_nwss_check['pcr_target_mic_lin'].notna()).sum() > 0

            sites_in_column = [site_name]
            if site_name in SITES_SAME_COLUMN:
                sites_in_column.extend(SITES_SAME_COLUMN[site_name])

            # Clip MGS display to NWSS end date (use full data for smoothing)
            mgs_display_max = pd.Timestamp(date_max)
            if site_name in nwss_end_by_site:
                mgs_display_max = min(mgs_display_max, pd.Timestamp(nwss_end_by_site[site_name]))

            max_mgs_value = 0
            for plot_site in sites_in_column:
                if mgs_data is not None:
                    site_mgs = _filter_to_mu(mgs_data[
                        (mgs_data['site_name'] == plot_site) &
                        (mgs_data['taxid'].isin(taxids))
                    ], plot_site).copy()
                    if not site_mgs.empty:
                        site_mgs = site_mgs.sort_values('date')
                        # Smooth on full data to avoid edge effects
                        mgs_smoothed = calculate_mmwr_smoothed_trend(site_mgs, 'date', mgs_col)
                        if not mgs_smoothed.empty:
                            smoothed_dates = pd.to_datetime(mgs_smoothed['date'])
                            display_min = pd.Timestamp(date_min)
                            if plot_site in SITE_SKIP_FIRST_N:
                                first_date = site_mgs['date'].min()
                                display_min = max(display_min, pd.Timestamp(first_date) + pd.Timedelta(days=1))
                            mgs_display = mgs_smoothed[
                                (smoothed_dates >= display_min) &
                                (smoothed_dates <= mgs_display_max)
                            ]
                            if not mgs_display.empty:
                                site_max = mgs_display['smoothed_values'].max()
                                if site_max > max_mgs_value:
                                    max_mgs_value = site_max

            if max_mgs_value > 0:
                subplot_mgs_max[key] = max_mgs_value * 1.1

            if nwss_data is not None:
                site_nwss = nwss_data[nwss_data['sewershed_id'].isin(sewershed_ids)].copy()
                site_nwss = filter_to_dominant_source(site_nwss)
                if not site_nwss.empty:
                    has_pmmov = site_has_pmmov.get(site_name, True)
                    nwss_col = 'pcr_target_mic_lin' if has_pmmov else 'pcr_target_avg_conc_lin'
                    valid_nwss = site_nwss[site_nwss[nwss_col].notna() & (site_nwss[nwss_col] >= 0)]
                    if not valid_nwss.empty:
                        nwss_smoothed = calculate_mmwr_smoothed_trend(valid_nwss, 'date', nwss_col)
                        if not nwss_smoothed.empty:
                            smoothed_dates = pd.to_datetime(nwss_smoothed['date'])
                            nwss_display = nwss_smoothed[
                                (smoothed_dates >= pd.Timestamp(date_min)) &
                                (smoothed_dates <= pd.Timestamp(date_max))
                            ]
                            if not nwss_display.empty:
                                subplot_nwss_max[key] = nwss_display['smoothed_values'].max() * 1.1

    # Second pass: plot each panel
    for row_idx, pathogen_key in enumerate(PATHOGENS_TO_PLOT):
        pathogen_config = NWSS_PATHOGEN_MAP[pathogen_key]
        pathogen_label = pathogen_config['label']
        taxids = pathogen_config['taxids']

        mgs_data = mgs_data_dict.get(pathogen_key)
        nwss_data = nwss_data_dict.get(pathogen_key)

        for col_idx, site_name in enumerate(SITES_TO_PLOT):
            ax1 = axes[row_idx, col_idx]
            ax2 = ax1.twinx()

            if site_name not in site_info_dict:
                ax1.text(0.5, 0.5, 'No match', ha='center', va='center',
                        transform=ax1.transAxes, fontsize=LABEL_FONT)
                continue

            sewershed_ids = site_info_dict[site_name]['sewershed_ids']

            if site_name not in site_date_ranges:
                ax1.text(0.5, 0.5, 'No overlap', ha='center', va='center',
                        transform=ax1.transAxes, fontsize=LABEL_FONT)
                continue

            if standardize_xlim and global_mgs_min is not None:
                date_min, date_max = global_mgs_min, global_mgs_max
            else:
                date_min, date_max = site_date_ranges[site_name]

            add_seasonal_shading(ax1, date_min, date_max)

            site_smoothed_data = {}
            sites_in_column = [site_name]
            if site_name in SITES_SAME_COLUMN:
                sites_in_column.extend(SITES_SAME_COLUMN[site_name])

            # Clip MGS display to NWSS end date (use full data for smoothing/correlation)
            mgs_display_max = pd.Timestamp(date_max)
            if site_name in nwss_end_by_site:
                mgs_display_max = min(mgs_display_max, pd.Timestamp(nwss_end_by_site[site_name]))

            for plot_site in sites_in_column:
                if mgs_data is not None:
                    site_mgs = _filter_to_mu(mgs_data[
                        (mgs_data['site_name'] == plot_site) &
                        (mgs_data['taxid'].isin(taxids))
                    ], plot_site).copy()

                    if not site_mgs.empty:
                        site_mgs = site_mgs.sort_values('date')
                        color = get_location_color(plot_site)

                        # Smooth on full data to avoid edge effects
                        site_mgs_smoothed = calculate_mmwr_smoothed_trend(site_mgs, 'date', mgs_col)

                        if not site_mgs_smoothed.empty:
                            # For display: clip to date range and skip first sample dates
                            smoothed_dates = pd.to_datetime(site_mgs_smoothed['date'])
                            display_min = pd.Timestamp(date_min)
                            if plot_site in SITE_SKIP_FIRST_N:
                                first_date = site_mgs['date'].min()
                                display_min = max(display_min, pd.Timestamp(first_date) + pd.Timedelta(days=1))
                            mgs_display = site_mgs_smoothed[
                                (smoothed_dates >= display_min) &
                                (smoothed_dates <= mgs_display_max)
                            ]
                            if not mgs_display.empty:
                                short_label = plot_site.split(',')[0]
                                ax1.plot(mgs_display['date'], mgs_display['smoothed_values'],
                                        color=color, linewidth=LINE_WIDTH * 1.5,
                                        alpha=LINE_ALPHA, label=short_label, zorder=4)

                            # Store full smoothed data for correlation
                            site_smoothed_data[plot_site] = {
                                'smoothed': site_mgs_smoothed,
                                'color': color,
                                'label': plot_site
                            }

            # Get NWSS data
            has_pmmov = site_has_pmmov.get(site_name, True)
            if nwss_data is not None:
                if clip_comparison_to_mgs:
                    if site_name in CLINICAL_SITES:
                        nwss_filter_min = boston_mgs_only_min if boston_mgs_only_min else date_min
                        nwss_filter_max = boston_mgs_only_max if boston_mgs_only_max else date_max
                    else:
                        nwss_filter_min, nwss_filter_max = site_date_ranges.get(site_name, (date_min, date_max))
                else:
                    nwss_filter_min, nwss_filter_max = date_min, date_max

                site_nwss = nwss_data[nwss_data['sewershed_id'].isin(sewershed_ids)].copy()
                site_nwss = filter_to_dominant_source(site_nwss)

                if not site_nwss.empty:
                    nwss_col = 'pcr_target_mic_lin' if has_pmmov else 'pcr_target_avg_conc_lin'
                    site_nwss = site_nwss[
                        (site_nwss[nwss_col] >= 0) & (site_nwss[nwss_col].notna())
                    ].copy()

                    if not site_nwss.empty:
                        nwss_smoothed = calculate_mmwr_smoothed_trend(site_nwss, 'date', nwss_col)
                        nwss_color = '#666666'

                        if not nwss_smoothed.empty:
                            nwss_smoothed_dates = pd.to_datetime(nwss_smoothed['date'])
                            nwss_display = nwss_smoothed[
                                (nwss_smoothed_dates >= pd.Timestamp(nwss_filter_min)) &
                                (nwss_smoothed_dates <= pd.Timestamp(nwss_filter_max))
                            ]
                            if not nwss_display.empty:
                                ax2.plot(nwss_display['date'], nwss_display['smoothed_values'],
                                        color=nwss_color, linewidth=LINE_WIDTH * 1.5,
                                        linestyle='--', alpha=LINE_ALPHA,
                                        label='NWSS', zorder=2)

                            # Calculate correlation
                            if site_smoothed_data:
                                from scipy.stats import spearmanr
                                has_multiple_sites = site_name in SITES_SAME_COLUMN

                                if has_multiple_sites:
                                    corr_items = []
                                    for site_id, site_info in site_smoothed_data.items():
                                        mgs_smoothed = site_info['smoothed']
                                        merged = pd.merge(
                                            mgs_smoothed[['mmwr_year', 'mmwr_week', 'smoothed_values']].rename(
                                                columns={'smoothed_values': 'mgs_value'}),
                                            nwss_smoothed[['mmwr_year', 'mmwr_week', 'smoothed_values']].rename(
                                                columns={'smoothed_values': 'nwss_value'}),
                                            on=['mmwr_year', 'mmwr_week'], how='inner'
                                        )
                                        if len(merged) >= 3:
                                            r, p = spearmanr(merged['mgs_value'], merged['nwss_value'])
                                            corr_items.append((format_r_with_stars(r, p), site_info['color']))
                                            all_correlations.append({
                                                'pathogen': pathogen_label,
                                                'mgs_site': site_id,
                                                'site_name': site_id,
                                                'comparison_type': 'NWSS',
                                                'comparison_source': 'NWSS ddPCR',
                                                'r_value': r, 'p_value': p, 'n_points': len(merged)
                                            })

                                    if corr_items:
                                        y_pos = 0.93
                                        font_size = CORR_FONT - 2
                                        for i, (text, color) in enumerate(corr_items):
                                            y = y_pos + i * (-0.14)
                                            ax1.plot(0.03, y, 'o', transform=ax1.transAxes,
                                                    color=color, markersize=6, zorder=10)
                                            ax1.text(0.06, y, text, transform=ax1.transAxes,
                                                    ha='left', va='center', fontsize=font_size,
                                                    color='black', zorder=10)
                                else:
                                    first_site_id = list(site_smoothed_data.keys())[0]
                                    first_site = site_smoothed_data[first_site_id]
                                    mgs_smoothed_data = first_site['smoothed']
                                    merged = pd.merge(
                                        mgs_smoothed_data[['mmwr_year', 'mmwr_week', 'smoothed_values']].rename(
                                            columns={'smoothed_values': 'mgs_value'}),
                                        nwss_smoothed[['mmwr_year', 'mmwr_week', 'smoothed_values']].rename(
                                            columns={'smoothed_values': 'nwss_value'}),
                                        on=['mmwr_year', 'mmwr_week'], how='inner'
                                    )
                                    if len(merged) >= 3:
                                        r, p = spearmanr(merged['mgs_value'], merged['nwss_value'])
                                        r_text = format_r_with_stars(r, p)
                                        all_correlations.append({
                                            'pathogen': pathogen_label,
                                            'mgs_site': first_site_id,
                                            'site_name': first_site_id,
                                            'comparison_type': 'NWSS',
                                            'comparison_source': 'NWSS ddPCR',
                                            'r_value': r, 'p_value': p, 'n_points': len(merged)
                                        })
                                        use_top_right = (site_name == 'Columbia WWTP, MO' and pathogen_key == 'sars-cov-2')
                                        if use_top_right:
                                            ax1.text(0.97, 0.95, r_text, transform=ax1.transAxes,
                                                    ha='right', va='top', fontsize=CORR_FONT)
                                        else:
                                            ax1.text(0.03, 0.95, r_text, transform=ax1.transAxes,
                                                    ha='left', va='top', fontsize=CORR_FONT)

            # Formatting
            key = (row_idx, col_idx)
            mgs_max = subplot_mgs_max.get(key)
            if mgs_max is not None and mgs_max > 0:
                ax1.set_ylim(bottom=-mgs_max * 0.05, top=mgs_max)
            nwss_max = subplot_nwss_max.get(key)
            if nwss_max is not None and nwss_max > 0:
                ax2.set_ylim(bottom=-nwss_max * 0.05, top=nwss_max)

            buffer = pd.Timedelta(days=7)
            ax1.set_xlim(date_min - buffer, date_max + buffer)
            ax1.set_ylabel('')
            ax2.set_ylabel('')

            if row_idx == n_rows - 1:
                import matplotlib.dates as mdates
                if standardize_xlim:
                    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                else:
                    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax1.set_xticklabels([])

            ax1.grid(True, alpha=0.3, zorder=1)
            ax1.set_axisbelow(True)
            ax1.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)

            ax1.tick_params(axis='both', labelsize=TICK_FONT)
            ax2.tick_params(axis='both', labelsize=TICK_FONT)

            from matplotlib.ticker import MaxNLocator, ScalarFormatter, FixedLocator

            class CleanScalarFormatter(ScalarFormatter):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.set_scientific(True)
                    self.set_powerlimits((-2, 2))
                def _set_format(self):
                    self.format = '%d'
                def __call__(self, x, pos=None):
                    if x == 0:
                        return '0'
                    return super().__call__(x, pos)

            def remove_duplicate_int_ticks(ax, axis='y'):
                if axis == 'y':
                    locs = ax.yaxis.get_major_locator()()
                    formatter = ax.yaxis.get_major_formatter()
                else:
                    locs = ax.xaxis.get_major_locator()()
                    formatter = ax.xaxis.get_major_formatter()
                ax.figure.canvas.draw()
                if axis == 'y':
                    formatter.set_axis(ax.yaxis)
                else:
                    formatter.set_axis(ax.xaxis)
                seen_labels = set()
                unique_locs = []
                for loc in locs:
                    label = formatter(loc)
                    if label not in seen_labels:
                        seen_labels.add(label)
                        unique_locs.append(loc)
                if axis == 'y':
                    ax.yaxis.set_major_locator(FixedLocator(unique_locs))
                else:
                    ax.xaxis.set_major_locator(FixedLocator(unique_locs))

            ax1.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            formatter1 = CleanScalarFormatter(useMathText=False)
            ax1.yaxis.set_major_formatter(formatter1)
            ax1.yaxis.get_offset_text().set_fontsize(TICK_FONT)
            remove_duplicate_int_ticks(ax1, 'y')

            ax2.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            formatter2 = CleanScalarFormatter(useMathText=False)
            ax2.yaxis.set_major_formatter(formatter2)
            ax2.yaxis.get_offset_text().set_fontsize(TICK_FONT)
            remove_duplicate_int_ticks(ax2, 'y')

    # Plot clinical column
    if has_clinical:
        clinical_col_idx = len(SITES_TO_PLOT)
        clinical_color = '#666666'

        if standardize_xlim and global_mgs_min is not None:
            boston_date_min, boston_date_max = global_mgs_min, global_mgs_max
        else:
            boston_date_min = boston_mgs_only_min
            boston_date_max = boston_mgs_only_max

        for row_idx, pathogen_key in enumerate(PATHOGENS_TO_PLOT):
            ax1 = axes[row_idx, clinical_col_idx]
            ax2 = ax1.twinx()

            clinical_col_name = CLINICAL_PATHOGEN_MAP.get(pathogen_key)
            pathogen_config = NWSS_PATHOGEN_MAP[pathogen_key]
            taxids = pathogen_config['taxids']

            if clinical_col_name is None or clinical_col_name not in clinical_data.columns:
                continue

            if clip_comparison_to_mgs:
                clinical_filter_min = boston_mgs_only_min if boston_mgs_only_min else boston_date_min
                clinical_filter_max = boston_mgs_only_max if boston_mgs_only_max else boston_date_max
            else:
                clinical_filter_min, clinical_filter_max = boston_date_min, boston_date_max

            clinical_all = clinical_data[['date', clinical_col_name]].copy()
            clinical_all = clinical_all[clinical_all[clinical_col_name].notna()]

            # Track clinical date range for clipping MGS curves to overlap period
            clinical_date_max = None
            clinical_display_filtered = clinical_all[
                (clinical_all['date'] >= clinical_filter_min) &
                (clinical_all['date'] <= clinical_filter_max)
            ]
            if not clinical_display_filtered.empty:
                cmax = clinical_display_filtered['date'].max()
                clinical_date_max = cmax.date() if hasattr(cmax, 'date') else cmax

            mgs_max_y = 0
            mgs_data = mgs_data_dict.get(pathogen_key)
            site_smoothed_data = {}

            for site_name in CLINICAL_SITES:
                if mgs_data is not None and boston_date_min is not None:
                    site_mgs = _filter_to_mu(mgs_data[
                        (mgs_data['site_name'] == site_name) &
                        (mgs_data['taxid'].isin(taxids))
                    ], site_name).copy()

                    if not site_mgs.empty:
                        site_mgs = site_mgs.sort_values('date')
                        color = get_location_color(site_name)

                        # Smooth on full data to avoid edge effects
                        mgs_smoothed = calculate_mmwr_smoothed_trend(site_mgs, 'date', mgs_col)

                        if not mgs_smoothed.empty:
                            # For display: clip to date range, skip first sample, clip to clinical range
                            smoothed_dates = pd.to_datetime(mgs_smoothed['date'])
                            display_min = pd.Timestamp(boston_date_min)
                            if site_name in SITE_SKIP_FIRST_N:
                                first_date = site_mgs['date'].min()
                                display_min = max(display_min, pd.Timestamp(first_date) + pd.Timedelta(days=1))
                            mgs_smoothed_display = mgs_smoothed[
                                (smoothed_dates >= display_min) &
                                (smoothed_dates <= pd.Timestamp(boston_date_max))
                            ]

                            # Further clip to clinical data date range for plotting
                            if clinical_date_max is not None:
                                display_dates = mgs_smoothed_display['date'].apply(
                                    lambda d: d.date() if hasattr(d, 'date') else d
                                )
                                mgs_smoothed_display = mgs_smoothed_display[
                                    display_dates <= clinical_date_max
                                ].copy()

                            short_label = site_name.split(',')[0]
                            if not mgs_smoothed_display.empty:
                                ax1.plot(mgs_smoothed_display['date'],
                                        mgs_smoothed_display['smoothed_values'],
                                        color=color, linewidth=LINE_WIDTH * 1.5,
                                        alpha=LINE_ALPHA, label=short_label, zorder=4)

                                site_max_y = mgs_smoothed_display['smoothed_values'].max()
                                if site_max_y > mgs_max_y:
                                    mgs_max_y = site_max_y

                            # Store full smoothed data for correlation
                            site_smoothed_data[site_name] = {
                                'smoothed': mgs_smoothed,
                                'color': color,
                                'label': short_label
                            }

            # Plot clinical data
            clinical_smoothed = None
            if not clinical_all.empty:
                clinical_smoothed = calculate_mmwr_smoothed_trend(
                    clinical_all, 'date', clinical_col_name
                )
                if clinical_smoothed is not None and not clinical_smoothed.empty:
                    smoothed_dates = pd.to_datetime(clinical_smoothed['date'])
                    clinical_smoothed_display = clinical_smoothed[
                        (smoothed_dates >= pd.Timestamp(clinical_filter_min)) &
                        (smoothed_dates <= pd.Timestamp(clinical_filter_max))
                    ]
                    if clinical_smoothed_display is not None and not clinical_smoothed_display.empty:
                        ax2.plot(clinical_smoothed_display['date'],
                                clinical_smoothed_display['smoothed_values'],
                                color=clinical_color, linewidth=LINE_WIDTH * 1.5,
                                linestyle=':', alpha=LINE_ALPHA, label='Clinical', zorder=2)

                    # Calculate correlation
                    if site_smoothed_data and clinical_smoothed is not None:
                        from scipy.stats import spearmanr
                        corr_items = []
                        for site_id, site_info in site_smoothed_data.items():
                            mgs_sm = site_info['smoothed']
                            merged = pd.merge(
                                mgs_sm[['mmwr_year', 'mmwr_week', 'smoothed_values']].rename(
                                    columns={'smoothed_values': 'mgs_value'}),
                                clinical_smoothed[['mmwr_year', 'mmwr_week', 'smoothed_values']].rename(
                                    columns={'smoothed_values': 'clinical_value'}),
                                on=['mmwr_year', 'mmwr_week'], how='inner'
                            )
                            if len(merged) >= 3:
                                r, p = spearmanr(merged['mgs_value'], merged['clinical_value'])
                                corr_items.append((format_r_with_stars(r, p), site_info['color']))
                                all_correlations.append({
                                    'pathogen': NWSS_PATHOGEN_MAP[pathogen_key]['label'],
                                    'mgs_site': site_id,
                                    'site_name': site_id,
                                    'comparison_type': 'Clinical',
                                    'comparison_source': 'MGH clinical testing',
                                    'r_value': r, 'p_value': p, 'n_points': len(merged)
                                })

                        if corr_items:
                            use_bottom = pathogen_key == 'sars-cov-2'
                            if use_bottom:
                                y_pos = 0.08
                                y_step = 0.14
                                corr_items = list(reversed(corr_items))
                            else:
                                y_pos = 0.93
                                y_step = -0.14
                            font_size = CORR_FONT - 2
                            for i, (text, color) in enumerate(corr_items):
                                y = y_pos + i * y_step
                                ax1.plot(0.03, y, 'o', transform=ax1.transAxes,
                                        color=color, markersize=6, zorder=10)
                                ax1.text(0.06, y, text, transform=ax1.transAxes,
                                        ha='left', va='center', fontsize=font_size,
                                        color='black', zorder=10)

                    # Set y limits
                    mgs_top = mgs_max_y * 1.1 if mgs_max_y > 0 else None
                    mgs_bottom = -mgs_max_y * 0.05 if mgs_max_y > 0 else 0
                    ax1.set_ylim(bottom=mgs_bottom, top=mgs_top)
                    if clinical_smoothed_display is not None and not clinical_smoothed_display.empty:
                        clinical_max = clinical_smoothed_display['smoothed_values'].max() * 1.1
                        if clinical_max > 0:
                            ax2.set_ylim(bottom=-clinical_max * 0.05, top=clinical_max)

            buffer = pd.Timedelta(days=7)
            if boston_date_min is not None and boston_date_max is not None:
                add_seasonal_shading(ax1, boston_date_min, boston_date_max)
            ax1.set_xlim(boston_date_min - buffer, boston_date_max + buffer)
            ax1.set_ylabel('')
            ax2.set_ylabel('')

            if row_idx == n_rows - 1:
                import matplotlib.dates as mdates
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax1.set_xticklabels([])

            ax1.grid(True, alpha=0.3, zorder=1)
            ax1.set_axisbelow(True)
            ax1.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax1.tick_params(axis='both', labelsize=TICK_FONT)
            ax2.tick_params(axis='both', labelsize=TICK_FONT)

            from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter, FixedLocator

            class CleanScalarFormatter2(ScalarFormatter):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.set_scientific(True)
                    self.set_powerlimits((-2, 2))
                def _set_format(self):
                    self.format = '%d'
                def __call__(self, x, pos=None):
                    if x == 0:
                        return '0'
                    return super().__call__(x, pos)

            ax1.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            formatter1 = CleanScalarFormatter2(useMathText=False)
            ax1.yaxis.set_major_formatter(formatter1)
            ax1.yaxis.get_offset_text().set_fontsize(TICK_FONT)

            def clinical_int_formatter(x, pos):
                if x == 0:
                    return '0'
                return f'{int(x)}'

            ax2.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both', integer=True))
            ax2.yaxis.set_major_formatter(FuncFormatter(clinical_int_formatter))

    # Layout adjustments
    plt.subplots_adjust(top=0.68, bottom=0.10, left=0.08, right=0.95)

    # Add header boxes and column titles
    from matplotlib.patches import Rectangle

    nwss_axes_positions = [axes[0, col_idx].get_position() for col_idx in range(len(SITES_TO_PLOT))]
    clinical_col_idx_pos = len(SITES_TO_PLOT) if has_clinical else None

    title_y = nwss_axes_positions[0].y1 + 0.035

    for col_idx, site_name in enumerate(SITES_TO_PLOT):
        bbox = nwss_axes_positions[col_idx]
        title_x = bbox.x0 + bbox.width / 2
        if site_name in SITES_SAME_COLUMN:
            site_title = 'Boston DITP, MA (MU)'
        else:
            site_title = site_name
        fig.text(title_x, title_y, site_title, fontsize=LABEL2_FONT,
                ha='center', va='bottom', weight='normal')

    if has_clinical:
        clinical_bbox = axes[0, clinical_col_idx_pos].get_position()
        clinical_title_x = clinical_bbox.x0 + clinical_bbox.width / 2
        fig.text(clinical_title_x, title_y, 'Boston, MA', fontsize=LABEL2_FONT,
                ha='center', va='bottom', weight='normal')

    # NWSS header box
    nwss_left = nwss_axes_positions[0].x0
    nwss_right = nwss_axes_positions[-1].x1

    header_height = 0.045
    header_y = title_y + 0.035

    rect_nwss = Rectangle((nwss_left, header_y), nwss_right - nwss_left, header_height,
                           transform=fig.transFigure,
                           facecolor='#CCCCCC', edgecolor='none', zorder=1, alpha=0.5)
    fig.patches.append(rect_nwss)

    fig.text((nwss_left + nwss_right) / 2, header_y + header_height / 2,
             'Wastewater PCR', fontsize=LABEL_FONT, weight='normal',
             ha='center', va='center', zorder=10)

    fig.text(nwss_left - 0.02, header_y + header_height + 0.005,
             'a', fontsize=LABEL_FONT + 4, weight='bold',
             ha='right', va='bottom', zorder=10)

    if has_clinical:
        clinical_bbox = axes[0, clinical_col_idx_pos].get_position()
        clinical_extend = 0.015
        rect_clinical = Rectangle((clinical_bbox.x0 - clinical_extend, header_y),
                                   clinical_bbox.width + 2 * clinical_extend, header_height,
                                   transform=fig.transFigure,
                                   facecolor='#CCCCCC', edgecolor='none', zorder=1, alpha=0.5)
        fig.patches.append(rect_clinical)

        fig.text(clinical_bbox.x0 + clinical_bbox.width / 2, header_y + header_height / 2,
                 'MGH clinical testing', fontsize=LABEL_FONT, weight='normal',
                 ha='center', va='center', zorder=10)

        fig.text(clinical_bbox.x0 - clinical_extend - 0.02, header_y + header_height + 0.005,
                 'b', fontsize=LABEL_FONT + 4, weight='bold',
                 ha='right', va='bottom', zorder=10)

    # Legends
    from matplotlib.lines import Line2D

    legend_y = header_y + header_height + 0.025
    legend_top = legend_y + 0.18

    col1_handles = []
    col1_labels = []
    all_legend_sites = list(SITES_TO_PLOT)
    for site in CLINICAL_SITES:
        if site not in all_legend_sites:
            all_legend_sites.append(site)

    for site_name in all_legend_sites:
        site_label = site_name
        if site_name in CLINICAL_SITES:
            site_label = site_label + ' (MU)'
        col1_handles.append(Line2D([0], [0], color=get_location_color(site_name),
                                   linewidth=LINE_WIDTH * 1.5))
        col1_labels.append(site_label)

    col2_handles = [
        Line2D([0], [0], color='#666666', linewidth=LINE_WIDTH * 1.5, linestyle='--')
    ]
    col2_labels = ['Wastewater PCR']
    if has_clinical:
        col2_handles.append(Line2D([0], [0], color='#666666', linewidth=LINE_WIDTH * 1.5,
                                   linestyle=':'))
        col2_labels.append('MGH clinical testing (Boston, MA)')

    title_x = nwss_left + 0.01
    legend_title_y = legend_top - 0.01
    fig.text(title_x, legend_title_y, 'CASPER wastewater sequencing',
             fontsize=LABEL_FONT, weight='normal', ha='left', va='top')

    legend1 = fig.legend(col1_handles, col1_labels, loc='upper left',
                        bbox_to_anchor=(nwss_left, legend_top - 0.025), frameon=False,
                        fontsize=LABEL2_FONT, ncol=1,
                        handletextpad=0.7, labelspacing=0.3)

    fig.canvas.draw()
    legend1_bbox = legend1.get_window_extent().transformed(fig.transFigure.inverted())
    col1_bottom = legend1_bbox.y0

    col1_width = 0.30
    legend_bottom_y = col1_bottom - 0.015
    legend2 = fig.legend(col2_handles, col2_labels, loc='lower left',
                        bbox_to_anchor=(nwss_left + col1_width, legend_bottom_y), frameon=False,
                        fontsize=LABEL2_FONT, ncol=1,
                        handletextpad=0.7, labelspacing=0.3)
    fig.add_artist(legend1)

    winter_patch = mpatches.Patch(color='#E6F2FF', alpha=0.5, label='Winter (Dec-Feb)')
    summer_patch = mpatches.Patch(color='#FFF3B0', alpha=0.5, label='Summer (Jun-Aug)')
    fig.legend(handles=[winter_patch, summer_patch], loc='lower right',
              bbox_to_anchor=(0.98, legend_bottom_y), frameon=False,
              fontsize=LABEL2_FONT)
    fig.add_artist(legend2)

    # Axis labels
    bottom_row_positions = [axes[n_rows-1, col].get_position() for col in range(n_cols)]
    all_axes_left = nwss_axes_positions[0].x0
    all_axes_bottom = bottom_row_positions[0].y0
    all_axes_top = nwss_axes_positions[0].y1

    # Pathogen row labels
    pathogen_labels = {'sars-cov-2': 'SARS-CoV-2', 'influenza_a': 'Influenza A', 'rsv': 'RSV'}
    row_label_width = 0.025
    row_label_x = 0.001
    pathogen_extend = 0.015

    first_row_bbox = axes[0, 0].get_position()
    last_row_bbox = axes[n_rows - 1, 0].get_position()
    single_box_top = first_row_bbox.y1 + pathogen_extend
    single_box_bottom = last_row_bbox.y0 - pathogen_extend
    single_box_height = single_box_top - single_box_bottom

    rect_pathogen_all = Rectangle((row_label_x, single_box_bottom), row_label_width, single_box_height,
                                   transform=fig.transFigure,
                                   facecolor='#CCCCCC', edgecolor='none', zorder=1, alpha=0.5)
    fig.patches.append(rect_pathogen_all)

    for row_idx, pathogen_key in enumerate(PATHOGENS_TO_PLOT):
        row_bbox = axes[row_idx, 0].get_position()
        pathogen_label = pathogen_labels.get(pathogen_key, pathogen_key)
        fig.text(row_label_x + row_label_width / 2, row_bbox.y0 + row_bbox.height / 2,
                 pathogen_label, fontsize=LABEL_FONT, ha='center', va='center',
                 rotation=90, zorder=10)

    y_center = (all_axes_bottom + all_axes_top) / 2
    y_label_x = row_label_x + row_label_width + 0.02
    fig.text(y_label_x, y_center, mgs_y_label,
             fontsize=LABEL_FONT, ha='center', va='center', rotation=90)

    col4_bbox = axes[1, 3].get_position()
    fig.text(col4_bbox.x1 + 0.032, y_center,
             'Wastewater PCR signal',
             fontsize=LABEL_FONT, ha='center', va='center', rotation=270)

    if has_clinical:
        col5_bbox = axes[1, clinical_col_idx_pos].get_position()
        fig.text(col5_bbox.x1 + 0.038, y_center,
                 'Positive tests',
                 fontsize=LABEL_FONT, ha='center', va='center', rotation=270)

    # Save correlation TSV
    tsv_path = None
    if save_path:
        tsv_path = str(Path(save_path).parent.parent / 'tables' / (Path(save_path).stem + '_correlations.tsv'))
    print_correlation_summary(all_correlations, save_tsv_path=tsv_path, verbose=verbose)

    if save_path:
        save_figure(fig, save_path)

    return fig, axes


# ============================================================================
# Clinical correlation table functions
# ============================================================================

# Boston sites for clinical comparison (both MU and SB sequenced)
# Tuples of (site_name, sequencing_lab)
_BOSTON_CLINICAL_SITES = [
    ('Boston DITP North, MA', 'MU'),
    ('Boston DITP South, MA', 'MU'),
    ('Boston DITP North, MA', 'SB'),
    ('Boston DITP South, MA', 'SB'),
]

# Note: no first-sample skip for correlation calculations — all data used


def generate_clinical_correlation_table(
    mgs_data_dict,
    clinical_data,
    normalization='pmmov',
):
    """
    Generate correlations between MGS and clinical data for Boston sites.

    Args:
        mgs_data_dict: dict mapping pathogen_key -> DataFrame with MGS data
        clinical_data: DataFrame with clinical positive test data
        normalization: 'pmmov', 'tobrfv', or 'raw' for MGS normalization

    Returns:
        List of correlation dicts
    """
    from scipy.stats import spearmanr
    import warnings

    if clinical_data is None or clinical_data.empty:
        return []

    # Get the MGS column based on normalization
    _norm_columns = {
        'pmmov': 'ra_clade_pmmov_norm',
        'tobrfv': 'ra_clade_tobrfv_norm',
        'raw': 'ra_clade',
    }
    mgs_col = _norm_columns.get(normalization, 'ra_clade_pmmov_norm')

    correlations = []

    for pathogen_key in PATHOGENS_TO_PLOT:
        pathogen_config = NWSS_PATHOGEN_MAP[pathogen_key]
        pathogen_label = pathogen_config['label']
        taxids = pathogen_config['taxids']

        mgs_data = mgs_data_dict.get(pathogen_key)
        if mgs_data is None:
            continue

        # Get clinical column name
        clinical_col = CLINICAL_PATHOGEN_MAP.get(pathogen_key)
        if clinical_col is None or clinical_col not in clinical_data.columns:
            continue

        for site_name, seq_lab in _BOSTON_CLINICAL_SITES:
            # Check if site has data
            site_mgs = mgs_data[
                (mgs_data['site_name'] == site_name) &
                (mgs_data['taxid'].isin(taxids))
            ].copy()

            # Filter by sequencing_lab for multi-lab sites
            if 'sequencing_lab' in site_mgs.columns:
                site_mgs = site_mgs[site_mgs['sequencing_lab'] == seq_lab]

            if site_mgs.empty:
                continue

            # Check if mgs_col exists
            if mgs_col not in site_mgs.columns:
                continue

            site_mgs = site_mgs.sort_values('date')

            # No first-sample skip for correlations — use all data

            # Apply MMWR smoothing to MGS data
            mgs_smoothed = calculate_mmwr_smoothed_trend(site_mgs, 'date', mgs_col)
            if mgs_smoothed.empty:
                continue

            # Use ALL clinical data for smoothing (avoids edge effects from
            # clipping before the 5-week MMWR moving average). The inner merge
            # on MMWR week below naturally restricts to overlapping weeks.
            clinical_all = clinical_data[
                clinical_data[clinical_col].notna()
            ].copy()

            if clinical_all.empty:
                continue

            # Apply MMWR smoothing to clinical data
            clinical_smoothed = calculate_mmwr_smoothed_trend(
                clinical_all, 'date', clinical_col
            )

            if clinical_smoothed.empty:
                continue

            # Merge on MMWR week and calculate correlation
            merged = pd.merge(
                mgs_smoothed[['mmwr_year', 'mmwr_week', 'smoothed_values']].rename(
                    columns={'smoothed_values': 'mgs_value'}),
                clinical_smoothed[['mmwr_year', 'mmwr_week', 'smoothed_values']].rename(
                    columns={'smoothed_values': 'clinical_value'}),
                on=['mmwr_year', 'mmwr_week'],
                how='inner'
            )

            if len(merged) >= 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r, p = spearmanr(merged['mgs_value'], merged['clinical_value'])

                correlations.append({
                    'pathogen': pathogen_label,
                    'mgs_site': site_name,
                    'site_name': site_name,
                    'seq_lab': seq_lab,
                    'comparison_type': 'Clinical',
                    'comparison_source': 'MGH clinical testing',
                    'normalization': normalization,
                    'r_value': r,
                    'p_value': p,
                    'n_points': len(merged)
                })

    return correlations


def generate_clinical_correlation_display_tables(
    mgs_data_dict,
    clinical_data,
    save_path=None
):
    """
    Generate formatted and raw correlation display tables for clinical comparison.

    Returns a dict with:
        'formatted': DataFrame with significance-star-annotated R values (manuscript style)
        'raw': DataFrame with separate R, p columns + median sampling frequencies
    """
    # Generate correlations for all three normalizations
    norm_labels = {'pmmov': 'PMMoV', 'tobrfv': 'ToBRFV', 'raw': 'Raw'}
    all_corrs = {}
    for norm_key in ['pmmov', 'tobrfv', 'raw']:
        corr_list = generate_clinical_correlation_table(
            mgs_data_dict, clinical_data,
            normalization=norm_key
        )
        if corr_list:
            all_corrs[norm_key] = pd.DataFrame(corr_list)

    if not all_corrs:
        return {'formatted': pd.DataFrame(), 'raw': pd.DataFrame()}

    # Merge into one DataFrame
    merged = None
    for norm_key, norm_label in norm_labels.items():
        if norm_key not in all_corrs:
            continue
        df = all_corrs[norm_key][['pathogen', 'mgs_site', 'site_name', 'seq_lab',
                                   'r_value', 'p_value', 'n_points']].copy()
        suffix = f'_{norm_label.lower()}'
        df = df.rename(columns={
            'r_value': f'r{suffix}', 'p_value': f'p{suffix}', 'n_points': f'n{suffix}'
        })
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df.drop(columns=['site_name'], errors='ignore'),
                             on=['pathogen', 'mgs_site', 'seq_lab'], how='outer')

    # Sort by seq_lab (MU first), site_name, pathogen
    pathogen_labels_map = {NWSS_PATHOGEN_MAP[k]['label']: i for i, k in enumerate(PATHOGENS_TO_PLOT)}
    seq_lab_order = {'MU': 0, 'SB': 1}
    merged['_seq'] = merged['seq_lab'].map(seq_lab_order).fillna(2)
    merged['_path'] = merged['pathogen'].map(pathogen_labels_map).fillna(99)
    merged = merged.sort_values(['_seq', 'site_name', '_path']).drop(columns=['_seq', '_path'])

    pathogens_list = ['SARS-CoV-2', 'Influenza A', 'RSV']
    norms_list = ['Raw', 'PMMoV', 'ToBRFV']

    def format_r_stars(r, p):
        if pd.isna(r) or pd.isna(p):
            return ''
        r_str = f'{r:.2f}'
        if p < 0.001: return f'{r_str}***'
        elif p < 0.01: return f'{r_str}**'
        elif p < 0.05: return f'{r_str}*'
        return r_str

    # --- Compute median sampling frequencies per (site, lab) ---
    site_freq = {}
    for _, freq_row in merged.drop_duplicates(['mgs_site', 'seq_lab']).iterrows():
        mgs_site = freq_row['mgs_site']
        freq_lab = freq_row['seq_lab']
        mgs_data = mgs_data_dict.get('sars-cov-2')
        if mgs_data is None or clinical_data is None:
            continue
        taxids = NWSS_PATHOGEN_MAP['sars-cov-2']['taxids']
        site_mgs = mgs_data[(mgs_data['site_name'] == mgs_site) & (mgs_data['taxid'].isin(taxids))]
        if 'sequencing_lab' in site_mgs.columns:
            site_mgs = site_mgs[site_mgs['sequencing_lab'] == freq_lab]
        clinical_col = CLINICAL_PATHOGEN_MAP.get('sars-cov-2')
        clinical_dates = clinical_data[clinical_data[clinical_col].notna()]['date'] if clinical_col in clinical_data.columns else pd.Series()
        if site_mgs.empty or clinical_dates.empty:
            continue
        mgs_min, mgs_max = site_mgs['date'].min(), site_mgs['date'].max()
        clin_min, clin_max = clinical_dates.min(), clinical_dates.max()
        overlap_start, overlap_end = max(mgs_min, clin_min), min(mgs_max, clin_max)
        mgs_dates = np.sort(site_mgs[(site_mgs['date'] >= overlap_start) & (site_mgs['date'] <= overlap_end)]['date'].unique())
        clin_dates_overlap = np.sort(clinical_dates[(clinical_dates >= overlap_start) & (clinical_dates <= overlap_end)].unique())
        mgs_freq = float(np.median(np.diff(mgs_dates).astype('timedelta64[D]').astype(int))) if len(mgs_dates) > 1 else np.nan
        clin_freq = float(np.median(np.diff(clin_dates_overlap).astype('timedelta64[D]').astype(int))) if len(clin_dates_overlap) > 1 else np.nan
        site_freq[(mgs_site, freq_lab)] = {'mgs_freq': mgs_freq, 'clinical_freq': clin_freq}

    # --- Build ordered list of (mgs_site, site_name, seq_lab) ---
    site_order = merged.drop_duplicates(['mgs_site', 'seq_lab'])[['mgs_site', 'site_name', 'seq_lab']].values.tolist()

    # --- Build formatted table ---
    fmt_cols = ['Sampling site', 'sequencing_lab', 'N (weeks)']
    for p in pathogens_list:
        for norm in norms_list:
            fmt_cols.append(f'{p} ({norm})')

    fmt_rows = []
    for mgs_site, site_name, seq_lab in site_order:
        site_data = merged[(merged['mgs_site'] == mgs_site) & (merged['seq_lab'] == seq_lab)]
        sc2 = site_data[site_data['pathogen'] == 'SARS-CoV-2']
        n_col = 'n_pmmov' if 'n_pmmov' in site_data.columns else 'n_tobrfv'
        n = int(sc2[n_col].iloc[0]) if not sc2.empty and pd.notna(sc2[n_col].iloc[0]) else ''
        row = {'Sampling site': site_name, 'sequencing_lab': seq_lab, 'N (weeks)': n}
        for p in pathogens_list:
            p_data = site_data[site_data['pathogen'] == p]
            for norm in norms_list:
                r_col = f'r_{norm.lower()}'
                p_col_name = f'p_{norm.lower()}'
                if not p_data.empty and r_col in p_data.columns and pd.notna(p_data[r_col].iloc[0]):
                    row[f'{p} ({norm})'] = format_r_stars(p_data[r_col].iloc[0], p_data[p_col_name].iloc[0])
                else:
                    row[f'{p} ({norm})'] = ''
        fmt_rows.append(row)
    formatted_df = pd.DataFrame(fmt_rows)

    # --- Build raw table ---
    raw_cols = ['Sampling site', 'sequencing_lab', 'N (weeks)']
    for p in pathogens_list:
        for norm in norms_list:
            raw_cols.extend([f'{p} R ({norm})', f'{p} p ({norm})'])
    raw_cols.extend(['Median clinical freq (days)', 'Median MGS freq (days)'])

    raw_rows = []
    for mgs_site, site_name, seq_lab in site_order:
        site_data = merged[(merged['mgs_site'] == mgs_site) & (merged['seq_lab'] == seq_lab)]
        sc2 = site_data[site_data['pathogen'] == 'SARS-CoV-2']
        n_col = 'n_pmmov' if 'n_pmmov' in site_data.columns else 'n_tobrfv'
        n = int(sc2[n_col].iloc[0]) if not sc2.empty and pd.notna(sc2[n_col].iloc[0]) else ''
        row = {'Sampling site': site_name, 'sequencing_lab': seq_lab, 'N (weeks)': n}
        for p in pathogens_list:
            p_data = site_data[site_data['pathogen'] == p]
            for norm in norms_list:
                r_col = f'r_{norm.lower()}'
                p_col_name = f'p_{norm.lower()}'
                if not p_data.empty and r_col in p_data.columns and pd.notna(p_data[r_col].iloc[0]):
                    row[f'{p} R ({norm})'] = p_data[r_col].iloc[0]
                    row[f'{p} p ({norm})'] = p_data[p_col_name].iloc[0]
                else:
                    row[f'{p} R ({norm})'] = ''
                    row[f'{p} p ({norm})'] = ''
        freq = site_freq.get((mgs_site, seq_lab), {})
        clin_f = freq.get('clinical_freq', np.nan)
        mgs_f = freq.get('mgs_freq', np.nan)
        row['Median clinical freq (days)'] = f'{clin_f:.0f}' if not np.isnan(clin_f) else ''
        row['Median MGS freq (days)'] = f'{mgs_f:.0f}' if not np.isnan(mgs_f) else ''
        raw_rows.append(row)
    raw_df = pd.DataFrame(raw_rows)

    if save_path:
        formatted_df.to_csv(save_path, sep='\t', index=False)
        raw_save = save_path.replace('.tsv', '_raw.tsv')
        raw_df.to_csv(raw_save, sep='\t', index=False)

    return {'formatted': formatted_df, 'raw': raw_df}
