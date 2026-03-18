#!/usr/bin/env python3
"""
Supplementary figure comparing MGS pathogen abundance with NWSS PCR data for ALL sites.

Creates a multi-row (sites) x 3-column (pathogens) figure showing MGS vs NWSS
comparison for all sites that have NWSS ddPCR data pairing.

Sites as rows, pathogens (SARS-CoV-2, Influenza A, RSV) as columns.
Uses same styling as the main panel figure (plot_mgs_nwss_panel.py).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# Import plotting config
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *
from plot_config import _state_from_site_name, filter_timeseries_data

# Import data loading functions from centralized module
from data_loaders import (
    load_nwss_data,
    load_mgs_nwss_matches,
    load_mgs_pathogen_data,
    load_sample_metadata,
    filter_to_dominant_source,
    NWSS_PATHOGEN_MAP
)

# Import correlation summary function from panel script
from plot_mgs_nwss_panel import print_correlation_summary, format_r_with_stars

# Import scipy for correlation calculations
from scipy.stats import spearmanr

# Pathogens to plot (columns)
PATHOGENS_TO_PLOT = ['sars-cov-2', 'influenza_a', 'rsv']

# Normalization method to column suffix mapping
NORMALIZATION_COLUMNS = {
    'pmmov': 'ra_clade_pmmov_norm',
    'rrna': 'ra_clade_rrna_norm',
    'tobrfv': 'ra_clade_tobrfv_norm',
    'raw': 'ra_clade',
}

# Normalization display names for y-axis labels
NORMALIZATION_LABELS = {
    'pmmov': 'CASPER wastewater sequencing signal (PMMoV-norm)',
    'rrna': 'CASPER wastewater sequencing signal (rRNA-norm)',
    'tobrfv': 'CASPER wastewater sequencing signal (ToBRFV-norm)',
    'raw': 'Relative abundance',
}

# Sites with temporal outliers to skip (first N samples)
# NOTE: Boston DITP first-sample skip is now handled by filter_timeseries_data
# in plot_config (MU-only). This dict is kept for any future site-specific skips.
SITE_SKIP_FIRST_N = {}

# Sites to exclude from supplementary figure (no overlapping NWSS data)
SITES_TO_EXCLUDE = [
    'Central Oklahoma (OK-A), OK',   # NWSS data ends before MGS sampling began
    'Central Oklahoma (OK-B), OK',   # NWSS data ends before MGS sampling began
    'Kansas City Blue River WWTP, MO',  # Overlap too short to be meaningful
    'Ontario, CA',                    # WWS cutoff leaves overlap too short
    'Palo Alto RWQCP, CA',           # WWS cutoff leaves overlap too short
    'Sacramento, CA',                # WWS cutoff leaves overlap too short
]

# Hardcoded short display names for this plot (to save space in row labels)
SITE_SHORT_NAMES = {
    'Boston DITP North, MA': 'Boston DITPN, MA',
    'Boston DITP South, MA': 'Boston DITPS, MA',
}

# Override R= text placement for specific (site_name, pathogen_key) pairs.
# Values: 'top_left' or 'top_right'
R_TEXT_PLACEMENT_OVERRIDES = {
    ("Chicago (CHI-A), IL", 'influenza_a'): 'top_left',
}

# Pathogen display names for table
PATHOGEN_DISPLAY_NAMES = {
    'sars-cov-2': 'SARS-CoV-2',
    'influenza_a': 'Influenza A',
    'rsv': 'RSV'
}


def generate_nwss_metadata_table(
    nwss_data_dict,
    site_info_dict,
    mgs_data_dict=None,
    save_path=None
):
    """
    Generate a supplementary table with NWSS metadata for each site x pathogen combination.

    Includes: Site name, Pathogen, PCR type, PMMoV normalization status, date range, frequency.

    Args:
        nwss_data_dict: dict mapping pathogen_key -> DataFrame with NWSS data
        site_info_dict: dict mapping site_name -> {sewershed_ids: [...], state: str, ...}
        mgs_data_dict: Optional dict mapping pathogen_key -> DataFrame with MGS data.
                       If provided, NWSS data is filtered to the MGS date range per site.
        save_path: Optional path to save TSV file

    Returns:
        DataFrame with NWSS metadata table
    """
    rows = []
    seen_site_pathogen = set()  # Track (site_name, pathogen) to avoid duplicates

    for site_name, site_info in sorted(site_info_dict.items()):
        sewershed_ids = site_info['sewershed_ids']

        for pathogen_key in PATHOGENS_TO_PLOT:
            # Skip if we've already seen this site-pathogen combination
            dedup_key = (site_name, pathogen_key)
            if dedup_key in seen_site_pathogen:
                continue

            nwss_data = nwss_data_dict.get(pathogen_key)
            if nwss_data is None:
                continue

            # Filter to this site's sewersheds
            site_nwss = nwss_data[nwss_data['sewershed_id'].isin(sewershed_ids)].copy()

            if site_nwss.empty:
                continue

            # Filter NWSS to MGS date range if MGS data provided
            if mgs_data_dict is not None:
                mgs_data = mgs_data_dict.get(pathogen_key)
                if mgs_data is not None and not mgs_data.empty:
                    # Get MGS data for this site
                    site_mgs = mgs_data[mgs_data['site_name'] == site_name]
                    if not site_mgs.empty:
                        mgs_date_min = site_mgs['date'].min()
                        mgs_date_max = site_mgs['date'].max()
                        # Filter NWSS to MGS date range
                        site_nwss = site_nwss[
                            (pd.to_datetime(site_nwss['sample_collect_date']) >= mgs_date_min) &
                            (pd.to_datetime(site_nwss['sample_collect_date']) <= mgs_date_max)
                        ]

            if site_nwss.empty:
                continue

            # Keep only the most frequent source during the overlap period
            site_nwss = filter_to_dominant_source(site_nwss)

            # Mark as seen
            seen_site_pathogen.add(dedup_key)

            # Extract metadata
            # PCR type - get unique values and format nicely
            pcr_types = site_nwss['pcr_type'].dropna().unique()
            # Standardize each PCR type individually
            standardized_types = set()
            for pt in pcr_types:
                pt_lower = pt.lower().strip()
                if pt_lower == 'ddpcr':
                    standardized_types.add('ddPCR')
                elif pt_lower == 'dpcr' or ('qiagen' in pt_lower and 'dpcr' in pt_lower):
                    standardized_types.add('dPCR')
                elif pt_lower == 'qpcr':
                    standardized_types.add('RT-qPCR')
                elif pt_lower == 'pcr':
                    standardized_types.add('RT-qPCR')
                else:
                    standardized_types.add(pt)
            pcr_type_str = ', '.join(sorted(standardized_types))

            # PCR gene target
            gene_target_str = ''
            if 'pcr_gene_target_agg' in site_nwss.columns:
                gene_targets = site_nwss['pcr_gene_target_agg'].dropna().unique()
                # Standardize and clean up gene target names
                cleaned_targets = set()
                for gt in gene_targets:
                    gt_clean = str(gt).strip().lower()
                    # Simplify common patterns
                    gt_clean = gt_clean.replace(' combined', '').replace('_', '')
                    gt_clean = gt_clean.replace(',', '+').replace(' ', '')
                    gt_clean = gt_clean.replace('gene', '')
                    gt_clean = gt_clean.upper()
                    cleaned_targets.add(gt_clean)
                gene_target_str = ', '.join(sorted(cleaned_targets))

            # Normalization target - check what fecal indicator is used
            norm_target_str = 'None'
            if 'hum_frac_target_mic' in site_nwss.columns:
                norm_targets = site_nwss['hum_frac_target_mic'].dropna().unique()
                if any('pepper' in str(t).lower() or 'pmmov' in str(t).lower()
                       for t in norm_targets):
                    norm_target_str = 'PMMoV'
                elif len(norm_targets) > 0:
                    # Use whatever target is specified
                    norm_target_str = str(norm_targets[0])

            # Also check if pcr_target_mic_lin has non-null values (indicates normalization was done)
            if norm_target_str == 'None':
                has_norm_data = (site_nwss['pcr_target_mic_lin'].notna() &
                                (site_nwss['pcr_target_mic_lin'] > 0)).any()
                if has_norm_data:
                    norm_target_str = 'PMMoV'  # Assume PMMoV if normalized data exists

            # Calculate median sampling frequency in days (using unique dates)
            site_nwss_dates = pd.to_datetime(site_nwss['sample_collect_date'])
            unique_dates = site_nwss_dates.drop_duplicates().sort_values()
            if len(unique_dates) > 1:
                date_diffs = unique_dates.diff().dropna().dt.days
                median_freq = int(round(date_diffs.median()))
            else:
                median_freq = None

            # Data source - get unique source designations
            source_str = ''
            if 'source' in site_nwss.columns:
                sources = site_nwss['source'].dropna().unique()
                source_display = {
                    'state_territory': 'State/local health dept.',
                    'cdc_verily': 'CDC/Verily',
                    'wws': 'WastewaterSCAN',
                }
                display_sources = sorted(set(
                    source_display.get(s.strip(), s) for s in sources
                ))
                source_str = ', '.join(display_sources)

            rows.append({
                'Site': site_name,
                'Pathogen': PATHOGEN_DISPLAY_NAMES.get(pathogen_key, pathogen_key),
                'PCR Type': pcr_type_str,
                'Gene Target': gene_target_str,
                'Normalization': norm_target_str,
                'Median Freq (days)': median_freq,
                'Source': source_str,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No NWSS metadata to generate")
        return df

    # Sort by region, state, site name, then pathogen
    from plot_config import sort_locations_by_state_and_name, REGION_ORDER, get_state_region

    # Add state column for sorting
    df['state'] = df['Site'].apply(_state_from_site_name)

    # Add region and region order for sorting
    region_order_map = {region: i for i, region in enumerate(REGION_ORDER)}
    df['_region'] = df['state'].apply(get_state_region)
    df['_region_order'] = df['_region'].map(region_order_map).fillna(len(REGION_ORDER))

    # Add pathogen order
    pathogen_order = {'SARS-CoV-2': 0, 'Influenza A': 1, 'RSV': 2}
    df['pathogen_order'] = df['Pathogen'].map(pathogen_order)

    # Sort by region, state, site name, then pathogen
    df = df.sort_values(['_region_order', 'state', 'Site', 'pathogen_order'])
    df = df.drop(columns=['state', '_region', '_region_order', 'pathogen_order'])

    # Save to file
    if save_path:
        df.to_csv(save_path, sep='\t', index=False)
        print(f"Saved NWSS metadata table to {save_path}")

    return df


def print_nwss_metadata_table_markdown(df):
    """Print the NWSS metadata table in markdown format for the manuscript."""
    print("\n" + "="*100)
    print("NWSS METADATA TABLE (for Supplementary Materials)")
    print("="*100 + "\n")

    # Print as markdown table
    has_source = 'Source' in df.columns
    if has_source:
        print("| Site | Pathogen | PCR Type | Gene Target | Normalization | Freq (days) | Source |")
        print("|------|----------|----------|-------------|---------------|-------------|--------|")
    else:
        print("| Site | Pathogen | PCR Type | Gene Target | Normalization | Freq (days) |")
        print("|------|----------|----------|-------------|---------------|-------------|")

    for _, row in df.iterrows():
        freq = row['Median Freq (days)'] if pd.notna(row['Median Freq (days)']) else 'N/A'
        base = f"| {row['Site']} | {row['Pathogen']} | {row['PCR Type']} | {row['Gene Target']} | {row['Normalization']} | {freq} |"
        if has_source:
            source = row.get('Source', '')
            base = f"{base} {source} |"
        print(base)

    print("\n" + "="*100 + "\n")


def generate_correlation_table(
    mgs_data_dict,
    nwss_data_dict,
    site_info_dict,
    normalization='pmmov',
    save_path=None
):
    """
    Generate a correlation table comparing MGS and NWSS data for all sites.

    This calculates correlations without generating the full plot figure.

    Args:
        mgs_data_dict: dict mapping pathogen_key -> DataFrame
        nwss_data_dict: dict mapping pathogen_key -> DataFrame
        site_info_dict: dict mapping site_name -> {sewershed_ids: [...], state: str}
        normalization: Normalization method for MGS data ('pmmov', 'rrna', 'tobrfv', 'raw')
        save_path: Optional path to save TSV file

    Returns:
        DataFrame with correlation results
    """
    import warnings

    # Get MGS column based on normalization method
    if normalization not in NORMALIZATION_COLUMNS:
        raise ValueError(f"Unknown normalization method: {normalization}. "
                        f"Options: {list(NORMALIZATION_COLUMNS.keys())}")
    mgs_col = NORMALIZATION_COLUMNS[normalization]

    # Get ordered sites
    mu_sites_all, sb_sites_all = get_ordered_sites(site_info_dict)
    all_sites_initial = mu_sites_all + sb_sites_all

    # Calculate date ranges for all sites
    site_date_ranges = {}
    site_has_pmmov = {}

    for site_name in all_sites_initial:
        if site_name not in site_info_dict:
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

            # Check if NWSS has PMMoV-normalized data
            if site_name not in site_has_pmmov and nwss_data is not None:
                site_nwss_check = nwss_data[nwss_data['sewershed_id'].isin(sewershed_ids)]
                site_has_pmmov[site_name] = (site_nwss_check['pcr_target_mic_lin'].notna()).sum() > 0

        if all_overlap_min is not None:
            site_date_ranges[site_name] = (all_overlap_min, all_overlap_max)

    # Filter to sites with data
    all_sites = [s for s in all_sites_initial if s in site_date_ranges]

    # Calculate correlations
    all_correlations = []

    for site_name in all_sites:
        if site_name not in site_info_dict or site_name not in site_date_ranges:
            continue

        site_info = site_info_dict[site_name]
        sewershed_ids = site_info['sewershed_ids']
        date_min, date_max = site_date_ranges[site_name]
        has_pmmov = site_has_pmmov.get(site_name, True)
        seq_labs = site_info.get('seq_labs', [site_info.get('seq_lab', 'MU')])

        for seq_lab in seq_labs:
            for pathogen_key in PATHOGENS_TO_PLOT:
                pathogen_config = NWSS_PATHOGEN_MAP[pathogen_key]
                pathogen_label = pathogen_config['label']
                taxids = pathogen_config['taxids']

                mgs_data = mgs_data_dict.get(pathogen_key)
                nwss_data = nwss_data_dict.get(pathogen_key)

                if mgs_data is None or nwss_data is None:
                    continue

                # Get MGS data (no date filter -- smooth full series, inner merge handles overlap)
                site_mgs = mgs_data[
                    (mgs_data['site_name'] == site_name) &
                    (mgs_data['taxid'].isin(taxids))
                ].copy()

                # Filter by sequencing_lab for multi-lab sites
                if len(seq_labs) > 1 and 'sequencing_lab' in site_mgs.columns:
                    site_mgs = site_mgs[site_mgs['sequencing_lab'] == seq_lab]

                if site_mgs.empty:
                    continue

                site_mgs = site_mgs.sort_values('date')

                # Skip first N dates if configured
                if site_name in SITE_SKIP_FIRST_N:
                    n_skip = SITE_SKIP_FIRST_N[site_name]
                    unique_dates = site_mgs['date'].unique()
                    if len(unique_dates) > n_skip:
                        skip_dates = unique_dates[:n_skip]
                        site_mgs = site_mgs[~site_mgs['date'].isin(skip_dates)]

                if site_mgs.empty:
                    continue

                # Apply MMWR smoothing to MGS
                mgs_smoothed = calculate_mmwr_smoothed_trend(site_mgs, 'date', mgs_col)

                if mgs_smoothed.empty:
                    continue

                # Get NWSS data (no date filter -- smooth full series, inner merge handles overlap)
                site_nwss = nwss_data[
                    (nwss_data['sewershed_id'].isin(sewershed_ids))
                ].copy()

                if site_nwss.empty:
                    continue

                # Keep only the most frequent source to avoid mixing labs
                site_nwss = filter_to_dominant_source(site_nwss)

                nwss_col = 'pcr_target_mic_lin' if has_pmmov else 'pcr_target_avg_conc_lin'
                site_nwss = site_nwss[
                    (site_nwss[nwss_col] >= 0) &
                    (site_nwss[nwss_col].notna())
                ].copy()

                if site_nwss.empty:
                    continue

                # Apply MMWR smoothing to NWSS
                nwss_smoothed = calculate_mmwr_smoothed_trend(site_nwss, 'date', nwss_col)

                if nwss_smoothed.empty:
                    continue

                # Calculate correlation on week-aligned data
                merged = pd.merge(
                    mgs_smoothed[['mmwr_year', 'mmwr_week', 'smoothed_values']].rename(
                        columns={'smoothed_values': 'mgs_value'}),
                    nwss_smoothed[['mmwr_year', 'mmwr_week', 'smoothed_values']].rename(
                        columns={'smoothed_values': 'nwss_value'}),
                    on=['mmwr_year', 'mmwr_week'],
                    how='inner'
                )

                if len(merged) >= 3:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r, p = spearmanr(merged['mgs_value'], merged['nwss_value'])

                    all_correlations.append({
                        'pathogen': pathogen_label,
                        'site_name': site_name,
                        'seq_lab': seq_lab,
                        'comparison_type': 'NWSS',
                        'comparison_source': 'NWSS ddPCR',
                        'normalization': normalization,
                        'r_value': r,
                        'p_value': p,
                        'n_points': len(merged)
                    })

    # Create DataFrame
    if not all_correlations:
        print(f"No correlations calculated for {normalization} normalization")
        return pd.DataFrame()

    corr_df = pd.DataFrame(all_correlations)
    corr_df = corr_df.sort_values(['pathogen', 'site_name'])

    # Add formatted R value with asterisk notation
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

    # Save to file if path provided
    if save_path:
        corr_df.to_csv(save_path, sep='\t', index=False)
        print(f"Saved {normalization} correlation table to {save_path}")

    return corr_df


def generate_combined_correlation_table(
    mgs_data_dict,
    nwss_data_dict,
    site_info_dict,
    save_path=None
):
    """
    Generate a combined correlation table with both PMMoV and ToBRFV normalization.

    Args:
        mgs_data_dict: dict mapping pathogen_key -> DataFrame
        nwss_data_dict: dict mapping pathogen_key -> DataFrame
        site_info_dict: dict mapping site_name -> {sewershed_ids: [...], state: str}
        save_path: Optional path to save TSV file

    Returns:
        DataFrame with correlation results for both normalization methods
    """
    # Generate correlations for both normalization methods
    results = {}
    for norm in ['pmmov', 'tobrfv']:
        corr_df = generate_correlation_table(
            mgs_data_dict, nwss_data_dict, site_info_dict,
            normalization=norm
        )
        if not corr_df.empty:
            results[norm] = corr_df

    if not results:
        print("No correlations calculated for any normalization method")
        return pd.DataFrame()

    # Merge PMMoV and ToBRFV results
    pmmov_df = results.get('pmmov')
    tobrfv_df = results.get('tobrfv')

    if pmmov_df is not None:
        pmmov_df = pmmov_df[['pathogen', 'site_name', 'seq_lab',
                              'r_value', 'p_value', 'n_points', 'r_formatted']].copy()
        pmmov_df = pmmov_df.rename(columns={
            'r_value': 'r_pmmov', 'p_value': 'p_pmmov',
            'n_points': 'n_points_pmmov', 'r_formatted': 'r_pmmov_formatted'
        })

    if tobrfv_df is not None:
        tobrfv_df = tobrfv_df[['pathogen', 'site_name',
                                'r_value', 'p_value', 'n_points', 'r_formatted']].copy()
        tobrfv_df = tobrfv_df.rename(columns={
            'r_value': 'r_tobrfv', 'p_value': 'p_tobrfv',
            'n_points': 'n_points_tobrfv', 'r_formatted': 'r_tobrfv_formatted'
        })

    if pmmov_df is not None and tobrfv_df is not None:
        combined = pd.merge(pmmov_df, tobrfv_df, on=['pathogen', 'site_name'], how='outer')
    elif pmmov_df is not None:
        combined = pmmov_df
    else:
        combined = tobrfv_df

    # Sort by seq_lab (MU first), site_name, then pathogen
    pathogen_order = {NWSS_PATHOGEN_MAP[k]['label']: i for i, k in enumerate(NWSS_PATHOGEN_MAP)}
    seq_lab_order = {'MU': 0, 'SB': 1}
    combined['_seq_lab_order'] = combined['seq_lab'].map(seq_lab_order).fillna(2)
    combined['_pathogen_order'] = combined['pathogen'].map(pathogen_order).fillna(len(pathogen_order))
    combined = combined.sort_values(['_seq_lab_order', 'site_name', '_pathogen_order'])
    combined = combined.drop(columns=['_seq_lab_order', '_pathogen_order'])

    if save_path:
        combined.to_csv(save_path, sep='\t', index=False)
        print(f"Saved combined correlation table to {save_path}")

    return combined


def generate_manuscript_correlation_table(
    combined_df,
    save_path=None,
    bold_best=True
):
    """
    Generate a wide-format correlation table for the manuscript.

    Produces a table with:
    - Rows: Sites grouped by sequencing lab (MU-sequenced, SB-sequenced)
    - Columns: Two groups (PMMoV-normalized, ToBRFV-normalized), each with 3 pathogens

    Args:
        combined_df: DataFrame from generate_combined_correlation_table
        save_path: Optional path to save TSV file
        bold_best: If True, bold the best correlation for each pathogen

    Returns:
        DataFrame in wide format suitable for manuscript
    """
    if combined_df.empty:
        print("No correlation data to format")
        return pd.DataFrame()

    df = combined_df.copy()

    # Standardize column names (handle both old and new naming conventions)
    col_renames = {
        'r_pmmov_formatted': 'r_pmmov_fmt',
        'r_tobrfv_formatted': 'r_tobrfv_fmt',
        'n_points_pmmov': 'n_pmmov',
        'n_points_tobrfv': 'n_tobrfv',
    }
    df = df.rename(columns={k: v for k, v in col_renames.items() if k in df.columns})

    # Create formatted R values if not present
    def format_r(r, p):
        if pd.isna(r) or pd.isna(p):
            return ''
        r_str = f"{r:.2f}"
        if p < 0.001:
            return f"{r_str}***"
        elif p < 0.01:
            return f"{r_str}**"
        elif p < 0.05:
            return f"{r_str}*"
        return r_str

    if 'r_pmmov_fmt' not in df.columns:
        df['r_pmmov_fmt'] = df.apply(lambda row: format_r(row.get('r_pmmov'), row.get('p_pmmov')), axis=1)
    if 'r_tobrfv_fmt' not in df.columns:
        df['r_tobrfv_fmt'] = df.apply(lambda row: format_r(row.get('r_tobrfv'), row.get('p_tobrfv')), axis=1)

    # Find best R value for each pathogen (across all sites and normalizations)
    best_r_per_pathogen = {}
    if bold_best:
        for pathogen in df['pathogen'].unique():
            pathogen_df = df[df['pathogen'] == pathogen]
            pmmov_vals = pathogen_df['r_pmmov'].dropna().tolist()
            tobrfv_vals = pathogen_df['r_tobrfv'].dropna().tolist() if 'r_tobrfv' in pathogen_df.columns else []
            all_vals = pmmov_vals + tobrfv_vals
            if all_vals:
                best_r_per_pathogen[pathogen] = max(all_vals)

    df['display_name'] = df['site_name']

    # Get state and region for sorting
    df['state'] = df['site_name'].apply(_state_from_site_name)
    df['_region'] = df['state'].apply(get_state_region)

    region_order_map = {region: i for i, region in enumerate(REGION_ORDER)}
    df['_region_order'] = df['_region'].map(region_order_map).fillna(len(REGION_ORDER))

    # Pathogen order for columns
    pathogen_order = {'SARS-CoV-2': 0, 'Influenza A': 1, 'RSV': 2}
    df['_pathogen_order'] = df['pathogen'].map(pathogen_order)

    # Create a unique site identifier
    df['site_key'] = df['site_name']

    # Create pivot tables for formatted values
    pmmov_fmt_pivot = df.pivot_table(
        index=['site_key', 'display_name', 'seq_lab', 'state', '_region', '_region_order'],
        columns='pathogen',
        values='r_pmmov_fmt',
        aggfunc='first'
    ).reset_index()

    pmmov_r_pivot = df.pivot_table(
        index=['site_key'],
        columns='pathogen',
        values='r_pmmov',
        aggfunc='first'
    ).reset_index()

    tobrfv_fmt_pivot = df.pivot_table(
        index=['site_key'],
        columns='pathogen',
        values='r_tobrfv_fmt',
        aggfunc='first'
    ).reset_index()

    tobrfv_r_pivot = df.pivot_table(
        index=['site_key'],
        columns='pathogen',
        values='r_tobrfv',
        aggfunc='first'
    ).reset_index()

    # Rename columns
    pmmov_fmt_pivot = pmmov_fmt_pivot.rename(columns={
        'SARS-CoV-2': 'pmmov_sars_fmt', 'Influenza A': 'pmmov_flu_fmt', 'RSV': 'pmmov_rsv_fmt'
    })
    pmmov_r_pivot = pmmov_r_pivot.rename(columns={
        'SARS-CoV-2': 'pmmov_sars_r', 'Influenza A': 'pmmov_flu_r', 'RSV': 'pmmov_rsv_r'
    })
    tobrfv_fmt_pivot = tobrfv_fmt_pivot.rename(columns={
        'SARS-CoV-2': 'tobrfv_sars_fmt', 'Influenza A': 'tobrfv_flu_fmt', 'RSV': 'tobrfv_rsv_fmt'
    })
    tobrfv_r_pivot = tobrfv_r_pivot.rename(columns={
        'SARS-CoV-2': 'tobrfv_sars_r', 'Influenza A': 'tobrfv_flu_r', 'RSV': 'tobrfv_rsv_r'
    })

    # Merge all pivots
    wide_df = pmmov_fmt_pivot.merge(pmmov_r_pivot, on='site_key', how='outer')
    wide_df = wide_df.merge(tobrfv_fmt_pivot, on='site_key', how='outer')
    wide_df = wide_df.merge(tobrfv_r_pivot, on='site_key', how='outer')

    # Sort by seq_lab (MU first), then by region, then by state, then by site name
    seq_lab_order = {'MU': 0, 'SB': 1}
    wide_df['_seq_lab_order'] = wide_df['seq_lab'].map(seq_lab_order).fillna(2)
    wide_df = wide_df.sort_values(
        ['_seq_lab_order', '_region_order', 'state', 'display_name']
    )

    # Helper to bold if best
    def maybe_bold(fmt_val, r_val, pathogen):
        if not bold_best or pd.isna(r_val) or fmt_val == '' or pd.isna(fmt_val):
            return fmt_val if not pd.isna(fmt_val) else ''
        best_r = best_r_per_pathogen.get(pathogen)
        if best_r is not None and abs(r_val - best_r) < 0.001:
            import re
            match = re.match(r'(-?[\d.]+)(\**)$', str(fmt_val))
            if match:
                num_part = match.group(1)
                stars = match.group(2)
                return f"**{num_part}**{stars}"
            return f"**{fmt_val}**"
        return fmt_val

    # Create final output dataframe with proper column names
    output_df = pd.DataFrame()
    output_df['Sampling site'] = wide_df['display_name'].values
    output_df['seq_lab'] = wide_df['seq_lab'].values

    # PMMoV-normalized columns with bolding
    output_df['SARS-CoV-2 (PMMoV)'] = [
        maybe_bold(fmt, r, 'SARS-CoV-2')
        for fmt, r in zip(wide_df.get('pmmov_sars_fmt', ['']*len(wide_df)),
                         wide_df.get('pmmov_sars_r', [np.nan]*len(wide_df)))
    ]
    output_df['Influenza A (PMMoV)'] = [
        maybe_bold(fmt, r, 'Influenza A')
        for fmt, r in zip(wide_df.get('pmmov_flu_fmt', ['']*len(wide_df)),
                         wide_df.get('pmmov_flu_r', [np.nan]*len(wide_df)))
    ]
    output_df['RSV (PMMoV)'] = [
        maybe_bold(fmt, r, 'RSV')
        for fmt, r in zip(wide_df.get('pmmov_rsv_fmt', ['']*len(wide_df)),
                         wide_df.get('pmmov_rsv_r', [np.nan]*len(wide_df)))
    ]

    # ToBRFV-normalized columns with bolding
    output_df['SARS-CoV-2 (ToBRFV)'] = [
        maybe_bold(fmt, r, 'SARS-CoV-2')
        for fmt, r in zip(wide_df.get('tobrfv_sars_fmt', ['']*len(wide_df)),
                         wide_df.get('tobrfv_sars_r', [np.nan]*len(wide_df)))
    ]
    output_df['Influenza A (ToBRFV)'] = [
        maybe_bold(fmt, r, 'Influenza A')
        for fmt, r in zip(wide_df.get('tobrfv_flu_fmt', ['']*len(wide_df)),
                         wide_df.get('tobrfv_flu_r', [np.nan]*len(wide_df)))
    ]
    output_df['RSV (ToBRFV)'] = [
        maybe_bold(fmt, r, 'RSV')
        for fmt, r in zip(wide_df.get('tobrfv_rsv_fmt', ['']*len(wide_df)),
                         wide_df.get('tobrfv_rsv_r', [np.nan]*len(wide_df)))
    ]

    # Clean up NaN values - replace with empty string
    output_df = output_df.fillna('')

    if save_path:
        output_df.to_csv(save_path, sep='\t', index=False)
        print(f"Saved manuscript correlation table to {save_path}")

    return output_df


def print_manuscript_correlation_table_markdown(df):
    """Print the manuscript correlation table in markdown format."""
    print("\n" + "="*120)
    print("SUPPLEMENTARY TABLE S3: Correlation between WW-MGS and NWSS PCR")
    print("="*120 + "\n")

    # Header
    print("| Sampling site | SARS-CoV-2 | Influenza A | RSV | SARS-CoV-2 | Influenza A | RSV |")
    print("|---------------|------------|-------------|-----|------------|-------------|-----|")
    print("|               | **PMMoV-normalized** ||| **ToBRFV-normalized** |||")

    current_seq_lab = None
    for _, row in df.iterrows():
        # Add section header if seq_lab changes
        if row['seq_lab'] != current_seq_lab:
            current_seq_lab = row['seq_lab']
            lab_name = "MU-sequenced" if current_seq_lab == 'MU' else "SB-sequenced"
            print(f"| **{lab_name}** | | | | | | |")

        # Print data row
        pmmov_sars = row.get('SARS-CoV-2 (PMMoV)', '')
        pmmov_flu = row.get('Influenza A (PMMoV)', '')
        pmmov_rsv = row.get('RSV (PMMoV)', '')
        tobrfv_sars = row.get('SARS-CoV-2 (ToBRFV)', '')
        tobrfv_flu = row.get('Influenza A (ToBRFV)', '')
        tobrfv_rsv = row.get('RSV (ToBRFV)', '')

        print(f"| {row['Sampling site']} | {pmmov_sars} | {pmmov_flu} | {pmmov_rsv} | {tobrfv_sars} | {tobrfv_flu} | {tobrfv_rsv} |")

    print("\n" + "="*120 + "\n")


def generate_nwss_correlation_display_tables(
    mgs_data_dict,
    nwss_data_dict,
    site_info_dict,
    save_path=None
):
    """
    Generate formatted and raw correlation display tables for NWSS PCR comparison.

    Returns a dict with:
        'formatted': DataFrame with significance-star-annotated R values (manuscript style)
        'raw': DataFrame with separate R, p columns + median sampling frequencies
    """
    # Generate correlations for all three normalizations
    corr_dfs = {}
    for norm in ['pmmov', 'tobrfv', 'raw']:
        corr = generate_correlation_table(
            mgs_data_dict, nwss_data_dict, site_info_dict,
            normalization=norm
        )
        if not corr.empty:
            corr_dfs[norm] = corr

    if not corr_dfs:
        return {'formatted': pd.DataFrame(), 'raw': pd.DataFrame()}

    # Merge all normalizations into one long-format DataFrame
    norm_labels = {'pmmov': 'PMMoV', 'tobrfv': 'ToBRFV', 'raw': 'Raw'}
    merged = None
    for norm_key, norm_label in norm_labels.items():
        if norm_key not in corr_dfs:
            continue
        df = corr_dfs[norm_key][['pathogen', 'site_name', 'seq_lab',
                                  'r_value', 'p_value', 'n_points', 'r_formatted']].copy()
        suffix = f'_{norm_label.lower()}'
        df = df.rename(columns={
            'r_value': f'r{suffix}', 'p_value': f'p{suffix}',
            'n_points': f'n{suffix}', 'r_formatted': f'r_fmt{suffix}'
        })
        if merged is None:
            merged = df
        else:
            merge_cols = ['pathogen', 'site_name', 'seq_lab']
            merged = pd.merge(merged, df,
                             on=merge_cols, how='outer')

    # Get ordered sites
    mu_sites, sb_sites = get_ordered_sites(site_info_dict)

    pathogens_list = ['SARS-CoV-2', 'Influenza A', 'RSV']
    norms_list = ['Raw', 'PMMoV', 'ToBRFV']

    # --- Compute median sampling frequencies per (site_name, seq_lab) ---
    site_freq = {}
    for _, row in merged.drop_duplicates(['site_name', 'seq_lab']).iterrows():
        site_name_key = row['site_name']
        seq_lab_key = row['seq_lab']
        site_info = site_info_dict.get(site_name_key, {})
        sewershed_ids = site_info.get('sewershed_ids', [])
        mgs_data = mgs_data_dict.get('sars-cov-2')
        nwss_data = nwss_data_dict.get('sars-cov-2')
        if mgs_data is None or nwss_data is None:
            continue
        taxids = NWSS_PATHOGEN_MAP['sars-cov-2']['taxids']
        site_mgs = mgs_data[(mgs_data['site_name'] == site_name_key) & (mgs_data['taxid'].isin(taxids))]
        # Filter by lab for multi-lab sites
        seq_labs = site_info.get('seq_labs', [seq_lab_key])
        if len(seq_labs) > 1 and 'sequencing_lab' in site_mgs.columns:
            site_mgs = site_mgs[site_mgs['sequencing_lab'] == seq_lab_key]
        site_nwss = nwss_data[nwss_data['sewershed_id'].isin(sewershed_ids)]
        site_nwss = filter_to_dominant_source(site_nwss)
        if site_mgs.empty or site_nwss.empty:
            continue
        mgs_min, mgs_max = site_mgs['date'].min(), site_mgs['date'].max()
        nwss_min, nwss_max = site_nwss['date'].min(), site_nwss['date'].max()
        overlap_start, overlap_end = max(mgs_min, nwss_min), min(mgs_max, nwss_max)
        mgs_dates = np.sort(site_mgs[(site_mgs['date'] >= overlap_start) & (site_mgs['date'] <= overlap_end)]['date'].unique())
        pcr_dates = np.sort(site_nwss[(site_nwss['date'] >= overlap_start) & (site_nwss['date'] <= overlap_end)]['date'].unique())
        mgs_freq = float(np.median(np.diff(mgs_dates).astype('timedelta64[D]').astype(int))) if len(mgs_dates) > 1 else np.nan
        pcr_freq = float(np.median(np.diff(pcr_dates).astype('timedelta64[D]').astype(int))) if len(pcr_dates) > 1 else np.nan
        site_freq[(site_name_key, seq_lab_key)] = {'mgs_freq': mgs_freq, 'pcr_freq': pcr_freq}

    # --- Build ordered list of (site_name, seq_lab) pairs ---
    site_lab_pairs = []
    for lab_label, sites in [('MU', mu_sites), ('SB', sb_sites)]:
        for s in sites:
            site_lab_pairs.append((s, lab_label))

    # --- Build formatted table (manuscript style with significance stars) ---
    fmt_cols = ['Sampling site', 'sequencing_lab', 'N (weeks)']
    for p in pathogens_list:
        for norm in norms_list:
            fmt_cols.append(f'{p} ({norm})')

    fmt_rows = []
    for site_name_key, seq_lab_key in site_lab_pairs:
        site_data = merged[(merged['site_name'] == site_name_key) & (merged['seq_lab'] == seq_lab_key)]
        if site_data.empty:
            continue
        sc2 = site_data[site_data['pathogen'] == 'SARS-CoV-2']
        n_col = 'n_pmmov' if 'n_pmmov' in site_data.columns else 'n_tobrfv'
        n = int(sc2[n_col].iloc[0]) if not sc2.empty and pd.notna(sc2[n_col].iloc[0]) else ''
        row = {'Sampling site': site_name_key, 'sequencing_lab': seq_lab_key, 'N (weeks)': n}
        for p in pathogens_list:
            p_data = site_data[site_data['pathogen'] == p]
            for norm in norms_list:
                col = f'{p} ({norm})'
                r_col = f'r_fmt_{norm.lower()}'
                row[col] = p_data[r_col].iloc[0] if not p_data.empty and r_col in p_data.columns and pd.notna(p_data[r_col].iloc[0]) else ''
        fmt_rows.append(row)
    formatted_df = pd.DataFrame(fmt_rows)

    # --- Build raw table (separate R and p columns + frequency) ---
    raw_cols = ['Sampling site', 'sequencing_lab', 'N (weeks)']
    for p in pathogens_list:
        for norm in norms_list:
            raw_cols.extend([f'{p} R ({norm})', f'{p} p ({norm})'])
    raw_cols.extend(['Median PCR freq (days)', 'Median MGS freq (days)'])

    raw_rows = []
    for site_name_key, seq_lab_key in site_lab_pairs:
        site_data = merged[(merged['site_name'] == site_name_key) & (merged['seq_lab'] == seq_lab_key)]
        if site_data.empty:
            continue
        sc2 = site_data[site_data['pathogen'] == 'SARS-CoV-2']
        n_col = 'n_pmmov' if 'n_pmmov' in site_data.columns else 'n_tobrfv'
        n = int(sc2[n_col].iloc[0]) if not sc2.empty and pd.notna(sc2[n_col].iloc[0]) else ''
        row = {'Sampling site': site_name_key, 'sequencing_lab': seq_lab_key, 'N (weeks)': n}
        for p in pathogens_list:
            p_data = site_data[site_data['pathogen'] == p]
            for norm in norms_list:
                r_col = f'r_{norm.lower()}'
                p_col = f'p_{norm.lower()}'
                if not p_data.empty and r_col in p_data.columns and pd.notna(p_data[r_col].iloc[0]):
                    row[f'{p} R ({norm})'] = p_data[r_col].iloc[0]
                    row[f'{p} p ({norm})'] = p_data[p_col].iloc[0]
                else:
                    row[f'{p} R ({norm})'] = ''
                    row[f'{p} p ({norm})'] = ''
        freq = site_freq.get((site_name_key, seq_lab_key), {})
        pcr_f = freq.get('pcr_freq', np.nan)
        mgs_f = freq.get('mgs_freq', np.nan)
        row['Median PCR freq (days)'] = f'{pcr_f:.0f}' if not np.isnan(pcr_f) else ''
        row['Median MGS freq (days)'] = f'{mgs_f:.0f}' if not np.isnan(mgs_f) else ''
        raw_rows.append(row)
    raw_df = pd.DataFrame(raw_rows)

    if save_path:
        formatted_df.to_csv(save_path, sep='\t', index=False)
        raw_save = save_path.replace('.tsv', '_raw.tsv')
        raw_df.to_csv(raw_save, sep='\t', index=False)

    return {'formatted': formatted_df, 'raw': raw_df}


def add_seasonal_shading(ax, date_min, date_max, alpha=0.3):
    """
    Add very faint blue shading for winter months (Dec, Jan, Feb)
    and faint yellow shading for summer months (Jun, Jul, Aug).
    """
    start_year = date_min.year
    end_year = date_max.year + 1

    winter_color = '#E6F2FF'
    summer_color = '#FFF3B0'

    for year in range(start_year - 1, end_year + 1):
        # Winter spans Dec of previous year to Feb of current year
        winter_start = pd.Timestamp(f'{year}-12-01')
        winter_end = pd.Timestamp(f'{year + 1}-03-01')

        shade_start = max(winter_start, date_min)
        shade_end = min(winter_end, date_max)

        if shade_start < shade_end:
            ax.axvspan(shade_start, shade_end, color=winter_color, alpha=alpha, zorder=0)

        # Summer spans Jun to Aug
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
    mgs_filtered = mgs_data[
        (mgs_data['site_name'] == site_name) &
        (mgs_data['taxid'].isin(taxids))
    ].copy()

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

    # Filter NWSS data for this site's sewersheds
    nwss_filtered = nwss_data[nwss_data['sewershed_id'].isin(sewershed_ids)]

    if nwss_filtered.empty:
        return None, None

    # Get date ranges
    mgs_min, mgs_max = mgs_filtered['date'].min(), mgs_filtered['date'].max()
    nwss_min, nwss_max = nwss_filtered['date'].min(), nwss_filtered['date'].max()

    # Calculate overlap
    overlap_min = max(mgs_min, nwss_min)
    overlap_max = min(mgs_max, nwss_max)

    if overlap_min <= overlap_max:
        return overlap_min, overlap_max

    return None, None


def get_ordered_sites(site_info_dict):
    """
    Get (site_name, seq_lab) pairs ordered by lab (MU first), then state, then name.

    For multi-lab sites, separate entries are returned for each lab.

    Args:
        site_info_dict: Dict mapping site_name -> site info (includes seq_labs key)

    Returns:
        Tuple of (mu_sites, sb_sites) where each is a list of site_names.
        Multi-lab sites appear in BOTH lists.
    """
    if not site_info_dict:
        return [], []

    # Create DataFrame with site info — one row per (site, lab)
    sites_data = []
    for site_name, info in site_info_dict.items():
        if site_name in SITES_TO_EXCLUDE:
            continue

        state = _state_from_site_name(site_name)
        seq_labs = info.get('seq_labs', [info.get('seq_lab', 'MU')])

        for lab in seq_labs:
            sites_data.append({
                'site_name': site_name,
                'state': state,
                'seq_lab': lab,
            })

    df = pd.DataFrame(sites_data)

    if df.empty:
        return [], []

    # Sort by state and name
    df = sort_locations_by_state_and_name(df, loc_id_column='site_name')

    # Split by sequencing lab
    mu_sites = df[df['seq_lab'] == 'MU']['site_name'].tolist()
    sb_sites = df[df['seq_lab'] == 'SB']['site_name'].tolist()

    return mu_sites, sb_sites


def _build_site_info_dict(matches):
    """
    Build site_info_dict from matches DataFrame returned by load_mgs_nwss_matches().

    Groups sites by sequencing lab using sample metadata.  For multi-lab sites
    (e.g. Boston DITP sequenced by both MU and SB), ``seq_labs`` contains all
    labs.  Downstream correlation functions iterate over ``seq_labs`` to compute
    per-lab correlations.

    Returns:
        dict mapping site_name -> {sewershed_ids: [...], state: str,
                                    seq_lab: str, seq_labs: [str]}
    """
    # Load sample metadata to determine sequencing lab per site
    lib_meta = load_sample_metadata()

    site_info_dict = {}
    for site_name in matches['site_name'].unique():
        site_matches = matches[matches['site_name'] == site_name]
        sewershed_ids = []
        for _, row in site_matches.iterrows():
            sewershed_ids.extend(row['sewershed_ids'])
        state = site_matches['state'].iloc[0] if 'state' in site_matches.columns else _state_from_site_name(site_name)

        # Determine sequencing labs from metadata
        labs = ['MU']  # default
        if not lib_meta.empty and 'sequencing_lab' in lib_meta.columns:
            detected = lib_meta[lib_meta['site_name'] == site_name]['sequencing_lab'].unique()
            if len(detected) >= 1:
                labs = sorted(detected.tolist())

        site_info_dict[site_name] = {
            'sewershed_ids': sewershed_ids,
            'state': state,
            'seq_lab': labs[0],
            'seq_labs': labs,
        }
    return site_info_dict


def plot_mgs_nwss_supplementary_all(
    mgs_data_dict,
    nwss_data_dict,
    site_info_dict,
    save_path=None,
    normalization='pmmov',
    verbose=False
):
    """
    Create supplementary figure comparing MGS and NWSS data for all sites.

    Sites as rows, pathogens (SARS-CoV-2, Influenza A, RSV) as columns.

    Args:
        mgs_data_dict: dict mapping pathogen_key -> DataFrame
        nwss_data_dict: dict mapping pathogen_key -> DataFrame
        site_info_dict: dict mapping site_name -> {sewershed_ids: [...], state: str}
        save_path: Optional path to save figure
        normalization: Normalization method for MGS data. Options:
                       'pmmov' (default) - PMMoV-normalized
                       'rrna' - rRNA-normalized (divide by 1 - rRNA fraction)
                       'tobrfv' - ToBRFV-normalized
                       'raw' - raw relative abundance (no normalization)

    Returns:
        fig, axes
    """
    # Get MGS column based on normalization method
    if normalization not in NORMALIZATION_COLUMNS:
        raise ValueError(f"Unknown normalization method: {normalization}. "
                        f"Options: {list(NORMALIZATION_COLUMNS.keys())}")
    mgs_col = NORMALIZATION_COLUMNS[normalization]
    mgs_y_label = NORMALIZATION_LABELS[normalization]

    # Get ordered sites (MU first, then SB)
    mu_sites_all, sb_sites_all = get_ordered_sites(site_info_dict)
    all_sites_initial = mu_sites_all + sb_sites_all

    n_pathogens = len(PATHOGENS_TO_PLOT)

    # Collect all correlations for summary
    all_correlations = []

    # First, calculate date ranges for all sites to filter out those with no overlap
    site_date_ranges = {}
    for site_name in all_sites_initial:
        if site_name not in site_info_dict:
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

    # Filter to only sites with actual date overlap
    mu_sites = [s for s in mu_sites_all if s in site_date_ranges]
    sb_sites = [s for s in sb_sites_all if s in site_date_ranges]
    all_sites = mu_sites + sb_sites

    n_mu = len(mu_sites)
    n_sb = len(sb_sites)
    n_sites = len(all_sites)

    if n_sites == 0:
        return None, None

    # Calculate global date range across all sites for shared x-axis
    global_date_min = min(site_date_ranges[s][0] for s in all_sites)
    global_date_max = max(site_date_ranges[s][1] for s in all_sites)
    global_date_range_days = (global_date_max - global_date_min).days

    # First pass: determine which sites have PMMoV-normalized NWSS data
    # and collect y-axis ranges per subplot (based on smoothed data)
    site_has_pmmov = {}
    subplot_mgs_max = {}
    subplot_nwss_max = {}

    # Load metadata for multi-lab filtering in ylim pass
    _lib_meta_ylim = load_sample_metadata()

    for site_idx, site_name in enumerate(all_sites):
        if site_name not in site_info_dict or site_name not in site_date_ranges:
            continue

        sewershed_ids = site_info_dict[site_name]['sewershed_ids']
        date_min, date_max = site_date_ranges[site_name]

        # Determine which lab this entry corresponds to (MU section vs SB section)
        entry_lab = 'MU' if site_idx < n_mu else 'SB'
        site_seq_labs = site_info_dict[site_name].get('seq_labs',
                            [site_info_dict[site_name].get('seq_lab', 'MU')])
        is_multi_lab = len(site_seq_labs) > 1

        for col_idx, pathogen_key in enumerate(PATHOGENS_TO_PLOT):
            taxids = NWSS_PATHOGEN_MAP[pathogen_key]['taxids']
            mgs_data = mgs_data_dict.get(pathogen_key)
            nwss_data = nwss_data_dict.get(pathogen_key)

            key = (site_idx, col_idx)

            # Check if NWSS has PMMoV-normalized data for this site
            if site_name not in site_has_pmmov and nwss_data is not None:
                site_nwss_check = nwss_data[nwss_data['sewershed_id'].isin(sewershed_ids)]
                site_has_pmmov[site_name] = (site_nwss_check['pcr_target_mic_lin'].notna()).sum() > 0

            # Track MGS start date (after skip) to clip NWSS display consistently
            mgs_start_date_yaxis = None

            # Get MGS max (from smoothed data)
            if mgs_data is not None:
                site_mgs = mgs_data[
                    (mgs_data['site_name'] == site_name) &
                    (mgs_data['taxid'].isin(taxids))
                ].copy()

                # For multi-lab sites, filter to the lab for this section
                if is_multi_lab and 'sequencing_lab' in site_mgs.columns:
                    site_mgs = site_mgs[site_mgs['sequencing_lab'] == entry_lab]
                elif is_multi_lab and not site_mgs.empty:
                    lab_accessions = _lib_meta_ylim[
                        _lib_meta_ylim['sequencing_lab'] == entry_lab
                    ]['sra_accession'].unique()
                    site_mgs = site_mgs[site_mgs['sra_accession'].isin(lab_accessions)]
                if not site_mgs.empty:
                    site_mgs = site_mgs.sort_values('date')
                    # Get filtered start date for display clipping
                    site_mgs_filt = filter_timeseries_data(site_mgs)
                    if not site_mgs_filt.empty:
                        mgs_start_date_yaxis = site_mgs_filt['date'].min()
                    # Smooth on full data to avoid edge effects
                    mgs_smoothed = calculate_mmwr_smoothed_trend(
                        site_mgs, 'date', mgs_col
                    )
                    if not mgs_smoothed.empty:
                        smoothed_dates = pd.to_datetime(mgs_smoothed['date'])
                        viz_min = pd.Timestamp(mgs_start_date_yaxis) if mgs_start_date_yaxis is not None else pd.Timestamp(date_min)
                        viz_min = max(viz_min, pd.Timestamp(date_min))
                        mgs_display = mgs_smoothed[
                            (smoothed_dates >= viz_min) &
                            (smoothed_dates <= pd.Timestamp(date_max))
                        ]
                        if not mgs_display.empty:
                            subplot_mgs_max[key] = mgs_display['smoothed_values'].max() * 1.1
                    elif not site_mgs.empty:
                        subplot_mgs_max[key] = site_mgs[mgs_col].max() * 1.1

            # Get NWSS max (from MMWR smoothed data)
            nwss_clip_min = mgs_start_date_yaxis if mgs_start_date_yaxis is not None else date_min
            if nwss_data is not None:
                site_nwss = nwss_data[
                    (nwss_data['sewershed_id'].isin(sewershed_ids))
                ].copy()
                site_nwss = filter_to_dominant_source(site_nwss)
                if not site_nwss.empty:
                    has_pmmov = site_has_pmmov.get(site_name, True)
                    nwss_col = 'pcr_target_mic_lin' if has_pmmov else 'pcr_target_avg_conc_lin'
                    valid_nwss = site_nwss[site_nwss[nwss_col].notna() & (site_nwss[nwss_col] >= 0)]
                    if not valid_nwss.empty:
                        nwss_smoothed = calculate_mmwr_smoothed_trend(
                            valid_nwss, 'date', nwss_col
                        )
                        if not nwss_smoothed.empty:
                            smoothed_dates = pd.to_datetime(nwss_smoothed['date'])
                            nwss_display = nwss_smoothed[
                                (smoothed_dates >= pd.Timestamp(nwss_clip_min)) &
                                (smoothed_dates <= pd.Timestamp(date_max))
                            ]
                            if not nwss_display.empty:
                                subplot_nwss_max[key] = nwss_display['smoothed_values'].max() * 1.1
                        elif not valid_nwss.empty:
                            subplot_nwss_max[key] = valid_nwss[nwss_col].max() * 1.1

    # --- 6-column layout: pair 2 sites per row ---
    import math
    n_mu_rows = math.ceil(n_mu / 2)
    n_sb_rows = math.ceil(n_sb / 2)
    n_data_rows = n_mu_rows + n_sb_rows

    row_height = 2.2
    header_height = 1.0
    legend_height = 0.8

    total_height = legend_height + header_height + (n_mu_rows * row_height) + header_height + (n_sb_rows * row_height) + 0.5
    fig = plt.figure(figsize=(18, total_height))

    # Height ratios: [MU header] [MU rows...] [SB header] [SB rows...]
    height_ratios = ([header_height/row_height] + [1]*n_mu_rows +
                     [header_height/row_height * 0.6] + [1]*n_sb_rows)

    # Width ratios: 3 data cols + spacer + 3 data cols
    spacer_ratio = 0.15
    width_ratios = [1, 1, 1, spacer_ratio, 1, 1, 1]

    gs = GridSpec(
        n_mu_rows + n_sb_rows + 2,  # +2 for headers
        7,  # 7 columns (3 + spacer + 3)
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.35,
        wspace=0.25,
        left=0.07,
        right=0.95,
        top=0.92,
        bottom=0.06
    )

    # Map from site index to axes
    axes_map = {}

    # Data column indices in the 7-col grid: left group = [0,1,2], right group = [4,5,6]
    left_cols = [0, 1, 2]
    right_cols = [4, 5, 6]

    # Create axes for MU section (skip row 0 which is the MU header)
    for pair_row in range(n_mu_rows):
        gs_row = pair_row + 1  # +1 for MU header
        for side in range(2):  # 0=left, 1=right
            site_idx_in_section = pair_row * 2 + side
            if site_idx_in_section >= n_mu:
                break
            site_idx_in_all = site_idx_in_section  # MU sites come first in all_sites
            cols = left_cols if side == 0 else right_cols
            site_axes = []
            for pathogen_idx, gc in enumerate(cols):
                ax = fig.add_subplot(gs[gs_row, gc])
                site_axes.append(ax)
            axes_map[site_idx_in_all] = site_axes

    # Create axes for SB section
    sb_header_gs_row = 1 + n_mu_rows  # MU header + MU data rows
    for pair_row in range(n_sb_rows):
        gs_row = sb_header_gs_row + 1 + pair_row  # +1 for SB header
        for side in range(2):
            site_idx_in_section = pair_row * 2 + side
            if site_idx_in_section >= n_sb:
                break
            site_idx_in_all = n_mu + site_idx_in_section
            cols = left_cols if side == 0 else right_cols
            site_axes = []
            for pathogen_idx, gc in enumerate(cols):
                ax = fig.add_subplot(gs[gs_row, gc])
                site_axes.append(ax)
            axes_map[site_idx_in_all] = site_axes

    # Determine which site indices need x-axis labels
    bottom_site_indices = set()
    if n_sb > 0:
        last_left = max(i for i in range(0, n_sb, 2))
        bottom_site_indices.add(n_mu + last_left)
        right_indices = [i for i in range(1, n_sb, 2)]
        if right_indices:
            bottom_site_indices.add(n_mu + max(right_indices))

    # Plot each site
    # Determine sequencing lab for multi-lab site filtering
    _lib_meta = load_sample_metadata()
    for site_idx, site_name in enumerate(all_sites):
        if site_name not in site_info_dict or site_name not in site_date_ranges:
            continue
        if site_idx not in axes_map:
            continue

        sewershed_ids = site_info_dict[site_name]['sewershed_ids']
        date_min, date_max = site_date_ranges[site_name]
        has_pmmov = site_has_pmmov.get(site_name, True)

        # Determine which lab this entry corresponds to (MU section vs SB section)
        entry_lab = 'MU' if site_idx < n_mu else 'SB'
        site_seq_labs = site_info_dict[site_name].get('seq_labs',
                            [site_info_dict[site_name].get('seq_lab', 'MU')])
        is_multi_lab = len(site_seq_labs) > 1

        for col_idx, pathogen_key in enumerate(PATHOGENS_TO_PLOT):
            ax1 = axes_map[site_idx][col_idx]
            ax2 = ax1.twinx()

            pathogen_config = NWSS_PATHOGEN_MAP[pathogen_key]
            pathogen_label = pathogen_config['label']
            taxids = pathogen_config['taxids']

            mgs_data = mgs_data_dict.get(pathogen_key)
            nwss_data = nwss_data_dict.get(pathogen_key)

            # Add seasonal shading using global date range
            add_seasonal_shading(ax1, global_date_min, global_date_max)

            # Get MGS data for this site
            mgs_start_date = None
            mgs_smoothed_data = None

            if mgs_data is not None:
                site_mgs = mgs_data[
                    (mgs_data['site_name'] == site_name) &
                    (mgs_data['taxid'].isin(taxids))
                ].copy()

                # For multi-lab sites, filter to the lab for this section
                if is_multi_lab and 'sequencing_lab' in site_mgs.columns:
                    site_mgs = site_mgs[site_mgs['sequencing_lab'] == entry_lab]
                elif is_multi_lab and not site_mgs.empty:
                    lab_accessions = _lib_meta[
                        _lib_meta['sequencing_lab'] == entry_lab
                    ]['sra_accession'].unique()
                    site_mgs = site_mgs[site_mgs['sra_accession'].isin(lab_accessions)]

                if not site_mgs.empty:
                    site_mgs = site_mgs.sort_values('date')
                    color = get_location_color(site_name)

                    # Smooth full data to avoid edge effects from 5-week moving average
                    mgs_smoothed = calculate_mmwr_smoothed_trend(
                        site_mgs, 'date', mgs_col
                    )

                    if not mgs_smoothed.empty:
                        # Store for correlation (full smoothed data)
                        mgs_smoothed_data = mgs_smoothed

                        # For visualization: determine display start from filtered data
                        site_mgs_viz = filter_timeseries_data(site_mgs)
                        if not site_mgs_viz.empty:
                            mgs_start_date = site_mgs_viz['date'].min()

                            # Clip smoothed output to filtered date range for display
                            smoothed_dates = pd.to_datetime(mgs_smoothed['date'])
                            viz_min = max(pd.Timestamp(date_min), pd.Timestamp(mgs_start_date))
                            mgs_display = mgs_smoothed[
                                (smoothed_dates >= viz_min) &
                                (smoothed_dates <= pd.Timestamp(date_max))
                            ]
                            if not mgs_display.empty:
                                ax1.plot(mgs_display['date'], mgs_display['smoothed_values'],
                                        color=color, linewidth=LINE_WIDTH * 1.5,
                                        alpha=LINE_ALPHA, zorder=4)

            # Get NWSS data for this site
            nwss_llod = None
            if nwss_data is not None:
                site_nwss = nwss_data[
                    (nwss_data['sewershed_id'].isin(sewershed_ids))
                ].copy()
                site_nwss = filter_to_dominant_source(site_nwss)

                if not site_nwss.empty:
                    nwss_col = 'pcr_target_mic_lin' if has_pmmov else 'pcr_target_avg_conc_lin'

                    # Get LLOD if available
                    if 'lod_sewage' in site_nwss.columns:
                        lod_values = site_nwss['lod_sewage'].dropna()
                        if len(lod_values) > 0:
                            nwss_llod = lod_values.median()

                    site_nwss = site_nwss[
                        (site_nwss[nwss_col] >= 0) &
                        (site_nwss[nwss_col].notna())
                    ].copy()

                    if not site_nwss.empty:
                        nwss_smoothed = calculate_mmwr_smoothed_trend(
                            site_nwss, 'date', nwss_col
                        )

                        nwss_color = '#666666'

                        if not nwss_smoothed.empty:
                            # Clip to display range for plotting
                            nwss_clip_min = mgs_start_date if mgs_start_date is not None else date_min
                            nwss_smoothed_dates = pd.to_datetime(nwss_smoothed['date'])
                            nwss_display = nwss_smoothed[
                                (nwss_smoothed_dates >= pd.Timestamp(nwss_clip_min)) &
                                (nwss_smoothed_dates <= pd.Timestamp(date_max))
                            ]
                            if not nwss_display.empty:
                                ax2.plot(nwss_display['date'], nwss_display['smoothed_values'],
                                        color=nwss_color, linewidth=LINE_WIDTH * 1.5,
                                        linestyle='--', alpha=LINE_ALPHA, zorder=2)

                            # Calculate correlation
                            if mgs_smoothed_data is not None:
                                import warnings
                                merged = pd.merge(
                                    mgs_smoothed_data[['mmwr_year', 'mmwr_week', 'smoothed_values']].rename(
                                        columns={'smoothed_values': 'mgs_value'}),
                                    nwss_smoothed[['mmwr_year', 'mmwr_week', 'smoothed_values']].rename(
                                        columns={'smoothed_values': 'nwss_value'}),
                                    on=['mmwr_year', 'mmwr_week'],
                                    how='inner'
                                )
                                if len(merged) >= 3:
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        r, p = spearmanr(merged['mgs_value'], merged['nwss_value'])
                                    # Collect for summary
                                    seq_lab = site_info_dict[site_name].get('seq_lab', 'MU')
                                    all_correlations.append({
                                        'pathogen': pathogen_label,
                                        'site_name': site_name,
                                        'seq_lab': seq_lab,
                                        'comparison_type': 'NWSS',
                                        'comparison_source': 'NWSS ddPCR',
                                        'r_value': r,
                                        'p_value': p,
                                        'n_points': len(merged)
                                    })
                                    # Display correlation with significance stars
                                    r_text = format_r_with_stars(r, p)

                                    # Determine placement
                                    placement_override = R_TEXT_PLACEMENT_OVERRIDES.get((site_name, pathogen_key))
                                    use_top_right = False
                                    if placement_override == 'top_left':
                                        use_top_right = False
                                    elif placement_override == 'top_right':
                                        use_top_right = True
                                    elif mgs_smoothed_data is not None and not mgs_smoothed_data.empty:
                                        global_range_days = (global_date_max - global_date_min).days
                                        left_cutoff = pd.Timestamp(global_date_min) + pd.Timedelta(days=global_range_days * 0.40)
                                        max_idx = mgs_smoothed_data['smoothed_values'].idxmax()
                                        peak_date = pd.Timestamp(mgs_smoothed_data.loc[max_idx, 'date'])
                                        if peak_date <= left_cutoff:
                                            use_top_right = True

                                    if use_top_right:
                                        ax1.text(0.97, 0.95, r_text, transform=ax1.transAxes,
                                                ha='right', va='top', fontsize=FONT_SIZE_SMALL)
                                    else:
                                        ax1.text(0.03, 0.95, r_text, transform=ax1.transAxes,
                                                ha='left', va='top', fontsize=FONT_SIZE_SMALL)

            # Formatting
            key = (site_idx, col_idx)
            mgs_max = subplot_mgs_max.get(key)
            nwss_max = subplot_nwss_max.get(key)
            if mgs_max is not None and mgs_max > 0:
                y1_bottom = -mgs_max * 0.05
                ax1.set_ylim(bottom=y1_bottom, top=mgs_max)
            else:
                ax1.set_ylim(bottom=-0.05, top=1)
            if nwss_max is not None and nwss_max > 0:
                y2_padding = nwss_max * 0.05
                ax2.set_ylim(bottom=-y2_padding, top=nwss_max)
            else:
                ax2.set_ylim(bottom=-0.05, top=1)

            # Set x limits to global date range for shared axis
            buffer = pd.Timedelta(days=7)
            ax1.set_xlim(global_date_min - buffer, global_date_max + buffer)

            # Determine which side (left=0 or right=1) this site is on
            if site_idx < n_mu:
                side = site_idx % 2
            else:
                side = (site_idx - n_mu) % 2

            # Column titles: appear on the first row of MU section for both left and right groups
            is_first_mu_left = (site_idx == 0)
            is_first_mu_right = (site_idx == 1 and n_mu >= 2)
            if is_first_mu_left or is_first_mu_right:
                ax1.set_title(pathogen_label, fontsize=FONT_SIZE_LARGE, pad=30)

            # Site name above row (horizontal, left-aligned above first pathogen column)
            if col_idx == 0:
                display_name = SITE_SHORT_NAMES.get(site_name, site_name)
                ax1.text(0.0, 1.12, display_name, transform=ax1.transAxes,
                        fontsize=FONT_SIZE_SMALL, ha='left', va='bottom')

            ax1.set_ylabel('')
            ax2.set_ylabel('')

            # X-axis formatting - only show on bottom row of SB section
            if site_idx in bottom_site_indices:
                import matplotlib.dates as mdates
                if global_date_range_days > 730:
                    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                elif global_date_range_days > 365:
                    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
                else:
                    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax1.set_xticklabels([])

            # Grid and styling
            ax1.grid(True, alpha=0.3, zorder=1)
            ax1.set_axisbelow(True)
            ax1.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)

            ax1.tick_params(axis='both', labelsize=FONT_SIZE_SMALL)
            ax2.tick_params(axis='both', labelsize=FONT_SIZE_SMALL)

            # Format y-axis with clean formatting and fewer ticks
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
                """Remove duplicate ticks that round to the same integer."""
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

                labels = [formatter(loc) for loc in locs]
                seen = set()
                unique_locs = []
                for loc, label in zip(locs, labels):
                    if label not in seen:
                        seen.add(label)
                        unique_locs.append(loc)

                if axis == 'y':
                    ax.yaxis.set_major_locator(FixedLocator(unique_locs))
                else:
                    ax.xaxis.set_major_locator(FixedLocator(unique_locs))

            # MGS axis
            ax1.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            formatter1 = CleanScalarFormatter(useMathText=False)
            ax1.yaxis.set_major_formatter(formatter1)
            ax1.yaxis.get_offset_text().set_fontsize(FONT_SIZE_SMALL)
            remove_duplicate_int_ticks(ax1, 'y')

            # NWSS axis
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            formatter2 = CleanScalarFormatter(useMathText=False)
            ax2.yaxis.set_major_formatter(formatter2)
            ax2.yaxis.get_offset_text().set_fontsize(FONT_SIZE_SMALL)
            remove_duplicate_int_ticks(ax2, 'y')

    # --- Section headers and decorations ---

    # MU section header
    if n_mu > 0:
        mu_first_ax = axes_map[0][0].get_position()
        mu_rightmost_x1 = max(axes_map[i][-1].get_position().x1
                              for i in range(n_mu) if i in axes_map)
        mu_leftmost_x0 = mu_first_ax.x0

        header_y = mu_first_ax.y1 + 0.030
        header_height_fig = 0.016
        header_x0 = mu_leftmost_x0
        header_width = mu_rightmost_x1 - mu_leftmost_x0

        rect_mu = Rectangle(
            (header_x0, header_y),
            header_width,
            header_height_fig,
            transform=fig.transFigure,
            facecolor='#CCCCCC',
            edgecolor='none',
            zorder=1,
            alpha=0.5
        )
        fig.patches.append(rect_mu)

        fig.text(
            (mu_leftmost_x0 + mu_rightmost_x1) / 2,
            header_y + header_height_fig / 2,
            'MU-sequenced',
            fontsize=FONT_SIZE_LARGE,
            weight='normal',
            ha='center',
            va='center',
            zorder=10
        )

    # SB section header
    if n_sb > 0:
        sb_first_idx = n_mu
        sb_first_ax = axes_map[sb_first_idx][0].get_position()
        sb_rightmost_x1 = max(axes_map[i][-1].get_position().x1
                               for i in range(n_mu, n_mu + n_sb) if i in axes_map)
        sb_leftmost_x0 = sb_first_ax.x0

        header_height_fig = 0.016

        # Position header centered between last MU row and first SB row
        if n_mu > 0:
            last_mu_idx = n_mu - 1
            last_mu_bottom = axes_map[last_mu_idx][0].get_position().y0
            gap_center = (last_mu_bottom + sb_first_ax.y1) / 2
            header_y = gap_center - header_height_fig / 2
        else:
            header_y = sb_first_ax.y1 + 0.024

        # Flush with leftmost y1 spine and rightmost y2 spine
        if n_mu > 0:
            header_x0 = min(mu_leftmost_x0, sb_leftmost_x0)
            header_right = max(mu_rightmost_x1, sb_rightmost_x1)
        else:
            header_x0 = sb_leftmost_x0
            header_right = sb_rightmost_x1
        header_width = header_right - header_x0

        rect_sb = Rectangle(
            (header_x0, header_y),
            header_width,
            header_height_fig,
            transform=fig.transFigure,
            facecolor='#CCCCCC',
            edgecolor='none',
            zorder=1,
            alpha=0.5
        )
        fig.patches.append(rect_sb)

        fig.text(
            header_x0 + header_width / 2,
            header_y + header_height_fig / 2,
            'SB-sequenced',
            fontsize=FONT_SIZE_LARGE,
            weight='normal',
            ha='center',
            va='center',
            zorder=10
        )

    # Add legend at top
    from matplotlib.lines import Line2D

    line_handles = [
        Line2D([0], [0], color='#666666', linewidth=LINE_WIDTH * 1.5, label='CASPER wastewater sequencing'),
        Line2D([0], [0], color='#666666', linewidth=LINE_WIDTH * 1.5, linestyle='--', label='Wastewater PCR'),
    ]

    winter_patch = mpatches.Patch(color='#E6F2FF', alpha=0.5, label='Winter (Dec-Feb)')
    summer_patch = mpatches.Patch(color='#FFF3B0', alpha=0.5, label='Summer (Jun-Aug)')
    seasonal_handles = [winter_patch, summer_patch]

    fig.legend(handles=line_handles, loc='upper left', bbox_to_anchor=(0.10, 0.956),
               ncol=1, fontsize=FONT_SIZE_LARGE, frameon=False)

    fig.legend(handles=seasonal_handles, loc='upper right', bbox_to_anchor=(0.95, 0.956),
               ncol=1, fontsize=FONT_SIZE_LARGE, frameon=False)

    # Shared axis labels
    mgs_y_label_short = 'CASPER wastewater sequencing signal'
    fig.text(0.04, 0.5, mgs_y_label_short, va='center', ha='center',
             rotation=90, fontsize=FONT_SIZE_LARGE)
    fig.text(0.98, 0.5, 'Wastewater PCR signal', va='center', ha='center',
             rotation=270, fontsize=FONT_SIZE_LARGE)

    # Save correlation TSV
    tsv_path = None
    if save_path:
        data_dir = Path(save_path).parent.parent / 'tables'
        tsv_path = str(data_dir / 'mgs_nwss_supplementary_all_correlations.tsv')
    print_correlation_summary(all_correlations, save_tsv_path=tsv_path, verbose=verbose)

    if save_path:
        save_figure(fig, save_path)

    return fig, axes_map
