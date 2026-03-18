#!/usr/bin/env python3
"""
Generate summary table and visualizations for sample collection across sites.

Creates comprehensive summary statistics including:
- Number of samples per site
- Date range of sampling
- Median sampling interval
- Mean and standard deviation of read pairs per sample
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Import plotting config and data loaders
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *
from data_loaders import (
    load_all_site_data, load_site_metadata, load_sample_metadata,
    load_sample_age_data, load_nwss_site_matches,
)


def calculate_sampling_interval(dates):
    """
    Calculate median sampling interval in days.

    Args:
        dates: List or Series of datetime objects

    Returns:
        Median interval in days (float)
    """
    if len(dates) < 2:
        return np.nan

    dates_sorted = sorted(dates)
    intervals = [(dates_sorted[i+1] - dates_sorted[i]).days
                 for i in range(len(dates_sorted) - 1)]

    return np.median(intervals)


def create_site_summary_table(site_data, output_file=None, separate_by_seq_lab=False):
    """
    Create comprehensive summary table for all sites.

    Args:
        site_data: DataFrame from load_all_site_data()
        output_file: Optional path to save CSV
        separate_by_seq_lab: If True, create separate tables for MU and SB sites

    Returns:
        DataFrame with summary statistics per site
        (or tuple of (mu_df, sb_df) if separate_by_seq_lab=True)
    """
    # Merge with libraries_metadata to get sequencing_lab
    libs_meta = load_sample_metadata()
    if 'sequencing_lab' not in site_data.columns:
        merge_cols = ['sra_accession'] if 'sra_accession' in site_data.columns else ['site_name', 'date']
        lab_info = libs_meta[['sra_accession', 'sequencing_lab']].drop_duplicates()
        site_data = site_data.merge(lab_info, on='sra_accession', how='left')

    # Load site metadata for city/state
    site_meta = load_site_metadata()
    meta_lookup = site_meta.set_index('site_name')

    # Group by (site_name, sequencing_lab) for multi-lab sites
    if 'sequencing_lab' in site_data.columns:
        site_groups = site_data.groupby(['site_name', 'sequencing_lab'])
    else:
        site_groups = [(name, 'Unknown', grp) for name, grp in site_data.groupby('site_name')]
        site_groups = site_data.groupby('site_name')

    summary_rows = []

    for group_key, group in site_groups:
        if isinstance(group_key, tuple):
            site_name, seq_lab = group_key
        else:
            site_name = group_key
            seq_lab = group['sequencing_lab'].mode().iloc[0] if 'sequencing_lab' in group.columns else 'Unknown'

        # Get metadata
        if site_name in meta_lookup.index:
            meta_row = meta_lookup.loc[site_name]
            city = meta_row.get('city', site_name)
            state = meta_row.get('state', 'Unknown')
        else:
            city = site_name
            state = 'Unknown'

        # Basic counts
        n_samples = len(group)

        # Date range
        start_date = group['date'].min()
        end_date = group['date'].max()
        date_range_days = (end_date - start_date).days

        # Sampling interval
        median_interval = calculate_sampling_interval(group['date'])

        # Read pair statistics
        mean_read_pairs = group['total_read_pairs'].mean()
        std_read_pairs = group['total_read_pairs'].std()
        median_read_pairs = group['total_read_pairs'].median()

        summary_rows.append({
            'site_name': site_name,
            'city': city,
            'state': state,
            'seq_lab': seq_lab,
            'n_samples': n_samples,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'date_range_days': date_range_days,
            'median_sampling_interval_days': median_interval,
            'mean_read_pairs': mean_read_pairs,
            'std_read_pairs': std_read_pairs,
            'median_read_pairs': median_read_pairs,
        })

    summary_df = pd.DataFrame(summary_rows)

    # Sort by state, then by number of samples
    summary_df = summary_df.sort_values(
        ['state', 'n_samples'],
        ascending=[True, False]
    ).reset_index(drop=True)

    if separate_by_seq_lab:
        # Split into MU and SB tables
        mu_df = summary_df[summary_df['seq_lab'] == 'MU'].copy().reset_index(drop=True)
        sb_df = summary_df[summary_df['seq_lab'].isin(['SB', 'BCL'])].copy().reset_index(drop=True)

        # Remove seq_lab column for display
        mu_df = mu_df.drop(columns=['seq_lab'])
        sb_df = sb_df.drop(columns=['seq_lab'])

        # Save if requested
        if output_file is not None:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            mu_path = output_path.parent / (output_path.stem + '_mu' + output_path.suffix)
            mu_df.to_csv(mu_path, index=False)
            print(f"Saved MU summary table to {mu_path}")

            sb_path = output_path.parent / (output_path.stem + '_sb' + output_path.suffix)
            sb_df.to_csv(sb_path, index=False)
            print(f"Saved SB summary table to {sb_path}")

        return mu_df, sb_df
    else:
        # Remove seq_lab column for display if not separating
        summary_df = summary_df.drop(columns=['seq_lab'])

        # Save if requested
        if output_file is not None:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(output_path, index=False)
            print(f"Saved summary table to {output_path}")

        return summary_df


def format_summary_table_for_display(summary_df):
    """
    Format summary table for nice display (markdown, latex, etc).

    Args:
        summary_df: DataFrame from create_site_summary_table()

    Returns:
        Formatted DataFrame with better column names and number formatting
    """
    display_df = summary_df.copy()

    # Create combined mean +/- std column in millions
    def format_mean_std(row):
        mean = row['mean_read_pairs']
        std = row['std_read_pairs']
        if pd.notna(mean) and pd.notna(std):
            mean_M = mean / 1e6
            std_M = std / 1e6
            return f'{mean_M:.0f}M +/- {std_M:.0f}M'
        elif pd.notna(mean):
            mean_M = mean / 1e6
            return f'{mean_M:.0f}M'
        else:
            return 'N/A'

    display_df['read_pairs_mean_std'] = display_df.apply(format_mean_std, axis=1)

    # Select and reorder columns
    display_df = display_df[[
        'site_name',
        'city',
        'state',
        'n_samples',
        'start_date',
        'end_date',
        'date_range_days',
        'median_sampling_interval_days',
        'read_pairs_mean_std',
    ]].copy()

    # Rename columns for display
    display_df.columns = [
        'Site Name',
        'City',
        'State',
        'N Samples',
        'Start Date',
        'End Date',
        'Date Range (days)',
        'Median Interval (days)',
        'Read Pairs (Mean +/- Std)',
    ]

    # Format median interval
    display_df['Median Interval (days)'] = display_df['Median Interval (days)'].apply(
        lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A'
    )

    return display_df


def load_sample_age_by_site():
    """
    Load median sample age (turnaround time) per site.

    Returns:
        Dict mapping site_name to median sample_age (days)
    """
    age_data = load_sample_age_data()

    if age_data.empty or 'sample_age' not in age_data.columns:
        return {}

    age_data['sample_age'] = pd.to_numeric(age_data['sample_age'], errors='coerce')
    valid = age_data[age_data['sample_age'].notna()]

    site_median_ages = {}
    for site_name, group in valid.groupby('site_name'):
        site_median_ages[site_name] = group['sample_age'].median()

    return site_median_ages


def parse_population_value(pop_str):
    """
    Parse population served value from string.

    Handles comma-separated numbers and semicolon-separated (Boise).
    """
    if pd.isna(pop_str):
        return np.nan

    pop_str = str(pop_str).strip()

    if ';' in pop_str:
        parts = pop_str.split(';')
        values = []
        for part in parts:
            clean = part.strip().replace(',', '')
            try:
                values.append(float(clean))
            except ValueError:
                continue
        return sum(values) if values else np.nan

    clean = pop_str.replace(',', '')
    try:
        return float(clean)
    except ValueError:
        return np.nan


def generate_table1_sampling_sites(site_data, save_path=None):
    """
    Generate Table 1: Summary of CASPER sampling sites.

    Computes start date and median sampling interval from actual data.
    Includes location type, sample type, population served from metadata.
    Organized by region (Northeast, Midwest, South, West), then state, then site.

    Args:
        site_data: DataFrame from load_all_site_data()
        save_path: Optional path to save CSV

    Returns:
        DataFrame with Table 1 data
    """
    # Load metadata
    site_meta = load_site_metadata()
    meta_lookup = site_meta.set_index('site_name')

    # Load NWSS sewershed IDs
    nwss_matches = load_nwss_site_matches()
    nwss_lookup = nwss_matches.set_index('site_name')['sewershed_id'].to_dict()

    # Load median sample age (turnaround time) per site
    site_median_ages = load_sample_age_by_site()

    # Get sequencing_lab per sample
    libs_meta = load_sample_metadata()
    if 'sequencing_lab' not in site_data.columns:
        lab_info = libs_meta[['sra_accession', 'sequencing_lab']].drop_duplicates()
        site_data = site_data.merge(lab_info, on='sra_accession', how='left')

    # Group site_data by (site_name, sequencing_lab) for multi-lab sites
    if 'sequencing_lab' in site_data.columns:
        site_groups = site_data.groupby(['site_name', 'sequencing_lab'])
    else:
        site_groups = site_data.groupby('site_name')

    rows = []
    for group_key, group in site_groups:
        if isinstance(group_key, tuple):
            site_name, seq_lab = group_key
        else:
            site_name = group_key
            seq_lab = group['sequencing_lab'].mode().iloc[0] if 'sequencing_lab' in group.columns else 'Unknown'

        # Get start date, end date, and median interval
        dates = group['date'].sort_values()
        start_date = dates.min()
        end_date = dates.max()

        # Calculate median sampling interval
        if len(dates) > 1:
            intervals = dates.diff().dropna().dt.days
            median_interval = int(round(intervals.median()))
        else:
            median_interval = None

        # Calculate median read depth (in millions)
        median_read_pairs = group['total_read_pairs'].median()

        # Normalize: BCL -> SB
        if seq_lab == 'BCL':
            seq_lab = 'SB'

        # Get metadata from site_metadata
        if site_name in meta_lookup.index:
            meta_row = meta_lookup.loc[site_name]
            state = meta_row.get('state', 'Unknown')
            loc_type = meta_row.get('loc_type', 'WWTP')
            sample_type = meta_row.get('sample_type', '24h composite')
            region = meta_row.get('region', get_state_region(state))

            # Population served
            pop_raw = meta_row.get('population_served', '')
            pop_val = parse_population_value(pop_raw)
            if pd.isna(pop_val):
                if loc_type in ['Hospital (service hatch)', 'Hospital (manhole)', 'Airport (manhole)']:
                    pop_served_display = 'n/a'
                else:
                    pop_served_display = '-'
            else:
                pop_served_display = str(pop_raw)
        else:
            state = 'Unknown'
            loc_type = 'WWTP'
            sample_type = '24h composite'
            region = 'Unknown'
            pop_served_display = '-'

        # Format median read pairs in millions
        if pd.notna(median_read_pairs):
            median_rp_display = f"{median_read_pairs / 1e6:.1f}M"
        else:
            median_rp_display = '-'

        # Get NWSS sewershed ID
        nwss_id = nwss_lookup.get(site_name)
        nwss_id_display = str(int(nwss_id)) if nwss_id is not None and pd.notna(nwss_id) else '-'

        # Get median turnaround time (sample age)
        median_turnaround = site_median_ages.get(site_name)
        if median_turnaround is not None:
            median_turnaround_display = int(round(median_turnaround))
        else:
            median_turnaround_display = '-'

        rows.append({
            'region': region,
            'state': state,
            'Sampling site': site_name,
            'NWSS sewershed ID': nwss_id_display,
            'Location type': loc_type,
            'Sample type': sample_type,
            'Approx. population served': pop_served_display,
            'Sequencing partner': seq_lab,
            'Sample collection start date': start_date.strftime('%Y-%m-%d'),
            'Sample collection end date': end_date.strftime('%Y-%m-%d'),
            'Number of samples': len(group),
            'Median read pairs': median_rp_display,
            'Median sampling interval (days)': median_interval,
            'Median turnaround (days)': median_turnaround_display,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No data for Table 1")
        return df

    # Sort by region order, then state alphabetically, then site name
    region_order_map = {region: i for i, region in enumerate(REGION_ORDER)}
    df['_region_order'] = df['region'].map(region_order_map).fillna(len(REGION_ORDER))
    df = df.sort_values(['_region_order', 'state', 'Sampling site'])

    # Drop internal columns
    df = df.drop(columns=['_region_order'])

    if save_path:
        sep = "\t" if str(save_path).endswith(".tsv") else ","
        df.to_csv(save_path, index=False, sep=sep)

    return df


def print_table1_markdown(df):
    """Print Table 1 in markdown format with region headers."""
    print("\n" + "="*140)
    print("TABLE 1: SUMMARY OF CASPER SAMPLING SITES")
    print("="*140 + "\n")

    # Print header
    cols = ['Sampling site', 'Location type', 'Sample type', 'Approx. population served',
            'Sequencing partner', 'Sample collection start date', 'Sample collection end date',
            'Number of samples', 'Median read pairs', 'Median sampling interval (days)', 'Median turnaround (days)']
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")

    current_region = None
    for _, row in df.iterrows():
        # Print region header if changed
        if row['region'] != current_region:
            current_region = row['region']
            print(f"| **{current_region}** | | | | | | | | | | |")

        # Print row
        values = [str(row[col]) if pd.notna(row[col]) else '' for col in cols]
        print("| " + " | ".join(values) + " |")

    print("\n" + "="*140 + "\n")
