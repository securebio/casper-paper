#!/usr/bin/env python3
"""Build the sampling-site summary table."""

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


