#!/usr/bin/env python3
"""
Bar plot of population served by sampling site.

Shows population served (on log scale) for each site, ordered by population size.
Sites without population data are shown as empty bars with hatching pattern.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import plotting config and data loaders
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *
from data_loaders import load_site_metadata


def parse_population_value(pop_str):
    """
    Parse population served value from string.

    Handles:
    - Comma-separated numbers: "1,234,567" -> 1234567
    - Semicolon-separated numbers (Boise): "148,300; 122,600" -> 270900 (sum)
    - NaN values

    Args:
        pop_str: Population string from CSV

    Returns:
        Float value of population, or NaN if not available
    """
    if pd.isna(pop_str):
        return np.nan

    pop_str = str(pop_str).strip()

    # Handle semicolon-separated values (Boise dual influent)
    if ';' in pop_str:
        parts = pop_str.split(';')
        values = []
        for part in parts:
            clean = part.strip().replace(',', '')
            try:
                values.append(float(clean))
            except ValueError:
                continue
        if values:
            return sum(values)
        else:
            return np.nan

    # Handle single value with commas
    clean = pop_str.replace(',', '')
    try:
        return float(clean)
    except ValueError:
        return np.nan


def load_population_data():
    """
    Load population served data from site_metadata.csv.

    Returns:
        DataFrame with columns: site_name, map_label, state, population_served, has_data
    """
    site_meta = load_site_metadata()

    # Parse population values
    site_meta['population_served'] = site_meta['population_served'].apply(parse_population_value)

    # Add flag for whether data is available
    site_meta['has_data'] = ~site_meta['population_served'].isna()

    # Use city column as map_label
    site_meta['map_label'] = site_meta['city']

    return site_meta[['site_name', 'map_label', 'state', 'population_served', 'has_data']].copy()


def plot_population_served(pop_data, save_path=None):
    """
    Create bar plot of population served by site.

    Sites are ordered by population size (largest first).
    Uses log scale for y-axis.
    Sites without data shown as empty bars with hatching.

    Args:
        pop_data: DataFrame from load_population_data()
        save_path: Optional path to save figure

    Returns:
        fig, ax: Matplotlib figure and axis
    """
    df = pop_data.copy()

    # Separate sites with and without data
    df_with_data = df[df['has_data']].copy()
    df_no_data = df[~df['has_data']].copy()

    # Sort sites with data by population (descending)
    df_with_data = df_with_data.sort_values('population_served', ascending=False)

    # Sort sites without data by state, then alphabetically by site_name
    if len(df_no_data) > 0:
        df_no_data = df_no_data.sort_values(['state', 'site_name'])

    # Combine: sites with data first (by population), then sites without data
    df = pd.concat([df_with_data, df_no_data], ignore_index=True)

    # Create figure
    fig, ax = create_figure(figsize=(14, 6))

    # Prepare data for plotting
    x_positions = np.arange(len(df))
    populations = df['population_served'].values
    has_data = df['has_data'].values

    # Create bars
    bars = []
    for i, (pop, has_data_flag, site_name) in enumerate(zip(populations, has_data, df['site_name'])):
        color = get_location_color(site_name)

        if has_data_flag:
            bar = ax.bar(i, pop, color=color, alpha=0.8,
                        edgecolor='black', linewidth=0.8)
        else:
            bar = ax.bar(i, 1000, color='none', alpha=0.8,
                        edgecolor=color, linewidth=1.5, hatch='///')
        bars.append(bar)

    # Set log scale for y-axis
    ax.set_yscale('log')

    # Set y-axis limits
    valid_pops = populations[has_data]
    if len(valid_pops) > 0:
        ymin = valid_pops.min() * 0.5
        ymax = valid_pops.max() * 2
        ax.set_ylim(ymin, ymax)

    # Labels and formatting
    ax.set_xlabel('', fontsize=FONT_SIZE_LARGE)
    ax.set_ylabel('Population served', fontsize=FONT_SIZE_LARGE)

    # X-axis: use site_name for tick labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df['site_name'], rotation=45, ha='right', fontsize=FONT_SIZE_BASE)

    # Y-axis: format with thousands separator
    from matplotlib.ticker import FuncFormatter
    def format_pop(x, p):
        if x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.0f}K'
        else:
            return f'{x:.0f}'
    ax.yaxis.set_major_formatter(FuncFormatter(format_pop))
    ax.tick_params(axis='y', labelsize=FONT_SIZE_BASE)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Grid for readability
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Add legend for sites without data and total population
    from matplotlib.patches import Rectangle

    total_pop = df[df['has_data']]['population_served'].sum()

    legend_elements = []

    if not has_data.all():
        legend_elements.append(
            Rectangle((0, 0), 1, 1, facecolor='none',
                     edgecolor='gray', linewidth=1.5, hatch='///', label='No data')
        )

    # Format total population
    if total_pop >= 1e6:
        total_str = f'Total: {total_pop/1e6:.1f}M'
    elif total_pop >= 1e3:
        total_str = f'Total: {total_pop/1e3:.0f}K'
    else:
        total_str = f'Total: {total_pop:.0f}'

    legend_elements.append(
        Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='none',
                 label=total_str)
    )

    ax.legend(handles=legend_elements, loc='upper right',
             frameon=False, fontsize=FONT_SIZE_BASE)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig, ax


def get_figure_caption():
    """
    Get figure caption text for notebook.

    Returns:
        String with figure caption
    """
    caption = (
        "Population served by each sampling site, ordered by population size. "
        "Sites without population data are shown as empty bars with diagonal hatching. "
        "For Boise (Boise RWF, ID), population represents the sum of two influent streams "
        "(West Boise WWTP: 148,300; Lander Street WWTP: 122,600; Total: 270,900)."
    )
    return caption
