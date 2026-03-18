#!/usr/bin/env python3
"""
Shared plotting configuration for CASPER manuscript figures.

This module provides consistent styling, colors, and formatting for all
figures. Site colors are assigned by state, with shading variation within
each state for multi-site regions.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd

# ============================================================================
# FONT AND STYLE CONFIGURATION
# ============================================================================

FONT_FAMILY = 'sans-serif'
FONT_SANS_SERIF = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
FONT_SIZE_BASE = 14
FONT_SIZE_SMALL = 12
FONT_SIZE_LARGE = 18
FONT_WEIGHT = 'normal'

DPI = 300

# ============================================================================
# COLOR SCHEME - STATE-BASED (tab10 palette)
# ============================================================================

_tab10 = plt.cm.tab10.colors


def _rgb_to_hex(rgb):
    """Convert RGB tuple (0-1 range) to hex string."""
    return mcolors.rgb2hex(rgb)


STATE_COLORS = {
    'Missouri': _rgb_to_hex(_tab10[3]),       # red
    'Massachusetts': _rgb_to_hex(_tab10[0]),  # blue
    'California': _rgb_to_hex(_tab10[1]),     # orange
    'Florida': _rgb_to_hex(_tab10[4]),        # purple
    'Illinois': _rgb_to_hex(_tab10[2]),       # green
    'New York': _rgb_to_hex(_tab10[8]),       # yellow-green
    'Iowa': _rgb_to_hex(_tab10[5]),           # brown
    'Idaho': _rgb_to_hex(_tab10[6]),          # pink
    'Oklahoma': '#008B8B',                     # dark cyan/teal
}


def _lighten_color(color, amount=0.3):
    """Lighten a color by mixing with white."""
    c = mcolors.to_rgb(color)
    return tuple(min(1, c[i] + (1 - c[i]) * amount) for i in range(3))


def _darken_color(color, amount=0.3):
    """Darken a color by mixing with black."""
    c = mcolors.to_rgb(color)
    return tuple(c[i] * (1 - amount) for i in range(3))


# Sites grouped by state, sorted alphabetically by site_name.
# Shading lightens alphabetically within each state.
_SITES_BY_STATE = {
    'California': [
        'Ontario, CA',
        'Palo Alto RWQCP, CA',
        'Riverside WQCP, CA',
        'Sacramento, CA',
        'Southern California, CA',
    ],
    'Florida': [
        'Miami-Dade CDWWTP, FL',
    ],
    'Idaho': [
        'Boise RWF, ID',
    ],
    'Illinois': [
        'Chicago (CHI-A), IL',
        'Chicago (CHI-B), IL',
        'Chicago (CHI-C), IL',
        'Chicago (CHI-D1), IL',
        'Chicago (CHI-D2), IL',
        'Chicago (CHI-D3), IL',
    ],
    'Iowa': [
        'Ottumwa WPCF, IA',
    ],
    'Massachusetts': [
        'Boston DITP North, MA',
        'Boston DITP South, MA',
    ],
    'Missouri': [
        'Columbia WWTP, MO',
        'Kansas City Blue River WWTP, MO',
        'Kansas City Westside WWTP, MO',
        'Milan WWTF, MO',
        'Monett WWTP, MO',
    ],
    'New York': [
        'NYC (Hospital A), NY',
        'NYC (Hospital B), NY',
        'NYC (Hospital C), NY',
        'NYC (Hospital D), NY',
    ],
    'Oklahoma': [
        'Central Oklahoma (OK-A), OK',
        'Central Oklahoma (OK-B), OK',
    ],
}

# State abbreviation suffix -> state name (for fallback color lookup)
_ABBREV_TO_STATE = {
    'CA': 'California', 'FL': 'Florida', 'ID': 'Idaho',
    'IL': 'Illinois', 'IA': 'Iowa', 'MA': 'Massachusetts', 'MO': 'Missouri',
    'NY': 'New York', 'OK': 'Oklahoma',
}


def _generate_location_colors():
    """Generate site_name -> color map with alphabetical lightening within states."""
    colors = {}
    for state, sites in _SITES_BY_STATE.items():
        base_color = STATE_COLORS[state]
        n = len(sites)
        for i, site in enumerate(sites):
            if n <= 1:
                colors[site] = base_color
            else:
                lighten_amount = (i / (n - 1)) * 0.4
                colors[site] = _rgb_to_hex(_lighten_color(base_color, lighten_amount))
    return colors


LOCATION_COLORS = _generate_location_colors()


def get_location_color(site_name):
    """Get color for a site_name, with fallback to state color."""
    if site_name in LOCATION_COLORS:
        return LOCATION_COLORS[site_name]

    # Try metadata lookup
    state = _state_from_site_name(site_name)
    if state != 'Unknown' and state in STATE_COLORS:
        return STATE_COLORS[state]

    return STATE_COLORS['Missouri']  # fallback


def get_state_color(state):
    """Get color for a state."""
    return STATE_COLORS.get(state, STATE_COLORS['Missouri'])


# ============================================================================
# TAXONOMIC COMPOSITION COLORS
# ============================================================================

TAXONOMIC_COLORS = {
    'ribosomal': '#8ed3c7',
    'unclassified': '#ffffb4',
    'archaea': '#bfbadb',
    'bacteria': '#fb8072',
    'eukaryota': '#81b1d3',
    'virus': '#fdb462',
}


def get_taxonomic_colors():
    """Get taxonomic category colors."""
    return TAXONOMIC_COLORS.copy()


VIRUS_HOST_COLORS = {
    'vertebrate': '#d95f02',
    'invertebrate': '#7570b3',
    'bacteriophage': '#1b9e77',
    'bacteria': '#1b9e77',
    'plant': '#66a61e',
    'metagenome': '#377eb8',
    'other': '#a6761d',
    'unknown': '#999999',
    'vertebrate_infecting': '#d95f02',
    'vertebrate_uncertain': '#e6ab02',
    'non_vertebrate': '#1b9e77',
}


def get_virus_host_colors():
    """Get virus host category colors."""
    return VIRUS_HOST_COLORS.copy()


# ============================================================================
# PLOT ELEMENT STYLING
# ============================================================================

LINE_WIDTH = 2.5
LINE_WIDTH_THIN = 1.5
LINE_WIDTH_THICK = 3.0

SCATTER_SIZE = 20
SCATTER_SIZE_SMALL = 10
SCATTER_SIZE_LARGE = 40

SCATTER_ALPHA = 0.6
LINE_ALPHA = 0.8

SMOOTH_WINDOW = 7

# ============================================================================
# DATE FORMATTING FOR TIME SERIES
# ============================================================================


def format_date_axis(ax, date_range_days=None):
    """Format x-axis for time series plots."""
    if date_range_days is not None:
        if date_range_days <= 90:
            locator = mdates.WeekdayLocator(interval=1)
        elif date_range_days <= 365:
            locator = mdates.MonthLocator()
        elif date_range_days <= 850:
            locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
        else:
            locator = mdates.MonthLocator(interval=6)
    else:
        locator = mdates.MonthLocator()

    formatter = mdates.DateFormatter('%b %Y')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


# ============================================================================
# LEGEND CONFIGURATION
# ============================================================================

LEGEND_FONTSIZE = FONT_SIZE_SMALL
LEGEND_FRAMEON = False
LEGEND_LOC = 'best'
LEGEND_BBOX_TO_ANCHOR = None


def setup_legend_outside(ax, ncol=1, loc='upper left', bbox_to_anchor=(1.02, 1)):
    """Place legend outside plot area."""
    ax.legend(
        fontsize=LEGEND_FONTSIZE,
        frameon=LEGEND_FRAMEON,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
    )


def setup_legend_below(ax, ncol=3):
    """Place legend below plot in multiple columns."""
    ax.legend(
        fontsize=LEGEND_FONTSIZE,
        frameon=LEGEND_FRAMEON,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=ncol,
    )


def setup_legend_above(ax, ncol=3):
    """Place legend above plot in multiple columns."""
    ax.legend(
        fontsize=LEGEND_FONTSIZE,
        frameon=LEGEND_FRAMEON,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
    )


# ============================================================================
# PANEL LABELS
# ============================================================================


def add_panel_labels(fig, axes, labels=None, x_offset=-0.01, y_offset=0.12,
                     fontsize=None):
    """Add panel labels (Nature style: lowercase bold)."""
    if labels is None:
        labels = [chr(ord('a') + i) for i in range(len(axes))]
    if fontsize is None:
        fontsize = FONT_SIZE_LARGE + 4

    for ax, label in zip(axes, labels):
        bbox = ax.get_position()
        fig.text(bbox.x0 + x_offset, bbox.y1 + y_offset, label,
                 fontsize=fontsize, fontweight='bold',
                 ha='right', va='bottom')


# ============================================================================
# FIGURE INITIALIZATION
# ============================================================================


def init_plotting_style():
    """Initialize matplotlib with consistent style settings."""
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.sans-serif': FONT_SANS_SERIF,
        'font.size': FONT_SIZE_BASE,
        'font.weight': FONT_WEIGHT,
        'axes.labelsize': FONT_SIZE_BASE,
        'axes.titlesize': FONT_SIZE_LARGE,
        'axes.labelweight': FONT_WEIGHT,
        'axes.titleweight': FONT_WEIGHT,
        'xtick.labelsize': FONT_SIZE_SMALL,
        'ytick.labelsize': FONT_SIZE_SMALL,
        'legend.fontsize': LEGEND_FONTSIZE,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': LINE_WIDTH,
        'patch.linewidth': 0.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })

    sns.set_style("ticks", {
        "axes.edgecolor": "0.15",
        "axes.linewidth": 0.8,
        "grid.color": "0.85",
    })


def create_figure(figsize=(8, 6), no_title=True):
    """Create a figure with standard settings."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def create_figure_grid(nrows, ncols, figsize=None, sharex=False, sharey=False):
    """Create a grid of subplots with standard settings."""
    if figsize is None:
        figsize = (ncols * 4, nrows * 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=sharex, sharey=sharey)
    return fig, axes


# ============================================================================
# LOCATION SORTING
# ============================================================================

US_REGIONS = {
    'Northeast': [
        'Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'New Jersey',
        'New York', 'Pennsylvania', 'Rhode Island', 'Vermont',
    ],
    'Midwest': [
        'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Michigan', 'Minnesota',
        'Missouri', 'Nebraska', 'North Dakota', 'Ohio', 'South Dakota',
        'Wisconsin',
    ],
    'South': [
        'Alabama', 'Arkansas', 'Delaware', 'District of Columbia', 'Florida',
        'Georgia', 'Kentucky', 'Louisiana', 'Maryland', 'Mississippi',
        'North Carolina', 'Oklahoma', 'South Carolina', 'Tennessee', 'Texas',
        'Virginia', 'West Virginia',
    ],
    'West': [
        'Alaska', 'Arizona', 'California', 'Colorado', 'Hawaii', 'Idaho',
        'Montana', 'Nevada', 'New Mexico', 'Oregon', 'Utah', 'Washington',
        'Wyoming',
    ],
}

STATE_TO_REGION = {}
for _region, _states in US_REGIONS.items():
    for _state in _states:
        STATE_TO_REGION[_state] = _region

REGION_ORDER = ['Northeast', 'Midwest', 'South', 'West']


def get_state_region(state):
    """Get the US Census region for a state."""
    return STATE_TO_REGION.get(state, 'Unknown')


_SITE_META_CACHE = None


def _get_site_meta():
    """Return site_metadata indexed by site_name (cached)."""
    global _SITE_META_CACHE
    if _SITE_META_CACHE is None:
        from data_loaders import load_site_metadata
        _SITE_META_CACHE = load_site_metadata().set_index('site_name')
    return _SITE_META_CACHE


def _state_from_site_name(site_name):
    """Look up full state name for a site_name from site metadata."""
    meta = _get_site_meta()
    if site_name in meta.index:
        return meta.loc[site_name, 'state']
    # Fallback: parse ", XX" suffix
    if ', ' in site_name:
        abbrev = site_name.rsplit(', ', 1)[-1].strip()
        return _ABBREV_TO_STATE.get(abbrev, 'Unknown')
    return 'Unknown'


def sort_locations_by_state_and_name(df, loc_id_column='site_name',
                                     use_paper_names=False, **kwargs):
    """Sort locations by region, then state, then site name.

    Args:
        df: DataFrame containing location data.
        loc_id_column: Name of the column containing site names (default: 'site_name').
        use_paper_names: Ignored (kept for API compatibility).

    Returns:
        DataFrame sorted by region, state, and site name.
    """
    if df.empty:
        return df

    df_sorted = df.copy()

    if loc_id_column not in df_sorted.columns:
        raise ValueError(f"Column '{loc_id_column}' not found in DataFrame")

    # Derive state if not present
    if 'state' not in df_sorted.columns:
        df_sorted['state'] = df_sorted[loc_id_column].apply(_state_from_site_name)

    df_sorted['_region'] = df_sorted['state'].apply(get_state_region)
    region_order_map = {r: i for i, r in enumerate(REGION_ORDER)}
    df_sorted['_region_order'] = df_sorted['_region'].map(region_order_map).fillna(len(REGION_ORDER))

    df_sorted = df_sorted.sort_values(
        ['_region_order', 'state', loc_id_column],
        ascending=[True, True, True],
    )

    df_sorted = df_sorted.drop(columns=['_region', '_region_order'])
    return df_sorted


def get_sorted_location_order(locations, **kwargs):
    """Get site names sorted by region, state, and name."""
    if len(locations) == 0:
        return list(locations)

    df = pd.DataFrame({'site_name': list(locations)})
    df_sorted = sort_locations_by_state_and_name(df, loc_id_column='site_name')
    return df_sorted['site_name'].tolist()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def apply_smooth(data, window=SMOOTH_WINDOW):
    """Apply rolling average smoothing (DEPRECATED — use calculate_mmwr_smoothed_trend)."""
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    return data.rolling(window=window, center=True, min_periods=1).mean()


def calculate_mmwr_smoothed_trend(df, date_col: str, value_col: str,
                                   smooth_window: int = 5):
    """Convert data to MMWR weeks with geometric mean aggregation and
    centered moving average.

    Methodology:
    - Data is aggregated to MMWR epidemiological weeks using geometric mean
      (arithmetic mean of log1p-transformed values, then expm1)
    - A centered moving average produces smoothed time series
    - Values are plotted at the MMWR week midpoint (Wednesday)

    Args:
        df: DataFrame with time series data.
        date_col: Name of the date column.
        value_col: Name of the value column to smooth.
        smooth_window: Weeks for centered moving average (default: 5).

    Returns:
        DataFrame with date_col, smoothed_values, mmwr_year, mmwr_week.
    """
    import epiweeks

    df_clean = df[[date_col, value_col]].copy().dropna().sort_values(date_col)

    if len(df_clean) < 2:
        return pd.DataFrame()

    df_clean['date'] = pd.to_datetime(df_clean[date_col])

    mmwr_data = []
    for _, row in df_clean.iterrows():
        epi_week = epiweeks.Week.fromdate(row['date'])
        mmwr_data.append({
            'date': row['date'],
            'mmwr_year': epi_week.year,
            'mmwr_week': epi_week.week,
            'value': row[value_col],
        })

    mmwr_df = pd.DataFrame(mmwr_data)
    mmwr_df['log_value'] = np.log1p(mmwr_df['value'])

    weekly_agg = mmwr_df.groupby(['mmwr_year', 'mmwr_week']).agg(
        log_value=('log_value', 'mean')
    ).reset_index()

    weekly_agg = weekly_agg.sort_values(['mmwr_year', 'mmwr_week'])

    weekly_agg['date'] = weekly_agg.apply(
        lambda row: epiweeks.Week(int(row['mmwr_year']),
                                   int(row['mmwr_week'])).startdate()
        + pd.Timedelta(days=3),
        axis=1,
    )

    weekly_agg['smoothed_log_values'] = weekly_agg['log_value'].rolling(
        window=smooth_window, center=True, min_periods=1
    ).mean()

    weekly_agg['smoothed_values'] = np.expm1(weekly_agg['smoothed_log_values'])

    return pd.DataFrame({
        date_col: weekly_agg['date'],
        'smoothed_values': weekly_agg['smoothed_values'],
        'mmwr_year': weekly_agg['mmwr_year'],
        'mmwr_week': weekly_agg['mmwr_week'],
    })


# ============================================================================
# TIME SERIES FILTERING
# ============================================================================

# Sites sequenced by both MU and SB labs
MULTI_LAB_SITES = [
    'Boston DITP North, MA',
    'Boston DITP South, MA',
]

# Drop all samples before a cutoff date (sparse early sampling)
TIMESERIES_TRUNCATE_BEFORE = {
    'NYC (Hospital C), NY': pd.Timestamp('2025-10-01'),
}

# Drop first sample for a site. Values are either None (drop regardless of lab)
# or a lab string (drop only for that lab).
TIMESERIES_SKIP_FIRST_SAMPLE = {
    'Boston DITP North, MA': 'MU',
    'Boston DITP South, MA': 'MU',
    'NYC (Hospital B), NY': None,
}


def filter_timeseries_data(df, loc_id_col='site_name', date_col='date',
                           lab_col='sequencing_lab'):
    """Filter DataFrame for time series plots.

    - NYC (Hospital C): Samples before Oct 2025 dropped.
    - Boston DITP North/South: First MU sample dropped.
    - NYC (Hospital B): First sample dropped.

    Args:
        df: DataFrame with site_name and date columns.
        loc_id_col: Column with site names (default: 'site_name').
        date_col: Column with dates (default: 'date').
        lab_col: Column with sequencing lab (default: 'sequencing_lab').
            Used for lab-specific first-sample drops. Ignored if not present.

    Returns:
        Filtered DataFrame.
    """
    df = df.copy()
    has_lab = lab_col in df.columns

    # Truncate sites before cutoff dates
    for site, cutoff in TIMESERIES_TRUNCATE_BEFORE.items():
        mask = (df[loc_id_col] == site) & (pd.to_datetime(df[date_col]) < cutoff)
        df = df[~mask]

    # Drop all rows for the first sample date from specific sites
    for site, lab in TIMESERIES_SKIP_FIRST_SAMPLE.items():
        site_mask = df[loc_id_col] == site
        if lab is not None and has_lab:
            site_mask = site_mask & (df[lab_col] == lab)
        if site_mask.any():
            site_data = df[site_mask]
            first_date = site_data[date_col].min()
            remaining_dates = site_data[site_data[date_col] != first_date]
            if len(remaining_dates) > 0:
                drop_mask = site_mask & (df[date_col] == first_date)
                df = df[~drop_mask]

    return df


def format_sci_notation(ax, axis='y', scilimits=(0, 0)):
    """Format axis to use scientific notation."""
    if axis in ['y', 'both']:
        ax.ticklabel_format(style='scientific', axis='y', scilimits=scilimits)
    if axis in ['x', 'both']:
        ax.ticklabel_format(style='scientific', axis='x', scilimits=scilimits)


def save_figure(fig, filepath, dpi=DPI, bbox_inches='tight', transparent=False):
    """Save figure with consistent settings."""
    from pathlib import Path
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches,
                transparent=transparent)
    print(f"Saved figure to {filepath}")


# ============================================================================
# SRA / MANUSCRIPT CONSTANTS
# ============================================================================

STATE_ABBREVIATIONS = {
    'California': 'CA',
    'Florida': 'FL',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Iowa': 'IA',
    'Massachusetts': 'MA',
    'Missouri': 'MO',
    'New York': 'NY',
    'Oklahoma': 'OK',
}

CASPER_COLOR = '#E64B35'
OTHER_COLOR = '#CCCCCC'


def format_bp(x, pos=None):
    """Format base pairs with human-readable units (for axis formatters)."""
    if x == 0:
        return '0'
    elif x >= 1e15:
        return f'{x/1e15:.0f} Pb'
    elif x >= 1e12:
        return f'{x/1e12:.0f} Tb'
    elif x >= 1e9:
        return f'{x/1e9:.0f} Gb'
    elif x >= 1e6:
        return f'{x/1e6:.0f} Mb'
    else:
        return f'{x:.0f}'


# ============================================================================
# INITIALIZE STYLE ON IMPORT
# ============================================================================

init_plotting_style()
