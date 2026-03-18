#!/usr/bin/env python3
"""
Map visualization of sampling sites across the US.

Shows sampling locations on a US map with Hawaii inset (if Hawaii sites exist).
Sites are colored by location and labeled by city. Multiple sites at the
same coordinates are jittered slightly for visibility.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Import plotting config and data loaders
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *
from data_loaders import load_site_metadata


def load_site_coordinates():
    """
    Load site coordinates from site_metadata.csv.

    Returns:
        DataFrame with columns: site_name, lat, lon, map_label, state, city
    """
    coords = load_site_metadata()

    # Derive map_label from city column
    coords['map_label'] = coords['city']

    return coords[['site_name', 'lat', 'lon', 'map_label', 'state', 'city']].copy()


def add_jitter(coords_df, jitter_amount=0.3):
    """
    Add random jitter to sites with duplicate coordinates.

    Args:
        coords_df: DataFrame with lat, lon columns
        jitter_amount: Amount of jitter in degrees

    Returns:
        DataFrame with jittered coordinates
    """
    df = coords_df.copy()

    # Find duplicate coordinates
    coord_groups = df.groupby(['lat', 'lon'])

    for (lat, lon), group in coord_groups:
        if len(group) > 1:
            # Add jitter to duplicates
            n_sites = len(group)
            np.random.seed(42)  # For reproducibility
            jitter_lat = np.random.uniform(-jitter_amount, jitter_amount, n_sites)
            jitter_lon = np.random.uniform(-jitter_amount, jitter_amount, n_sites)

            df.loc[group.index, 'lat'] = lat + jitter_lat
            df.loc[group.index, 'lon'] = lon + jitter_lon

    return df


def plot_us_sampling_map(coords_df, save_path=None):
    """
    Plot US map with sampling sites using cartopy.

    Shows continental US as main map with Hawaii as separate inset
    (only if Hawaii sites exist in the data).
    Sites are colored by location and labeled by city name (one label per city).
    Minimalist style: white background, black national borders, light gray state lines.

    Special handling:
    - Chicago sites: consolidated to one point per map_label
    - Southern California and Central Oklahoma: larger, diffuse marker for location uncertainty

    Args:
        coords_df: DataFrame from load_site_coordinates()
        save_path: Optional path to save figure

    Returns:
        fig, axes: Matplotlib figure and axes (main, inset)
    """
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Custom label offset positions (lat_offset, lon_offset) for each location
    # Positive lat_offset = north, Positive lon_offset = east
    LABEL_OFFSETS = {
        'Boston, MA': (0.95, 0.0),
        'Chicago, IL': (0.95, 0.0),
        'Columbia, MO': (-0.9, 5.5),
        'Kansas City, MO': (-0.8, -6.35),
        'Boise, ID': (0.8, 0.0),
        'Ottumwa, IA': (0.85, -1.5),
        'Milan, MO': (-0.3, -4.35),
        'Monett, MO': (-0.9, 4.6),
        'Riverside, CA': (-2.7, -4.1),
        'Ontario, CA': (-0.4, -4.3),
        'Palo Alto, CA': (-2.7, -4.1),
        'Southern California, CA': (0.4, 9.1),
        'Sacramento, CA': (0.8, 0.0),
        'Miami, FL': (0.8, 1.0),
        'New York, NY': (-1.9, 0.0),
        'Central Oklahoma, OK': (-2.6, 0.0),
    }

    df = coords_df.copy()

    # Consolidate to one point per map_label
    df = df.groupby('map_label', as_index=False).first()

    # Separate continental US and Hawaii
    df_conus = df[df['state'] != 'Hawaii'].copy()
    df_hawaii = df[df['state'] == 'Hawaii'].copy()

    # Create figure
    fig = plt.figure(figsize=(16, 7))

    # Use Albers Equal Area projection for US
    projection = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5,
                                      standard_parallels=(29.5, 45.5))
    ax_main = plt.axes(projection=projection)

    # Set extent for continental US
    ax_main.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

    # Add minimalist map features - US only
    import cartopy.io.shapereader as shpreader

    states_shp = shpreader.natural_earth(resolution='10m',
                                          category='cultural',
                                          name='admin_1_states_provinces')

    us_state_geoms = []
    for record in shpreader.Reader(states_shp).records():
        if record.attributes['admin'] == 'United States of America':
            us_state_geoms.append(record.geometry)

    ax_main.add_geometries(us_state_geoms, ccrs.PlateCarree(),
                          facecolor='white', edgecolor='lightgray',
                          linewidth=0.5, zorder=1)

    # Get US country boundary for outline
    countries_shp = shpreader.natural_earth(resolution='10m',
                                            category='cultural',
                                            name='admin_0_countries')

    us_country_geom = None
    for record in shpreader.Reader(countries_shp).records():
        if record.attributes['ADMIN'] == 'United States of America':
            us_country_geom = record.geometry
            break

    if us_country_geom:
        ax_main.add_geometries([us_country_geom], ccrs.PlateCarree(),
                              facecolor='none', edgecolor='black',
                              linewidth=1.5, zorder=2)

    # Turn off frame and ticks
    ax_main.spines['geo'].set_visible(False)
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    # Plot continental US sites
    for idx, row in df_conus.iterrows():
        site_name = row['site_name']
        lat = row['lat']
        lon = row['lon']
        map_label = row['map_label']
        color = get_location_color(site_name)

        # Special handling for Southern California and Central Oklahoma: larger, diffuse marker
        if map_label in ['Southern California, CA', 'Central Oklahoma, OK']:
            for size, alpha_val in [(1500, 0.07), (1300, 0.08), (1100, 0.1), (900, 0.12),
                                     (750, 0.14), (600, 0.17), (450, 0.21), (350, 0.26), (250, 0.32)]:
                ax_main.scatter(lon, lat, color=color, s=size, alpha=alpha_val,
                               edgecolors='none', zorder=2.5,
                               transform=ccrs.PlateCarree())
            ax_main.scatter(lon, lat, color=color, s=200, alpha=0.42,
                           edgecolors='none', zorder=3,
                           transform=ccrs.PlateCarree())
        else:
            ax_main.scatter(lon, lat, color=color, s=200, alpha=0.8,
                           edgecolors='black', linewidths=1.5, zorder=3,
                           transform=ccrs.PlateCarree())

    # Add labels for continental US (one per unique map_label)
    labels_added = set()
    for idx, row in df_conus.iterrows():
        label = row['map_label']
        if label not in labels_added:
            label_sites = df_conus[df_conus['map_label'] == label]
            mean_lat = label_sites['lat'].mean()
            mean_lon = label_sites['lon'].mean()

            lat_offset, lon_offset = LABEL_OFFSETS.get(label, (0.8, 0.0))

            ax_main.text(mean_lon + lon_offset, mean_lat + lat_offset, label,
                        fontsize=FONT_SIZE_LARGE,
                        ha='center', va='bottom', zorder=4,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 alpha=0.7, edgecolor='none'),
                        transform=ccrs.PlateCarree())
            labels_added.add(label)

    # Create Hawaii inset if there are Hawaii sites
    if len(df_hawaii) > 0:
        hawaii_projection = ccrs.AlbersEqualArea(central_longitude=-157, central_latitude=20,
                                                  standard_parallels=(19, 22))
        ax_inset = inset_axes(ax_main, width="25%", height="25%", loc='lower left',
                             bbox_to_anchor=(-0.01, -0.01, 1, 1),
                             bbox_transform=ax_main.transAxes,
                             axes_class=cartopy.mpl.geoaxes.GeoAxes,
                             axes_kwargs=dict(projection=hawaii_projection))

        ax_inset.set_extent([-160.5, -154.5, 18.5, 22.5], crs=ccrs.PlateCarree())

        hawaii_geom = None
        for record in shpreader.Reader(states_shp).records():
            if record.attributes.get('name') == 'Hawaii':
                hawaii_geom = record.geometry
                break

        if hawaii_geom:
            ax_inset.add_geometries([hawaii_geom], ccrs.PlateCarree(),
                                   facecolor='white', edgecolor='black',
                                   linewidth=1.5, zorder=1)

        ax_inset.spines['geo'].set_edgecolor('black')
        ax_inset.spines['geo'].set_linewidth(1.0)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])

        for idx, row in df_hawaii.iterrows():
            site_name = row['site_name']
            lat = row['lat']
            lon = row['lon']
            color = get_location_color(site_name)

            ax_inset.scatter(lon, lat, color=color, s=200, alpha=0.8,
                           edgecolors='black', linewidths=1.5, zorder=3,
                           transform=ccrs.PlateCarree())

        for idx, row in df_hawaii.iterrows():
            label = row['map_label']
            if label not in labels_added:
                label_sites = df_hawaii[df_hawaii['map_label'] == label]
                mean_lat = label_sites['lat'].mean()
                mean_lon = label_sites['lon'].mean()

                lat_offset, lon_offset = LABEL_OFFSETS.get(label, (0.3, 0.0))

                ax_inset.text(mean_lon + lon_offset, mean_lat + lat_offset, label,
                            fontsize=FONT_SIZE_SMALL,
                            ha='center', va='bottom', zorder=4,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                     alpha=0.7, edgecolor='none'),
                            transform=ccrs.PlateCarree())
                labels_added.add(label)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig, ax_main


def plot_site_legend(coords_df, save_path=None):
    """
    Create a standalone legend figure with each site getting its own handle.

    Uses circles with black outline matching the map markers.
    Shows ALL sites individually (not just one per city).

    Layout with section titles:
    - SB-sequenced
    - MU-sequenced
    - NYC sites
    - CHI-C (airport) site

    Args:
        coords_df: DataFrame from load_site_coordinates()
        save_path: Optional path to save figure

    Returns:
        fig: Matplotlib figure
    """
    from matplotlib.lines import Line2D
    from data_loaders import load_sample_metadata

    df = coords_df.copy()

    # Get sequencing lab info
    libs_meta = load_sample_metadata()
    site_labs = libs_meta.groupby('site_name')['sequencing_lab'].first().to_dict()

    # Load full site metadata (not just what's in coords_df, which is deduplicated)
    site_meta = load_site_metadata()

    # Categorize sites into groups
    mu_sites = []
    chi_c_sites = []
    nyc_sites = []
    sb_sites = []

    for _, row in site_meta.iterrows():
        site_name = row['site_name']
        lab = site_labs.get(site_name, 'Unknown')

        if 'NYC' in site_name:
            nyc_sites.append(row)
        elif 'CHI-C' in site_name:
            chi_c_sites.append(row)
        elif lab in ['SB', 'BCL']:
            sb_sites.append(row)
        elif 'Boston' in site_name:
            # Boston sites may appear in both MU and SB
            mu_sites.append(row)
            if site_name not in [r['site_name'] for r in sb_sites]:
                sb_sites.append(row)
        else:
            mu_sites.append(row)

    # Sort each group by region, state, site_name
    def sort_key(row):
        site_name = row['site_name']
        state = row['state']
        region = get_state_region(state)
        region_order = REGION_ORDER.index(region) if region in REGION_ORDER else len(REGION_ORDER)
        return (region_order, state, site_name)

    mu_sites = sorted(mu_sites, key=sort_key)
    chi_c_sites = sorted(chi_c_sites, key=sort_key)
    nyc_sites = sorted(nyc_sites, key=sort_key)
    sb_sites = sorted(sb_sites, key=sort_key)

    def create_handle_label(row):
        site_name = row['site_name']
        color = get_location_color(site_name)
        handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                       markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                       linestyle='None')
        return handle, site_name

    # Create handles and labels for each group
    sb_handles_labels = [create_handle_label(r) for r in sb_sites]
    mu_handles_labels = [create_handle_label(r) for r in mu_sites]
    nyc_handles_labels = [create_handle_label(r) for r in nyc_sites]
    chi_c_handles_labels = [create_handle_label(r) for r in chi_c_sites]

    # Layout with section titles
    column_titles = ['SB-sequenced', 'MU-sequenced', None, None]
    column_data = [sb_handles_labels, mu_handles_labels, nyc_handles_labels, chi_c_handles_labels]

    total_entries = sum(len(d) for d in column_data)
    num_sections = sum(1 for d in column_data if d)

    fig_width = 4
    fig_height = max(2, (total_entries + num_sections * 1.5) * 0.35)

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.axis('off')

    all_handles = []
    all_labels = []

    for title, data in zip(column_titles, column_data):
        if data:
            if all_handles:
                all_handles.append(plt.Line2D([0], [0], color='none', marker='None', linestyle='None'))
                all_labels.append('')

            if title is not None:
                all_handles.append(plt.Line2D([0], [0], color='none', marker='None', linestyle='None'))
                all_labels.append(title)

            handles, labels = zip(*data)
            all_handles.extend(handles)
            all_labels.extend(labels)

    legend = ax.legend(all_handles, all_labels, loc='upper left', ncol=1,
                      frameon=False, fontsize=FONT_SIZE_LARGE,
                      handletextpad=0.5)

    valid_titles = [t for t in column_titles if t is not None]
    for text in legend.get_texts():
        if text.get_text() in valid_titles:
            text.set_fontweight('bold')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=600, transparent=True)

    return fig
