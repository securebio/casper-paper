#!/usr/bin/env python3
"""
Map visualization of sampling sites across the US.

Shows sampling locations on a map of the continental US. Sites are coloured
by location, labelled by city, and shaped by what is being sampled. Sites
sharing a city centroid are spread into a small cluster so each is visible.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
from matplotlib.font_manager import FontProperties

# Import plotting config and data loaders
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *
from data_loaders import load_site_metadata


def load_site_coordinates():
    """
    Load site coordinates from site_metadata.csv.

    Returns:
        DataFrame with columns: site_name, lat, lon, map_label, state, city,
        loc_type, labs
    """
    coords = load_site_metadata()

    # Derive map_label from city column
    coords['map_label'] = coords['city']

    # Which laboratory processed each site. Boston DITP North and South were
    # split between both, so this is a list, not a scalar.
    from data_loaders import load_sample_metadata
    labs = (load_sample_metadata()
            .groupby('site_name')['sequencing_lab']
            .agg(lambda col: sorted(set(col))))
    coords['labs'] = coords['site_name'].map(labs)
    coords['labs'] = coords['labs'].apply(
        lambda v: list(v) if isinstance(v, (list, tuple)) else ['MU'])

    return coords[['site_name', 'lat', 'lon', 'map_label', 'state', 'city',
                   'loc_type', 'labs']].copy()


# Marker shape encodes what is being sampled. Coordinates are city centroids,
# so shape is the only thing distinguishing e.g. the Chicago airport site from
# the Chicago treatment plants sharing that point.
LOC_TYPE_MARKERS = {
    'WWTP': 'o',
    'Hospital': '^',
    'Airport': 's',
}
# Triangles and squares read smaller than a circle of equal area, so nudge.
MARKER_SIZE_SCALE = {'o': 1.0, '^': 1.25, 's': 0.85}
MARKER_SIZE = 170

# Cities that asked not to disclose a precise location: drawn with a diffuse
# halo instead of a crisp point.
DIFFUSE_LABELS = ['Southern California, CA', 'Central Oklahoma, OK']

# Offset radius in degrees for co-located sites. Cities holding several sites
# of differing type get a wider spread so each shape stays readable.
CLUSTER_RADIUS = 0.3
CLUSTER_RADIUS_BY_CITY = {
    'Chicago, IL': 0.42,
    'New York, NY': 0.42,
    'Boston, MA': 0.30,
    'Kansas City, MO': 0.16,
}

# Legend font, kept apart from the city labels so the shape key reads as
# annotation rather than as another place name. Plain 'Avenir' rather than
# 'Avenir Next': the latter is registered only at weight 700, so matplotlib
# renders it bold and it overpowers the map labels.
LEGEND_FONT_FAMILY = 'Avenir'
# Same typeface, registered at weight 700; 'Avenir' carries no bold face.
LEGEND_TITLE_FONT_FAMILY = 'Avenir Next'


def loc_type_marker(loc_type):
    """Map a site's loc_type ('Hospital (manhole)', 'WWTP', ...) to a marker."""
    for prefix, marker in LOC_TYPE_MARKERS.items():
        if str(loc_type).startswith(prefix):
            return marker
    return 'o'


def draw_site_marker(ax, lon, lat, marker, labs, transform, zorder=5,
                     alpha=0.85, edgecolor='black', edgewidth=1.5):
    """Draw one site, coloured by the laboratory that processed it.

    Sites split between both laboratories get a two-tone marker, MU on the
    left and SB on the right. scatter() cannot do this -- fillstyle is a
    Line2D property -- so markers go through plot(), whose markersize is a
    diameter in points rather than scatter's area.
    """
    from matplotlib.markers import MarkerStyle

    size = np.sqrt(MARKER_SIZE * MARKER_SIZE_SCALE[marker])
    common = dict(linestyle='none', markersize=size, markeredgecolor=edgecolor,
                  markeredgewidth=edgewidth, alpha=alpha, zorder=zorder,
                  transform=transform)

    if len(labs) == 1:
        ax.plot(lon, lat, marker=marker,
                markerfacecolor=LAB_COLORS[labs[0]], **common)
        return

    for lab, fill in zip(labs, ('left', 'right')):
        ax.plot(lon, lat, marker=MarkerStyle(marker, fillstyle=fill),
                markerfacecolor=LAB_COLORS[lab], **common)


def cluster_offsets(n, lat, radius=CLUSTER_RADIUS):
    """Deterministic (dlat, dlon) offsets spreading n co-located sites.

    Sites within a city share one centroid coordinate, so these offsets are a
    legibility device, not geography: a fixed layout is used rather than random
    jitter so the figure is reproducible and reads as deliberate. Longitude is
    widened by 1/cos(lat) so the arrangement stays circular once projected.

    n=1 sits at the centroid, n=2 forms a horizontal pair, and larger groups
    form a ring starting at the top. Offsets are deliberately small: markers
    are meant to overlap, showing that a city holds several sites without
    implying they sit that far apart.
    """
    lon_scale = 1.0 / max(np.cos(np.radians(lat)), 0.1)

    if n <= 1:
        return [(0.0, 0.0)]
    if n == 2:
        return [(0.0, -radius * lon_scale), (0.0, radius * lon_scale)]

    # From five sites up, seat one in the middle: a bare ring of that many
    # points grows wide enough to collide with neighbouring city labels.
    centre = [(0.0, 0.0)] if n >= 5 else []
    n_ring = n - len(centre)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n_ring, endpoint=False)
    ring = [(radius * np.sin(a), radius * lon_scale * np.cos(a)) for a in angles]
    return centre + ring


def add_cluster_offsets(coords_df, radius=CLUSTER_RADIUS):
    """Add lat_plot/lon_plot columns spreading co-located sites within a city.

    lat/lon are left untouched so labels and location-uncertainty halos can
    still be anchored to the city centroid.
    """
    df = coords_df.copy()
    df['lat_plot'] = df['lat']
    df['lon_plot'] = df['lon']

    for label, group in df.groupby('map_label'):
        if len(group) <= 1:
            continue
        group_radius = CLUSTER_RADIUS_BY_CITY.get(label, radius)
        # Sort for a stable ring order regardless of input row order.
        idx = group.sort_values('site_name').index
        offsets = cluster_offsets(len(idx), group['lat'].mean(), radius=group_radius)
        for i, (dlat, dlon) in zip(idx, offsets):
            df.loc[i, 'lat_plot'] = df.loc[i, 'lat'] + dlat
            df.loc[i, 'lon_plot'] = df.loc[i, 'lon'] + dlon

    return df


def plot_us_sampling_map(coords_df, save_path=None):
    """
    Plot US map with sampling sites using cartopy.

    Sites are colored by location and labeled by city name (one label per city),
    with marker shape indicating location type.
    Minimalist style: white background, black national borders, light gray state lines.

    Special handling:
    - Chicago sites: consolidated to one point per map_label
    - Southern California and Central Oklahoma: larger, diffuse marker for location uncertainty

    Args:
        coords_df: DataFrame from load_site_coordinates()
        save_path: Optional path to save figure

    Returns:
        fig, ax_main: Matplotlib figure and map axes
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Custom label offset positions (lat_offset, lon_offset) for each location
    # Positive lat_offset = north, Positive lon_offset = east
    LABEL_OFFSETS = {
        'Boston, MA': (0.95, 0.0),
        'Chicago, IL': (1.35, 0.0),
        'Columbia, MO': (-0.9, 5.5),
        'Kansas City, MO': (-0.8, -7.1),
        'Boise, ID': (0.8, 0.0),
        'Ottumwa, IA': (0.85, -1.5),
        'Milan, MO': (-0.3, -4.35),
        'Monett, MO': (-0.9, 4.6),
        'Riverside, CA': (-2.7, -4.1),
        'Ontario, CA': (-0.4, -4.3),
        'Palo Alto, CA': (-2.7, -4.1),
        'Southern California, CA': (0.4, 9.1),
        'Sacramento, CA': (0.8, 0.0),
        'Miami, FL': (0.0, -1.6),
        'New York, NY': (-2.4, 0.0),
        'Central Oklahoma, OK': (-2.6, 0.0),
    }

    # Labels default to sitting above their marker. Entries here anchor a
    # label to a different side; Miami sits to the west so it does not run
    # into the right-hand legend.
    LABEL_ALIGNMENTS = {
        'Miami, FL': ('right', 'center'),
    }

    df = coords_df.copy()

    # Every site is drawn. Sites sharing a city centroid are spread into a
    # small cluster (add_cluster_offsets), which is what makes co-located
    # sites of different types -- e.g. the Chicago airport site among the
    # Chicago treatment plants -- distinguishable at all.
    df_conus = add_cluster_offsets(df)

    # Create figure
    fig = plt.figure(figsize=(17.5, 7))

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

    # Sites whose exact location is not disclosed get a diffuse halo, drawn once
    # per city at the centroid so the haze reads as location uncertainty rather
    # than as one blob per site.
    for label in DIFFUSE_LABELS:
        group = df_conus[df_conus['map_label'] == label]
        if group.empty:
            continue
        labs = sorted({lab for v in group['labs'] for lab in v})
        color = LAB_COLORS[labs[0]]
        lon, lat = group['lon'].mean(), group['lat'].mean()
        for size, alpha_val in [(1500, 0.07), (1300, 0.08), (1100, 0.1), (900, 0.12),
                                (750, 0.14), (600, 0.17), (450, 0.21), (350, 0.26), (250, 0.32)]:
            ax_main.scatter(lon, lat, color=color, s=size, alpha=alpha_val,
                            edgecolors='none', zorder=2.5,
                            transform=ccrs.PlateCarree())
        # A single soft marker regardless of how many sites the city holds:
        # resolving them would imply a precision the coordinates do not carry.
        draw_site_marker(ax_main, lon, lat, 'o', labs, ccrs.PlateCarree(),
                         alpha=0.5, edgecolor='none', edgewidth=0)

    # Plot every remaining site at its offset position, in its location-type
    # shape. Non-WWTP sites are drawn last so the marker that distinguishes a
    # cluster -- the Chicago airport square among its treatment plants -- is
    # never hidden underneath an overlapping circle.
    df_points = df_conus[~df_conus['map_label'].isin(DIFFUSE_LABELS)].copy()
    df_points['marker'] = df_points['loc_type'].map(loc_type_marker)
    df_points = df_points.sort_values('marker', key=lambda c: c.eq('o'), ascending=False)

    for idx, row in df_points.iterrows():
        draw_site_marker(ax_main, row['lon_plot'], row['lat_plot'],
                         row['marker'], row['labs'], ccrs.PlateCarree())

    # Add labels for continental US (one per unique map_label)
    labels_added = set()
    for idx, row in df_conus.iterrows():
        label = row['map_label']
        if label not in labels_added:
            label_sites = df_conus[df_conus['map_label'] == label]
            mean_lat = label_sites['lat'].mean()
            mean_lon = label_sites['lon'].mean()

            lat_offset, lon_offset = LABEL_OFFSETS.get(label, (0.8, 0.0))

            ha, va = LABEL_ALIGNMENTS.get(label, ('center', 'bottom'))

            ax_main.text(mean_lon + lon_offset, mean_lat + lat_offset, label,
                        fontsize=FONT_SIZE_LARGE,
                        ha=ha, va=va, zorder=4,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 alpha=0.7, edgecolor='none'),
                        transform=ccrs.PlateCarree())
            labels_added.add(label)

    # Two keys: shape carries location type, fill carries processing
    # laboratory. Shape swatches stay neutral grey so they are not read as a
    # third laboratory colour.
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle

    def _swatch(marker, facecolor, label, fillstyle='full'):
        return Line2D([0], [0], linestyle='none', label=label,
                      marker=MarkerStyle(marker, fillstyle=fillstyle),
                      markerfacecolor=facecolor, markeredgecolor='black',
                      markeredgewidth=1.2,
                      markersize=11 * (MARKER_SIZE_SCALE[marker] ** 0.5))

    shape_handles = [
        _swatch('o', '#BBBBBB', 'Treatment plant'),
        _swatch('^', '#BBBBBB', 'Hospital'),
        _swatch('s', '#BBBBBB', 'Airport'),
    ]
    # "Both" is one marker drawn twice, each half filled, matching Boston.
    lab_handles = [
        _swatch('o', LAB_COLORS['MU'], 'University of Missouri'),
        _swatch('o', LAB_COLORS['SB'], 'SecureBio'),
        (_swatch('o', LAB_COLORS['MU'], 'Both', fillstyle='left'),
         _swatch('o', LAB_COLORS['SB'], 'Both', fillstyle='right')),
    ]

    common = dict(frameon=False, fontsize=FONT_SIZE_LARGE, labelspacing=0.7,
                  handletextpad=0.1, borderpad=0.0)
    title_font = FontProperties(family=LEGEND_TITLE_FONT_FAMILY, weight='bold',
                                size=FONT_SIZE_LARGE)

    # Both keys stack just outside the right edge of the map, processing
    # partner above location type. The axes is shrunk to leave room: legends
    # carry clip_on=True, so anything overflowing the canvas is cut even by
    # bbox_inches='tight'. clip_on is cleared below as a second guard.
    ax_main.set_position([0.0, 0.0, 0.74, 1.0])
    lab_legend = ax_main.legend(
        handles=lab_handles, labels=['University of Missouri', 'SecureBio', 'Both'],
        title='Sample processing partner',
        loc='upper left', bbox_to_anchor=(0.90, 1.0),
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
        title_fontproperties=title_font, **common)
    ax_main.add_artist(lab_legend)

    shape_legend = ax_main.legend(handles=shape_handles, title='Location type',
                                  loc='upper left', bbox_to_anchor=(0.90, 0.62),
                                  title_fontproperties=title_font, **common)

    for legend in (lab_legend, shape_legend):
        legend.set_clip_on(False)
        # Titles centre by default, which reads as misaligned above a
        # left-aligned column of swatches.
        legend._legend_box.align = 'left'
        for text in legend.get_texts():
            text.set_fontfamily(LEGEND_FONT_FAMILY)

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
        marker = loc_type_marker(row.get('loc_type', 'WWTP'))
        handle = Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                       markersize=12 * (MARKER_SIZE_SCALE[marker] ** 0.5),
                       markeredgecolor='black', markeredgewidth=1.5,
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
