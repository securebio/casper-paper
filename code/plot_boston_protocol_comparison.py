#!/usr/bin/env python3
"""
Supplementary figure: MU vs SB protocol comparison at the paired Boston site.

Boston DITP North and South were sampled once and the material split between
two laboratories, so every SB library has an MU library from the same outfall
on the same collection date. That makes the two series a paired benchmark: any
difference between them is protocol, not biology, because the biology is
literally the same water.

This figure asks three questions of that benchmark, one per row:

* row 1 -- do the two protocols track each other? Both Tobamovirus-normalized
  trends on one axis, so a systematic level offset between protocols is visible
  as well as the agreement in shape.
* row 2 -- does either protocol agree better with wastewater PCR?
* row 3 -- does either protocol agree better with MGH clinical testing?

Columns are the three respiratory targets of Figure 5, each split into the two
outfalls (South, then North).

Smoothing, correlation and styling follow Figure 5 exactly: MMWR-week
aggregation with a log1p-shifted mean for the wastewater series and an
arithmetic mean for clinical counts, a centered 5-week moving average, and
Spearman correlation on the smoothed weekly series.

Two conventions are specific to this figure:

* Correlations are computed only over MMWR weeks in which *both* laboratories
  have data. MU ran from March 2024 and SB from February 2025, so an
  unrestricted comparison would credit MU with an extra season of signal and
  the two agreement numbers would not be comparable.
* Trends are smoothed on each laboratory's full series to avoid edge effects,
  then clipped to the display window, as in Figure 5.
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter

sys.path.insert(0, str(Path(__file__).parent))

from plot_config import (
    LINE_ALPHA,
    LINE_WIDTH,
    NORMALIZATION_COLUMNS,
    add_stacked_rotated_label,
    get_location_color,
    save_figure,
)
from smoothing import (
    MIN_OVERLAP_WEEKS,
    SOURCE_KERNELS,
    calculate_mmwr_smoothed_trend,
    spearman_on_smoothed,
)
from data_loaders import NWSS_PATHOGEN_MAP, filter_to_dominant_source
from plot_mgs_pcr_clinical_lineage_panel import (
    CLINICAL_PATHOGEN_MAP,
    add_seasonal_shading,
    format_r_with_stars,
)

# The paired site: one collection, split between laboratories. South first so
# the column order matches the outfall order used elsewhere in the manuscript.
SITES = ['Boston DITP South, MA', 'Boston DITP North, MA']

# Laboratory -> line style. Colour carries the outfall, dash carries the lab.
LABS = ['MU', 'SB']
LAB_STYLES = {'MU': '-', 'SB': '--'}

PATHOGEN_KEYS = ['sars-cov-2', 'influenza_a', 'rsv']

# (row key, left-band label, name used in the correlation table). The 'mgs' row
# has no comparison series; it is the two laboratories against each other.
ROWS = [
    ('mgs', 'Wastewater sequencing', 'MU vs SB'),
    ('pcr', 'vs wastewater PCR', 'vs wastewater PCR'),
    ('clinical', 'vs clinical testing', 'vs clinical testing'),
]

# Grey comparison series, styled as in Figure 5.
COMPARISON_COLOR = '#666666'
COMPARISON_STYLE = {'pcr': '--', 'clinical': ':'}

# NWSS column holding raw concentration, as used by Figure 5 (not the
# PMMoV-normalized pcr_target_mic_lin).
NWSS_COL = 'pcr_target_avg_conc_lin'

# Unnormalized relative abundance: clade read pairs over total read pairs. The
# level ratio between protocols is reported against this as well as against the
# Tobamovirus-normalized column, because normalization divides by a quantity the
# protocol also affects -- so a normalized ratio measures the reported metric,
# not recovery.
RAW_ABUNDANCE_COL = 'ra_clade'

# Font sizes, matching Figure 5's scale.
LABEL_FONT = 22
LABEL2_FONT = 19
TICK_FONT = 15
CORR_FONT = 13

FIG_SIZE = (19, 13)

OUTPUT_PATH = '../figures/boston_protocol_comparison.png'
# Formatted for the supplement; the tidy long-form source is written alongside
# it with a _raw suffix, following the other correlation tables.
TABLE_PATH = '../tables/boston_protocol_comparison.tsv'


# =============================================================================
# Series preparation
# =============================================================================

def _lab_samples(mgs_data, site, lab, taxids):
    """Per-sample rows for one outfall, one laboratory, one target."""
    d = mgs_data[
        (mgs_data['site_name'] == site) & (mgs_data['taxid'].isin(taxids))
    ]
    if 'sequencing_lab' in d.columns:
        d = d[d['sequencing_lab'] == lab]
    return d.sort_values('date').copy()


def _week_key(smoothed):
    """The set of (mmwr_year, mmwr_week) tuples a smoothed series covers."""
    if smoothed is None or smoothed.empty:
        return set()
    return set(zip(smoothed['mmwr_year'], smoothed['mmwr_week']))


def _restrict_to_weeks(smoothed, weeks):
    """Drop MMWR weeks outside ``weeks``, so two labs are judged on one window."""
    if smoothed is None or smoothed.empty or not weeks:
        return None
    mask = [(y, w) in weeks for y, w in
            zip(smoothed['mmwr_year'], smoothed['mmwr_week'])]
    out = smoothed[mask]
    return out if not out.empty else None


def prepare_site_series(mgs_data_dict, nwss_data_dict, clinical_data,
                        mgs_col):
    """Smooth every series this figure draws, keyed by (pathogen, site).

    Returns:
        dict mapping (pathogen_key, site) -> dict with:
            'labs'    : {lab: {'samples', 'smoothed'}}
            'paired'  : set of MMWR weeks covered by both laboratories
            'pcr'     : smoothed NWSS series, or None
            'clinical': smoothed clinical series, or None
    """
    series = {}
    for pathogen_key in PATHOGEN_KEYS:
        taxids = NWSS_PATHOGEN_MAP[pathogen_key]['taxids']
        mgs_data = mgs_data_dict.get(pathogen_key)
        nwss_data = nwss_data_dict.get(pathogen_key)

        # NWSS reports one sewershed for the whole DITP catchment, so the PCR
        # series is shared by both outfalls; it is smoothed once per target.
        pcr_smoothed = None
        if nwss_data is not None and not nwss_data.empty:
            site_nwss = filter_to_dominant_source(nwss_data.copy())
            site_nwss = site_nwss[
                site_nwss[NWSS_COL].notna() & (site_nwss[NWSS_COL] >= 0)
            ]
            if not site_nwss.empty:
                s = calculate_mmwr_smoothed_trend(site_nwss, 'date', NWSS_COL)
                pcr_smoothed = s if not s.empty else None

        clinical_smoothed = None
        clinical_col = CLINICAL_PATHOGEN_MAP.get(pathogen_key)
        if (clinical_data is not None and clinical_col is not None
                and clinical_col in clinical_data.columns):
            clin = clinical_data[['date', clinical_col]].dropna()
            if not clin.empty:
                s = calculate_mmwr_smoothed_trend(
                    clin, 'date', clinical_col,
                    kernel=SOURCE_KERNELS['clinical'])
                clinical_smoothed = s if not s.empty else None

        for site in SITES:
            labs = {}
            for lab in LABS:
                if mgs_data is None:
                    continue
                samples = _lab_samples(mgs_data, site, lab, taxids)
                if samples.empty:
                    continue
                smoothed = calculate_mmwr_smoothed_trend(samples, 'date', mgs_col)
                raw = calculate_mmwr_smoothed_trend(samples, 'date',
                                                    RAW_ABUNDANCE_COL)
                labs[lab] = {
                    'samples': samples,
                    'smoothed': smoothed if not smoothed.empty else None,
                    'smoothed_raw': raw if not raw.empty else None,
                }

            paired = set.intersection(*[
                _week_key(v['smoothed']) for v in labs.values()
            ]) if len(labs) == len(LABS) else set()

            series[(pathogen_key, site)] = {
                'labs': labs,
                'paired': paired,
                'pcr': pcr_smoothed,
                'clinical': clinical_smoothed,
            }
    return series


# =============================================================================
# Correlations
# =============================================================================

# Metric labels. The between-protocol block reports every statistic against
# both, because whether normalization makes the two laboratories comparable is
# the question this comparison exists to answer.
METRIC_NORM = 'Tobamovirus-normalized'
METRIC_RAW = 'Total read relative abundance'
METRIC_COLUMNS = {METRIC_NORM: None, METRIC_RAW: RAW_ABUNDANCE_COL}


def _median_ratio(merged, col):
    """Median MU/SB ratio of ``col`` over dates where both laboratories detected.

    Reported whatever the number of contributing dates; the count travels with
    the ratio so a reader can weigh it rather than having it withheld.
    """
    both = merged[(merged[f'mu_{col}'] > 0) & (merged[f'sb_{col}'] > 0)]
    if both.empty:
        return np.nan, 0
    ratio = float(np.exp(np.median(
        np.log(both[f'mu_{col}'] / both[f'sb_{col}']))))
    return ratio, len(both)


def _paired_sample_agreement(labs, mgs_col, raw_col=RAW_ABUNDANCE_COL):
    """Per-laboratory agreement on matched collection dates, per metric.

    Both laboratories ran the same collection dates, so there is no cadence to
    reconcile and no smoothing is applied: smoothing here would only let each
    point borrow strength from its neighbours and overstate significance. The
    correlation and the level ratio are returned for the normalized and the
    unnormalized metric alike.

    Returns:
        dict mapping metric label -> {'r', 'p', 'n', 'n_both', 'ratio'}, or
        None if a laboratory is missing or too few dates are shared.
    """
    if len(labs) < len(LABS):
        return None
    columns = {METRIC_NORM: mgs_col, METRIC_RAW: raw_col}
    cols = list(dict.fromkeys(columns.values()))
    frames = [labs[lab]['samples'][['date'] + cols].rename(
                  columns={c: f'{prefix}_{c}' for c in cols})
              for lab, prefix in (('MU', 'mu'), ('SB', 'sb'))]
    merged = pd.merge(frames[0], frames[1], on='date', how='inner').dropna(
        subset=[f'mu_{mgs_col}', f'sb_{mgs_col}'])
    if len(merged) < MIN_OVERLAP_WEEKS:
        return None

    from scipy.stats import spearmanr
    out = {}
    for metric, col in columns.items():
        r, p = spearmanr(merged[f'mu_{col}'], merged[f'sb_{col}'])
        ratio, n_both = _median_ratio(merged, col)
        out[metric] = {'r': r, 'p': p, 'n': len(merged),
                       'n_both': n_both, 'ratio': ratio}
    return out


def compute_correlations(series, mgs_col):
    """All agreement statistics behind the figure and table, as a tidy frame.

    Correlations follow one convention throughout, matching Figure 5 and the
    other correlation tables in the supplement: MMWR-week aggregation with a
    centered five-week moving average, correlated by Spearman on the weeks two
    series share. Reporting a single statistic keeps the between-protocol and
    benchmark blocks comparable, which two regimes would not.

    Level ratios are per-sample quantities rather than temporal comparisons, so
    they are computed on matched collection dates; the unsmoothed matched-date
    correlation is computed alongside them for the Results text.

    Every comparison is restricted to dates or weeks covered by both
    laboratories, so MU and SB are judged over an identical window.
    """
    records = []
    for pathogen_key in PATHOGEN_KEYS:
        label = NWSS_PATHOGEN_MAP[pathogen_key]['label']
        for site in SITES:
            entry = series[(pathogen_key, site)]
            labs, paired = entry['labs'], entry['paired']

            restricted = {
                lab: _restrict_to_weeks(v['smoothed'], paired)
                for lab, v in labs.items()
            }

            if len(labs) == len(LABS):
                restricted_raw = {
                    lab: _restrict_to_weeks(v.get('smoothed_raw'), paired)
                    for lab, v in labs.items()
                }
                for metric, pair in ((METRIC_NORM, restricted),
                                     (METRIC_RAW, restricted_raw)):
                    r, p, n = spearman_on_smoothed(pair['MU'], pair['SB'])
                    records.append({
                        'pathogen': label, 'site_name': site,
                        'comparison': 'MU vs SB', 'series': 'smoothed weekly',
                        'metric': metric,
                        'r_value': r, 'p_value': p, 'n_points': n})

                # Retained for reference and for the Results text: the
                # unsmoothed matched-date statistic. Not shown in the table,
                # which reports one convention throughout.
                per_metric = _paired_sample_agreement(labs, mgs_col)
                if per_metric is not None:
                    for metric, stats in per_metric.items():
                        records.append({
                            'pathogen': label, 'site_name': site,
                            'comparison': 'MU vs SB', 'series': 'paired samples',
                            'metric': metric,
                            'r_value': stats['r'], 'p_value': stats['p'],
                            'n_points': stats['n'],
                            'n_both_detected': stats['n_both'],
                            'median_mu_sb_ratio': stats['ratio']})

            for source, comparison in (('pcr', 'vs wastewater PCR'),
                                       ('clinical', 'vs clinical testing')):
                if entry[source] is None:
                    continue
                for lab in LABS:
                    if restricted.get(lab) is None:
                        continue
                    r, p, n = spearman_on_smoothed(restricted[lab], entry[source])
                    records.append({
                        'pathogen': label, 'site_name': site,
                        'comparison': comparison, 'seq_lab': lab,
                        'series': 'smoothed weekly', 'metric': METRIC_NORM,
                        'r_value': r, 'p_value': p, 'n_points': n})

    df = pd.DataFrame(records)
    cols = ['pathogen', 'site_name', 'comparison', 'seq_lab', 'series',
            'metric', 'r_value', 'p_value', 'n_points', 'n_both_detected',
            'median_mu_sb_ratio']
    return df.reindex(columns=[c for c in cols if c in df.columns])


def _format_r_with_p(r, p):
    """``R (p)`` as used by the other correlation tables in the supplement."""
    if pd.isna(r) or pd.isna(p):
        return ''
    p_str = f'{p:.3f}' if p >= 0.001 else f'{p:.1e}'
    return f'{r:.2f} ({p_str})'


# Display rows. Each is a dict so the selection keys stay readable; ``kind`` is
# 'section' for a group heading, 'corr' for a correlation cell and 'ratio' for
# a level-ratio cell. Within a block every row varies exactly one thing.
_DISPLAY_ROWS = [
    {'label': 'Between protocols', 'kind': 'section'},
    {'label': f'Correlation ({METRIC_RAW.lower()})', 'kind': 'corr',
     'comparison': 'MU vs SB', 'series': 'smoothed weekly', 'metric': METRIC_RAW},
    {'label': 'Correlation (tobamovirus normalized)', 'kind': 'corr',
     'comparison': 'MU vs SB', 'series': 'smoothed weekly', 'metric': METRIC_NORM},
    {'label': f'Median MU/SB ratio ({METRIC_RAW.lower()})', 'kind': 'ratio',
     'comparison': 'MU vs SB', 'series': 'paired samples', 'metric': METRIC_RAW,
     'n_text': '\u2013'},
    {'label': 'Median MU/SB ratio (tobamovirus normalized)', 'kind': 'ratio',
     'comparison': 'MU vs SB', 'series': 'paired samples', 'metric': METRIC_NORM,
     'n_text': '\u2013'},
    {'label': 'vs wastewater PCR', 'kind': 'section'},
    {'label': 'MU', 'kind': 'corr', 'comparison': 'vs wastewater PCR',
     'series': 'smoothed weekly', 'metric': METRIC_NORM, 'lab': 'MU'},
    {'label': 'SB', 'kind': 'corr', 'comparison': 'vs wastewater PCR',
     'series': 'smoothed weekly', 'metric': METRIC_NORM, 'lab': 'SB'},
    {'label': 'vs clinical testing', 'kind': 'section'},
    {'label': 'MU', 'kind': 'corr', 'comparison': 'vs clinical testing',
     'series': 'smoothed weekly', 'metric': METRIC_NORM, 'lab': 'MU'},
    {'label': 'SB', 'kind': 'corr', 'comparison': 'vs clinical testing',
     'series': 'smoothed weekly', 'metric': METRIC_NORM, 'lab': 'SB'},
]

# Columns run outfall-major so the three targets of one outfall sit together,
# rendering as a merged two-level header in the manuscript.
_COLUMN_ORDER = [(site, key) for site in SITES for key in PATHOGEN_KEYS]


def _column_name(site, pathogen_key):
    outfall = 'South' if site == SITES[0] else 'North'
    return f'{outfall}: {NWSS_PATHOGEN_MAP[pathogen_key]["label"]}'


def build_display_table(corr_df):
    """Lay the statistics out for the supplement: comparisons down, panels across.

    Sectioned by what is being compared, with one column per panel of the
    figure. Cells are ``R (p)``, except the ratio rows, which carry the median
    MU/SB ratio and, as n, the number of dates it rests on. That count can be
    as low as one, so it is reported alongside every ratio rather than used to
    suppress any of them.
    """
    rows = []
    for spec in _DISPLAY_ROWS:
        kind = spec['kind']
        row = {'Comparison': spec['label'],
               'N': '' if kind == 'section' else spec.get('n_text', '')}
        for site, pathogen_key in _COLUMN_ORDER:
            column = _column_name(site, pathogen_key)
            if kind == 'section':
                row[column] = ''
                continue

            match = corr_df[
                (corr_df['pathogen'] == NWSS_PATHOGEN_MAP[pathogen_key]['label'])
                & (corr_df['site_name'] == site)
                & (corr_df['comparison'] == spec['comparison'])
                & (corr_df['series'] == spec['series'])
                & (corr_df['metric'] == spec['metric'])]
            if spec.get('lab') is not None:
                match = match[match['seq_lab'] == spec['lab']]
            if match.empty:
                row[column] = ''
                continue
            record = match.iloc[0]

            if kind == 'ratio':
                ratio = record.get('median_mu_sb_ratio')
                n_both = record.get('n_both_detected')
                n_note = '' if pd.isna(n_both) else f' (n = {int(n_both)})'
                row[column] = ('\u2013' if pd.isna(ratio)
                               else f'{ratio:.1f}') + n_note
            elif record['n_points'] < MIN_OVERLAP_WEEKS:
                row[column] = ''
            else:
                row[column] = _format_r_with_p(record['r_value'],
                                               record['p_value'])
                if not row['N']:
                    row['N'] = str(int(record['n_points']))
        rows.append(row)

    ordered = (['Comparison', 'N']
               + [_column_name(site, key) for site, key in _COLUMN_ORDER])
    return pd.DataFrame(rows)[ordered]


def print_protocol_comparison_statistics(corr_df):
    """Report the paired-protocol agreement numbers for the response text."""
    print("\n" + "=" * 86)
    print("MU vs SB protocol agreement at the paired Boston DITP site")
    print("=" * 86)

    reportable = corr_df[corr_df['n_points'] >= MIN_OVERLAP_WEEKS]

    between = reportable[(reportable['comparison'] == 'MU vs SB')
                         & (reportable['series'] == 'paired samples')]
    for metric in [METRIC_RAW, METRIC_NORM]:
        sub = between[between['metric'] == metric]
        if sub.empty:
            continue
        print(f"\n--- MU vs SB, matched collection dates ({metric}) ---")
        print(f"{'Pathogen':<14} {'Outfall':<24} {'R':>7} {'p':>10} {'n':>5} "
              f"{'MU/SB':>7} {'n both':>7}")
        for _, row in sub.iterrows():
            ratio = row.get('median_mu_sb_ratio')
            n_both = row.get('n_both_detected')
            print(f"{row['pathogen']:<14} {row['site_name']:<24} "
                  f"{row['r_value']:7.2f} {row['p_value']:10.2e} "
                  f"{int(row['n_points']):5d} "
                  f"{ratio:7.2f} {int(n_both):7d}"
                  if pd.notna(ratio) else
                  f"{row['pathogen']:<14} {row['site_name']:<24} "
                  f"{row['r_value']:7.2f} {row['p_value']:10.2e} "
                  f"{int(row['n_points']):5d}")

    for comparison in ['vs wastewater PCR', 'vs clinical testing']:
        sub = reportable[(reportable['comparison'] == comparison)
                         & (reportable['metric'] == METRIC_NORM)]
        if sub.empty:
            continue
        print(f"\n--- Agreement with external benchmark: {comparison} ---")
        print(f"{'Pathogen':<14} {'Outfall':<24} {'MU R':>7} {'SB R':>7} "
              f"{'diff':>7} {'n':>5}")
        for (pathogen, site), grp in sub.groupby(['pathogen', 'site_name'],
                                                 sort=False):
            r = {row['seq_lab']: row['r_value'] for _, row in grp.iterrows()}
            n = int(grp['n_points'].max())
            if 'MU' not in r or 'SB' not in r:
                continue
            print(f"{pathogen:<14} {site:<24} {r['MU']:7.2f} {r['SB']:7.2f} "
                  f"{r['MU'] - r['SB']:7.2f} {n:5d}")

    paired = between[between['metric'] == METRIC_NORM]
    if not paired.empty:
        print(f"\nBetween-protocol Spearman R (matched dates, {METRIC_NORM}): "
              f"median {paired['r_value'].median():.2f}, range "
              f"{paired['r_value'].min():.2f}-{paired['r_value'].max():.2f} "
              f"across {len(paired)} target-outfall pairs.")
    ratios = paired.get('median_mu_sb_ratio')
    if ratios is not None and ratios.notna().any():
        print(f"Median per-sample MU/SB abundance ratio: "
              f"{ratios.dropna().median():.2f} "
              f"(range {ratios.dropna().min():.2f}-{ratios.dropna().max():.2f}).")
    print("=" * 86 + "\n")


# =============================================================================
# Plotting
# =============================================================================

class _CleanScalarFormatter(ScalarFormatter):
    """Figure 5's tick formatter: scientific offset, integer mantissa, bare 0."""

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


def _format_test_count(value, _pos=None):
    """Plain counts for the clinical axis, abbreviated with k from a thousand.

    Positive tests are counts in the hundreds, where the default scientific
    offset (an axis labelled 0, 1, 3 above a bare '1e2') costs the reader a
    multiplication to recover a number they could simply have been shown.
    """
    if value == 0:
        return '0'
    if abs(value) < 1_000:
        return f'{value:.0f}'
    thousands = value / 1_000
    return (f'{thousands:.0f}k' if abs(thousands) >= 10 or thousands ==
            int(thousands) else f'{thousands:.1f}k')


def _apply_axis_style(ax, is_right=False):
    ax.tick_params(axis='both', labelsize=TICK_FONT)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    ax.yaxis.set_major_formatter(_CleanScalarFormatter(useMathText=False))
    ax.yaxis.get_offset_text().set_fontsize(TICK_FONT - 2)
    if not is_right:
        ax.grid(True, alpha=0.3, zorder=1)
        ax.set_axisbelow(True)


def _clip(smoothed, date_min, date_max):
    """Restrict a smoothed series to a display window."""
    if smoothed is None or smoothed.empty:
        return None
    dates = pd.to_datetime(smoothed['date'])
    out = smoothed[(dates >= pd.Timestamp(date_min))
                   & (dates <= pd.Timestamp(date_max))]
    return out if not out.empty else None


def _shared_window(series):
    """The paired window: collection dates on which both laboratories ran.

    Every panel uses this one window so the rows are directly stackable in
    time. The wastewater PCR and clinical series end earlier than the
    sequencing series, and are simply drawn to wherever their data stops
    rather than trimming the axis -- which keeps the point at which each
    external benchmark runs out visible instead of hiding it.
    """
    starts, ends = [], []
    for entry in series.values():
        if len(entry['labs']) == len(LABS):
            common = set(entry['labs']['MU']['samples']['date']) & \
                set(entry['labs']['SB']['samples']['date'])
            if common:
                starts.append(min(common))
                ends.append(max(common))
    if not starts:
        return None, None
    return pd.Timestamp(min(starts)), pd.Timestamp(max(ends))


def pcr_units(nwss_data_dict):
    """The concentration unit NWSS reports for this site, for the axis label.

    Read from the data rather than hard-coded: Boston is reported by
    WastewaterSCAN in copies per gram of dry sludge, but other reporters use
    copies per litre, and an axis that names the wrong matrix is worse than one
    that names none.
    """
    units = set()
    for df in nwss_data_dict.values():
        if df is None or df.empty or 'pcr_target_units' not in df.columns:
            continue
        units.update(filter_to_dominant_source(df.copy())
                     ['pcr_target_units'].dropna().unique())
    return units.pop() if len(units) == 1 else None


def _benchmark_end(series, row_key):
    """Last date the comparison series for ``row_key`` carries data.

    The sequencing trend is smoothed over each laboratory's full series, but
    drawn only as far as the series it is being compared against, so no stretch
    of curve is shown that no correlation was computed over.
    """
    if row_key == 'mgs':
        return None
    end = None
    for entry in series.values():
        s = entry[row_key]
        if s is not None and not s.empty:
            latest = pd.to_datetime(s['date']).max()
            end = latest if end is None else max(end, latest)
    return end


def _group_mgs_top(series, pathogen_key, date_min, date_max):
    """Upper left-axis limit shared by the two outfall panels of one target.

    Taken over both outfalls, both laboratories and the whole paired window,
    then applied to all six panels of the pathogen group. Sharing it across the
    South/North pair is what makes the two outfalls readable against each
    other; extending it down the rows keeps the sequencing trends on one scale
    even though the lower two rows are drawn over a shorter period.
    """
    top = 0
    for site in SITES:
        for lab_entry in series[(pathogen_key, site)]['labs'].values():
            display = _clip(lab_entry['smoothed'], date_min, date_max)
            if display is not None:
                top = max(top, display['smoothed_values'].max())
    return top * 1.1 if top > 0 else None


# Targets whose signal sits at the left of the paired window, where a
# top-left annotation would land on the curve. Their correlations are set
# against the right-hand edge instead.
RIGHT_ANNOTATED = {'influenza_a', 'rsv'}


def _annotate_correlations(ax, entries, side='left'):
    """Stack ``(prefix, text)`` correlation labels in a top corner of a panel."""
    x, align = (0.04, 'left') if side == 'left' else (0.96, 'right')
    for i, (prefix, text) in enumerate(entries):
        y = 0.94 - i * 0.13
        label = f'{prefix} {text}' if prefix else text
        ax.text(x, y, label, transform=ax.transAxes, ha=align, va='center',
                fontsize=CORR_FONT, color='black', zorder=10)


def plot_boston_protocol_comparison(mgs_data_dict, nwss_data_dict,
                                    clinical_data, save_path=None,
                                    normalization='tobamovirus',
                                    table_path=None):
    """Draw the 3x6 paired-protocol comparison grid.

    Args:
        mgs_data_dict: pathogen key -> per-sample MGS DataFrame.
        nwss_data_dict: pathogen key -> NWSS PCR DataFrame.
        clinical_data: wide clinical testing DataFrame (date + pathogen cols).
        save_path: where to write the figure, or None.
        normalization: key into NORMALIZATION_COLUMNS; 'tobamovirus' by default.
        table_path: where to write the correlation TSV, or None.

    Returns:
        (fig, axes, corr_df, display_table)
    """
    mgs_col = NORMALIZATION_COLUMNS[normalization]
    series = prepare_site_series(mgs_data_dict, nwss_data_dict, clinical_data,
                                 mgs_col)
    corr_df = compute_correlations(series, mgs_col)

    n_rows, n_cols = len(ROWS), len(PATHOGEN_KEYS) * len(SITES)

    fig = plt.figure(figsize=FIG_SIZE)
    # The two outfalls of a target share a left axis, so only the South panel
    # is labelled and the pair can sit almost flush. A narrow spacer column
    # separates the targets, where a labelled left axis does need clearance.
    width_ratios = []
    for i in range(len(PATHOGEN_KEYS)):
        if i:
            width_ratios.append(0.24)
        width_ratios.extend([1, 1])
    gs = GridSpec(n_rows, len(width_ratios), figure=fig,
                  width_ratios=width_ratios, wspace=0.10, hspace=0.22)
    grid_cols = [c for c, w in enumerate(width_ratios) if w == 1]

    axes = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
        for col in range(n_cols):
            axes[row, col] = fig.add_subplot(gs[row, grid_cols[col]])

    date_min, date_max = _shared_window(series)

    # One left-axis limit per pathogen group, shared by its South and North
    # panels and by every row.
    group_tops = {
        pathogen_key: _group_mgs_top(series, pathogen_key, date_min, date_max)
        for pathogen_key in PATHOGEN_KEYS
    }

    for row_idx, (row_key, _, comparison_name) in enumerate(ROWS):
        # Rows carrying a comparison series stop where that series stops.
        # Smoothing has already run over each laboratory's full record, so this
        # trims the drawn curve without changing its shape.
        row_end = _benchmark_end(series, row_key)
        draw_max = date_max if row_end is None else min(date_max,
                                                        pd.Timestamp(row_end))

        for col_idx in range(n_cols):
            pathogen_key = PATHOGEN_KEYS[col_idx // len(SITES)]
            site = SITES[col_idx % len(SITES)]
            entry = series[(pathogen_key, site)]
            ax1 = axes[row_idx, col_idx]
            ax2 = ax1.twinx() if row_key != 'mgs' else None

            add_seasonal_shading(ax1, date_min, date_max)

            color = get_location_color(site)
            for lab in LABS:
                lab_entry = entry['labs'].get(lab)
                if lab_entry is None:
                    continue
                display = _clip(lab_entry['smoothed'], date_min, draw_max)
                if display is None:
                    continue
                ax1.plot(display['date'], display['smoothed_values'],
                         color=color, linewidth=LINE_WIDTH * 1.5,
                         linestyle=LAB_STYLES[lab], alpha=LINE_ALPHA,
                         label=lab, zorder=4)

            if row_key != 'mgs':
                comparison = _clip(entry[row_key], date_min, draw_max)
                if comparison is not None:
                    ax2.plot(comparison['date'], comparison['smoothed_values'],
                             color=COMPARISON_COLOR, linewidth=LINE_WIDTH * 1.5,
                             linestyle=COMPARISON_STYLE[row_key],
                             alpha=LINE_ALPHA, zorder=2)
                    top = comparison['smoothed_values'].max() * 1.1
                    if top > 0:
                        ax2.set_ylim(bottom=-top * 0.05, top=top)

            group_top = group_tops[pathogen_key]
            if group_top:
                ax1.set_ylim(bottom=-group_top * 0.05, top=group_top)

            # Correlation annotations, taken from the exported table so figure
            # and table can never disagree. Every panel carries the same
            # statistic -- Spearman on the MMWR-week smoothed series -- which is
            # also what the panel draws.
            pathogen_label = NWSS_PATHOGEN_MAP[pathogen_key]['label']
            rows = corr_df[(corr_df['pathogen'] == pathogen_label)
                           & (corr_df['site_name'] == site)
                           & (corr_df['series'] == 'smoothed weekly')
                           & (corr_df['metric'] == METRIC_NORM)
                           & (corr_df['n_points'] >= MIN_OVERLAP_WEEKS)]
            sub = rows[rows['comparison'] == comparison_name]
            entries = [
                ('' if row_key == 'mgs' else r['seq_lab'],
                 f"{format_r_with_stars(r['r_value'], r['p_value'])} "
                 f"({int(r['n_points'])})")
                for _, r in sub.iterrows()
            ]
            _annotate_correlations(
                ax1, entries,
                side='right' if pathogen_key in RIGHT_ANNOTATED else 'left')

            buffer = pd.Timedelta(days=7)
            ax1.set_xlim(date_min - buffer, date_max + buffer)
            ax1.set_ylabel('')

            # Panels sit close together, so date labels are kept sparse to
            # stop them running into the neighbouring panel.
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            if row_idx == n_rows - 1:
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45,
                         ha='right')
            else:
                ax1.set_xticklabels([])

            _apply_axis_style(ax1)
            if col_idx % len(SITES) != 0:
                ax1.set_yticklabels([])
            if ax2 is not None:
                _apply_axis_style(ax2, is_right=True)
                if row_key == 'clinical':
                    ax2.yaxis.set_major_formatter(
                        FuncFormatter(_format_test_count))
                ax2.set_ylabel('')
                # The benchmark series is shared by the two outfalls of a
                # target but differs between targets, so the scale is labelled
                # once per pathogen group rather than once per figure.
                if col_idx % len(SITES) != len(SITES) - 1:
                    ax2.set_yticklabels([])

    _add_annotations(fig, axes, n_rows, n_cols,
                     pcr_unit=pcr_units(nwss_data_dict))

    display_table = build_display_table(corr_df)
    if table_path:
        Path(table_path).parent.mkdir(parents=True, exist_ok=True)
        display_table.to_csv(table_path, sep='\t', index=False)
        raw_path = table_path.replace('.tsv', '_raw.tsv')
        corr_df.to_csv(raw_path, sep='\t', index=False)
        print(f"Saved {table_path} and {raw_path}")

    if save_path:
        save_figure(fig, save_path)

    return fig, axes, corr_df, display_table


def _add_annotations(fig, axes, n_rows, n_cols, pcr_unit=None):
    """Pathogen headers, outfall titles, row band, legend and axis labels."""
    plt.subplots_adjust(top=0.82, bottom=0.06, left=0.10, right=0.93)

    top_positions = [axes[0, c].get_position() for c in range(n_cols)]
    title_y = top_positions[0].y1 + 0.012
    header_y = title_y + 0.035
    header_height = 0.038

    for col_idx in range(n_cols):
        bbox = top_positions[col_idx]
        outfall = 'DITP South' if col_idx % len(SITES) == 0 else 'DITP North'
        fig.text(bbox.x0 + bbox.width / 2, title_y, outfall,
                 fontsize=LABEL2_FONT, ha='center', va='bottom')

    for i, pathogen_key in enumerate(PATHOGEN_KEYS):
        left = top_positions[i * len(SITES)].x0
        right = top_positions[i * len(SITES) + len(SITES) - 1].x1
        fig.patches.append(Rectangle(
            (left, header_y), right - left, header_height,
            transform=fig.transFigure, facecolor='#CCCCCC', edgecolor='none',
            zorder=1, alpha=0.5))
        fig.text((left + right) / 2, header_y + header_height / 2,
                 NWSS_PATHOGEN_MAP[pathogen_key]['label'],
                 fontsize=LABEL_FONT, ha='center', va='center', zorder=10)

    # Row band on the left, mirroring Figure 5's pathogen band.
    band_x, band_width, band_pad = 0.001, 0.028, 0.012
    band_top = axes[0, 0].get_position().y1 + band_pad
    band_bottom = axes[n_rows - 1, 0].get_position().y0 - band_pad
    fig.patches.append(Rectangle(
        (band_x, band_bottom), band_width, band_top - band_bottom,
        transform=fig.transFigure, facecolor='#CCCCCC', edgecolor='none',
        zorder=1, alpha=0.5))
    for row_idx, (_, row_label, _) in enumerate(ROWS):
        bbox = axes[row_idx, 0].get_position()
        fig.text(band_x + band_width / 2, bbox.y0 + bbox.height / 2, row_label,
                 fontsize=LABEL2_FONT, ha='center', va='center', rotation=90,
                 zorder=10)

    # Left y-axis label, shared by every panel.
    y_center = ((axes[n_rows - 1, 0].get_position().y0
                 + axes[0, 0].get_position().y1) / 2)
    add_stacked_rotated_label(fig, band_x + band_width + 0.022, y_center,
                              'Wastewater sequencing signal\n(pathogen / tobamovirus read pairs)',
                              rotation=90, main_size=LABEL2_FONT)

    # Right y-axis labels: the benchmark differs by row, so each is labelled
    # against its own row rather than once for the figure.
    pcr_label = ('Wastewater PCR concentration\n(%s)' % pcr_unit) if pcr_unit \
        else 'Wastewater PCR concentration'
    right_labels = {'pcr': pcr_label, 'clinical': 'Positive tests'}
    for row_idx, (row_key, _, _) in enumerate(ROWS):
        if row_key not in right_labels:
            continue
        bbox = axes[row_idx, n_cols - 1].get_position()
        add_stacked_rotated_label(fig, bbox.x1 + 0.045, bbox.y0 + bbox.height / 2,
                                  right_labels[row_key], rotation=270,
                                  main_size=LABEL2_FONT)

    # Legends sit above the pathogen headers, anchored by their lower edge so
    # they grow upward into the margin rather than down onto the headers.
    legend_y = header_y + header_height + 0.03
    left_edge = top_positions[0].x0

    lab_handles = [
        Line2D([0], [0], color='#4d4d4d', linewidth=LINE_WIDTH * 1.5,
               linestyle=LAB_STYLES[lab]) for lab in LABS
    ]
    lab_labels = ['MU protocol', 'SB protocol']
    lab_legend = fig.legend(lab_handles, lab_labels, loc='lower left',
                            bbox_to_anchor=(left_edge, legend_y), frameon=False,
                            fontsize=LABEL2_FONT, ncol=1, handletextpad=0.7,
                            labelspacing=0.3)

    # Title placed off the measured legend box, so it never lands on an entry.
    fig.canvas.draw()
    lab_bbox = lab_legend.get_window_extent().transformed(
        fig.transFigure.inverted())
    fig.text(left_edge, lab_bbox.y1 + 0.008, 'CASPER wastewater sequencing',
             fontsize=LABEL_FONT, ha='left', va='bottom')

    comparison_handles = [
        Line2D([0], [0], color=COMPARISON_COLOR, linewidth=LINE_WIDTH * 1.5,
               linestyle=COMPARISON_STYLE['pcr']),
        Line2D([0], [0], color=COMPARISON_COLOR, linewidth=LINE_WIDTH * 1.5,
               linestyle=COMPARISON_STYLE['clinical']),
    ]
    fig.legend(comparison_handles,
               ['Wastewater PCR', 'MGH clinical testing'],
               loc='lower left', bbox_to_anchor=(left_edge + 0.24, legend_y),
               frameon=False, fontsize=LABEL2_FONT, ncol=1,
               handletextpad=0.7, labelspacing=0.3)

    import matplotlib.patches as mpatches
    fig.legend(handles=[
        mpatches.Patch(color='#E6F2FF', alpha=0.5, label='Winter (Dec-Feb)'),
        mpatches.Patch(color='#FFF3B0', alpha=0.5, label='Summer (Jun-Aug)'),
    ], loc='lower right', bbox_to_anchor=(0.98, legend_y), frameon=False,
        fontsize=LABEL2_FONT)
    fig.add_artist(lab_legend)


def main(save_path=OUTPUT_PATH, table_path=TABLE_PATH):
    from data_loaders import (
        load_clinical_positives_tests_data,
        load_mgs_nwss_matches,
        load_mgs_pathogen_data,
        load_nwss_data,
    )

    matches = load_mgs_nwss_matches()
    sewershed_ids = sorted({
        int(sid)
        for _, row in matches[matches['site_name'].isin(SITES)].iterrows()
        for sid in row['sewershed_ids']
    })

    nwss_all = load_nwss_data()
    nwss_data_dict = {
        key: df[df['sewershed_id'].isin(sewershed_ids)].copy()
        for key, df in nwss_all.items()
    }

    mgs_data_dict = {}
    for pathogen_key in PATHOGEN_KEYS:
        mgs = load_mgs_pathogen_data(
            taxids=NWSS_PATHOGEN_MAP[pathogen_key]['taxids'])
        if not mgs.empty:
            mgs_data_dict[pathogen_key] = mgs

    clinical_data = load_clinical_positives_tests_data()

    fig, axes, corr_df, display_table = plot_boston_protocol_comparison(
        mgs_data_dict, nwss_data_dict, clinical_data,
        save_path=save_path, table_path=table_path)
    print_protocol_comparison_statistics(corr_df)
    print(display_table.to_string(index=False))
    return fig, axes, corr_df, display_table


if __name__ == '__main__':
    main()
