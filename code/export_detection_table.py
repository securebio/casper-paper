#!/usr/bin/env python3
"""
Supplementary table: detection frequency and sequencing depth per pathogen.

Reports, for each target in the Figure 4c panel, how often it was detected at
all and how many read pairs sat behind a detection. Supports the requests for
the absolute alignment counts underlying the normalized abundance trends, and
for a sense of the spread behind them.

Every statistic here is a plain per-sample descriptive over the full release
(all 1,206 samples, no temporal-outlier filtering), so the columns are directly
comparable down the table. Dispersion of observations *around the smoothed
trend* is deliberately not included: it is computed per site-and-laboratory
series and then summarized across series, excludes non-detects, and depends on
the smoothing kernel, so it does not share a denominator with these columns. It
is reported in the Results text instead.

Two summaries are given for each of the median and the IQR:

* over all samples, where non-detects count as zero. This is what the release
  looks like as a whole, but it collapses to zero for any target detected in
  fewer than half of samples.
* over detecting samples only. This is the depth behind an actual detection,
  and is only interpretable alongside the non-detect column: a median of 2 read
  pairs for a target absent from 91% of samples means "when anything was seen,
  it was typically 2 reads".
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data_loaders import load_all_relative_abundance
from plot_vv_pathogen_combined import PATHOGENS

# Read pairs assigned to the pathogen clade by the VV pipeline. A sample is a
# non-detect when this is zero, equivalent to zero normalized abundance.
COUNT_COL = 'clade_counts'

OUTPUT_PATH = '../tables/pathogen_detection.tsv'


def _count(value):
    return f'{value:,.0f}'


def _iqr(counts):
    if counts.empty:
        return ''
    return f'{counts.quantile(0.25):,.0f}-{counts.quantile(0.75):,.0f}'


def non_detects(all_counts, detected):
    """How often the target was absent entirely, as 'x / n'.

    Reported as a fraction rather than a bare percentage because a non-detect is
    a bound rather than a measurement: the reader needs the denominator to judge
    how much evidence sits behind it.
    """
    n = len(all_counts)
    zero = int((all_counts == 0).sum())
    if not n:
        return {'Non-detects': ''}
    return {'Non-detects': f'{zero:,} / {n:,} ({100 * zero / n:.1f}%)'}


def depth_all_samples(all_counts, detected):
    """Median and IQR over every sample, counting non-detects as zero."""
    return {
        'Median reads (all)': _count(all_counts.median()),
        'IQR (all)': _iqr(all_counts),
    }


def depth_detected(all_counts, detected):
    """Median and IQR over detecting samples only."""
    return {
        'Median reads (detected)': _count(detected.median()) if len(detected) else '',
        'IQR (detected)': _iqr(detected),
    }


def observed_range(all_counts, detected):
    """Full observed span across all samples.

    The lower bound is zero for every target in this panel, since all of them
    are absent from at least one sample; the informative half is the maximum.
    """
    if all_counts.empty:
        return {'Range (all)': ''}
    return {'Range (all)': f'{all_counts.min():,.0f}-{all_counts.max():,.0f}'}


# Column builders, applied per pathogen in table order.
METRICS = [non_detects, depth_all_samples, observed_range, depth_detected]


def build_detection_table(df=None):
    """Assemble the per-pathogen detection table over the full release.

    Args:
        df: Pre-loaded relative-abundance data, else loaded here. Loaded without
            temporal-outlier filtering so the denominator is the whole release.

    Returns:
        pd.DataFrame, one row per pathogen.
    """
    if df is None:
        df = load_all_relative_abundance()

    records = []
    for taxid, label in PATHOGENS:
        all_counts = df[df['taxid'] == taxid][COUNT_COL].dropna()
        detected = all_counts[all_counts > 0]
        record = {'Pathogen': label}
        for metric in METRICS:
            record.update(metric(all_counts, detected))
        records.append(record)

    return pd.DataFrame(records)


def main(save_path=OUTPUT_PATH):
    table = build_detection_table()
    print("Detection frequency and sequencing depth per pathogen\n")
    print(table.to_string(index=False))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_path, sep='\t', index=False)
        print(f"\nSaved {save_path}")
    return table


if __name__ == '__main__':
    main()
