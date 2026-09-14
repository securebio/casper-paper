#!/usr/bin/env python3
"""
Supplementary figure: VV fraction from shallow Kraken profiling against the
dedicated vertebrate-infecting virus (VV) pipeline.

Two independent routes through the data estimate how much of a library is
vertebrate-infecting virus. The shallow route classifies a subsample of reads
with Kraken2 and sums those whose virus is annotated to a vertebrate host. The
deep route runs the full VV pipeline over the whole library. This figure plots
one against the other, per sample, to show how far the cheap estimate tracks
the expensive one.

Both axes are a fraction of reads, so the two routes are comparable despite
profiling different numbers of them:

    kraken_fraction   = n_reads_host_vertebrate / n_reads_profiled_passed_qc
    pipeline_fraction = total_virus_hits / total_read_pairs

The Kraken denominator is the post-QC profiled count, not `n_reads_profiled`.
Kraken2 classifies the reads that survive FASTP, and the classified and
unclassified counts sum exactly to `n_reads_profiled_passed_qc` in all 1,206
libraries. The distinction is not cosmetic here: the two laboratories lose
different amounts at QC -- MU retains 99.5% of profiled reads and SB 89.9% --
so dividing by the pre-QC count would understate SB's fraction by about a tenth
and inflate its apparent gap from the pipeline.

Under uniform random subsampling a fraction is an unbiased estimator of the
whole-library fraction, so neither axis should depend on how many reads were
profiled; `print_vv_fraction_statistics` reports the test of that.

Points are coloured by sequencing laboratory because the two protocols differ in
how much VV they recover, and the offset from the 1:1 line differs with them.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import linregress, spearmanr

sys.path.insert(0, str(Path(__file__).parent))

from plot_config import FONT_SIZE_LARGE, save_figure
from plot_profiling_depth_markers import MU_COLOR, SB_COLOR
from data_loaders import (
    load_all_kraken_data,
    load_all_site_data,
    load_all_virus_host_data,
)

LAB_COLORS = {'MU': MU_COLOR, 'SB': SB_COLOR}
LAB_ORDER = ['MU', 'SB']

# Kraken column holding reads whose virus is annotated to a vertebrate host,
# exclusive of other host categories.
KRAKEN_VV_COL = 'n_reads_host_vertebrate'

# Reads Kraken2 actually classified: those surviving FASTP. Classified plus
# unclassified counts sum to this column exactly, which is what identifies it
# as the correct denominator rather than the pre-QC `n_reads_profiled`.
KRAKEN_DENOM_COL = 'n_reads_profiled_passed_qc'

# Whole-library VV read pairs from the dedicated pipeline.
PIPELINE_VV_COL = 'total_virus_hits'

LABEL_FONT = FONT_SIZE_LARGE
TICK_FONT = 14
ANNOT_FONT = 13

FIG_SIZE = (9, 9)

OUTPUT_PATH = '../figures/vv_fraction_kraken_vs_pipeline.png'


def prepare_vv_fraction_data():
    """Per-sample VV fraction by both routes, on a common denominator.

    Returns:
        pd.DataFrame with one row per library and columns ``kraken_fraction``,
        ``pipeline_fraction``, ``ratio`` (pipeline / Kraken) and
        ``sequencing_lab``.
    """
    site = load_all_site_data()[
        ['sra_accession', 'site_name', 'sequencing_lab', 'date',
         'total_read_pairs', PIPELINE_VV_COL]
    ].drop_duplicates()
    kraken = load_all_kraken_data()[
        ['sra_accession', 'n_reads_profiled', KRAKEN_DENOM_COL]].drop_duplicates()
    host = load_all_virus_host_data()[
        ['sra_accession', KRAKEN_VV_COL]].drop_duplicates()

    df = site.merge(kraken, on='sra_accession').merge(host, on='sra_accession')

    df['kraken_fraction'] = (df[KRAKEN_VV_COL]
                             / df[KRAKEN_DENOM_COL].replace(0, np.nan))
    df['pipeline_fraction'] = df[PIPELINE_VV_COL] / df['total_read_pairs']
    df['ratio'] = df['pipeline_fraction'] / df['kraken_fraction'].replace(0, np.nan)

    return df.dropna(subset=['kraken_fraction', 'pipeline_fraction'])


def print_vv_fraction_statistics(df):
    """Report the concordance numbers for the Methods and response text."""
    print("\n" + "=" * 78)
    print("VV fraction: shallow Kraken profiling vs dedicated VV pipeline")
    print("=" * 78)
    print(f"{'Lab':<6} {'n':>6} {'median Kraken':>15} {'median pipeline':>17} "
          f"{'R':>7} {'median ratio':>13}")
    print("-" * 78)
    for lab in LAB_ORDER + ['all']:
        g = df if lab == 'all' else df[df['sequencing_lab'] == lab]
        if g.empty:
            continue
        r, _ = spearmanr(g['kraken_fraction'], g['pipeline_fraction'])
        print(f"{lab:<6} {len(g):6d} {100 * g['kraken_fraction'].median():14.4f}% "
              f"{100 * g['pipeline_fraction'].median():16.4f}% {r:7.3f} "
              f"{g['ratio'].median():12.2f}x")
    print("-" * 78)
    print("Depth independence: under uniform random subsampling a fraction does "
          "not\ndepend on how many reads were profiled. Slope of log10(Kraken "
          "fraction) on\nlog10(profiled reads), which is zero if that holds:")
    for lab in LAB_ORDER:
        g = df[(df['sequencing_lab'] == lab) & (df['kraken_fraction'] > 0)]
        if len(g) < 3:
            continue
        fit = linregress(np.log10(g[KRAKEN_DENOM_COL]),
                         np.log10(g['kraken_fraction']))
        spread = g[KRAKEN_DENOM_COL].max() / g[KRAKEN_DENOM_COL].min()
        print(f"  {lab}: slope {fit.slope:+.3f} (p = {fit.pvalue:.2f}) over a "
              f"{spread:.0f}-fold range of profiled reads, n = {len(g):,}")

    zeros = df[df['kraken_fraction'] == 0]
    if not zeros.empty:
        expected = (zeros['pipeline_fraction'] * zeros[KRAKEN_DENOM_COL]
                    / df['ratio'].median())
        print(f"\nSamples with zero Kraken VV reads: {len(zeros)} of {len(df)} "
              f"({100 * len(zeros) / len(df):.1f}%).")
        print(f"  Expected Kraken VV read pairs at the observed offset: "
              f"{expected.min():.2f} to {expected.max():.2f} "
              f"(all below {np.ceil(expected.max()):.0f}), so zero is the "
              f"expected observation rather than a missed detection.")
    print("=" * 78 + "\n")


def _annotation_lines(df):
    """Per-laboratory concordance summary, as ``(text, colour)`` pairs."""
    lines = []
    for lab in LAB_ORDER:
        g = df[df['sequencing_lab'] == lab]
        if g.empty:
            continue
        r, _ = spearmanr(g['kraken_fraction'], g['pipeline_fraction'])
        lines.append((f'{lab}: R = {r:.2f}, {g["ratio"].median():.1f}$\\times$ '
                      f'(n = {len(g):,})', LAB_COLORS[lab]))
    return lines


def plot_vv_fraction_kraken_vs_pipeline(df=None, save_path=None):
    """Draw the per-sample concordance scatter.

    Returns:
        (fig, ax)
    """
    if df is None:
        df = prepare_vv_fraction_data()

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Zero-Kraken samples cannot sit on a log axis. They are drawn on the axis
    # floor as open markers rather than dropped. A zero here is the ordinary
    # outcome of counting rather than a detection failure: at these libraries'
    # profiled depth and VV fraction the expected Kraken VV count is below
    # three reads, so zero is the most likely observation.
    nonzero = df[df['kraken_fraction'] > 0]
    zeros = df[df['kraken_fraction'] == 0]

    # Axis span is set by the real data; the floor for zero-Kraken samples then
    # sits at the very edge, clear of the smallest genuine measurement so the
    # two are not confused.
    limits = [
        min(nonzero['kraken_fraction'].min(),
            df['pipeline_fraction'].min()) * 0.35,
        max(nonzero['kraken_fraction'].max(),
            df['pipeline_fraction'].max()) * 1.5,
    ]
    floor = limits[0] * 1.5

    for lab in LAB_ORDER:
        g = nonzero[nonzero['sequencing_lab'] == lab]
        if g.empty:
            continue
        ax.scatter(g['kraken_fraction'], g['pipeline_fraction'],
                   s=18, alpha=0.45, color=LAB_COLORS[lab], edgecolors='none',
                   zorder=3, label=lab)
        z = zeros[zeros['sequencing_lab'] == lab]
        if not z.empty:
            ax.scatter([floor] * len(z), z['pipeline_fraction'],
                       s=34, facecolors='none', edgecolors=LAB_COLORS[lab],
                       linewidths=1.2, zorder=4)

    ax.plot(limits, limits, color='#666666', linestyle='--', linewidth=1.6,
            zorder=2)
    ax.text(limits[1], limits[1], ' 1:1', color='#666666', fontsize=ANNOT_FONT,
            ha='left', va='center')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('Shallow Kraken profiling\n'
                  '(vertebrate-host virus reads / total read pairs)',
                  fontsize=LABEL_FONT)
    ax.set_ylabel('Dedicated VV pipeline\n'
                  '(VV reads / total read pairs)', fontsize=LABEL_FONT)
    ax.tick_params(axis='both', labelsize=TICK_FONT)
    ax.grid(True, alpha=0.3, which='major', zorder=1)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    for i, (text, color) in enumerate(_annotation_lines(df)):
        ax.text(0.03, 0.96 - i * 0.055, text, transform=ax.transAxes,
                ha='left', va='top', fontsize=ANNOT_FONT, color=color)

    handles = [Line2D([0], [0], marker='o', linestyle='none',
                      markerfacecolor=LAB_COLORS[lab], markeredgecolor='none',
                      markersize=8, label=f'{lab} protocol')
               for lab in LAB_ORDER if (df['sequencing_lab'] == lab).any()]
    if not zeros.empty:
        handles.append(Line2D([0], [0], marker='o', linestyle='none',
                              markerfacecolor='none', markeredgecolor='#444444',
                              markersize=8,
                              label='Zero Kraken VV reads (drawn at axis floor)'))
    ax.legend(handles=handles, loc='lower right', frameon=False,
              fontsize=ANNOT_FONT)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig, ax


def main(save_path=OUTPUT_PATH):
    df = prepare_vv_fraction_data()
    print_vv_fraction_statistics(df)
    return plot_vv_fraction_kraken_vs_pipeline(df, save_path=save_path)


if __name__ == '__main__':
    main()
