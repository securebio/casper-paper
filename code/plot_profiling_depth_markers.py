#!/usr/bin/env python3
"""
Profiling depth and normalization marker read counts (supplementary figure).

Shows, by sequencing laboratory:
- (a) the size of the Kraken taxonomic profiling subsample
- (b) Tobamovirus genus reads recovered in that subsample
- (c) PMMoV reads recovered in that subsample

Panels (b) and (c) share an x-axis so the two markers can be compared
directly. Guide lines mark the read counts at which Poisson counting
error reaches 10%, 5% and 1%, which is the quantity that matters when a
marker is used as a normalization denominator.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *  # noqa: F401,F403
from data_loaders import load_all_kraken_data

MU_COLOR = '#1f77b4'
SB_COLOR = '#ff7f0e'

TOBAMOVIRUS_COL = 'n_reads_tobamovirus_12234_clade_non_rrna'
PMMOV_COL = 'n_reads_pmmov_12239_clade_non_rrna'

# Poisson counting error 1/sqrt(N) reaches these values at N reads.
CV_GUIDES = [(100, '10%'), (400, '5%'), (10_000, '1%')]


def _log_hist(ax, mu_vals, sb_vals, lo, hi, nbins=40):
    """Overlaid MU/SB histogram on a log-spaced bin grid."""
    bins = np.logspace(np.log10(lo), np.log10(hi), nbins)
    ax.hist(mu_vals, bins=bins, color=MU_COLOR, alpha=0.75,
            edgecolor='none', label=f'MU (n={len(mu_vals):,})')
    ax.hist(sb_vals, bins=bins, color=SB_COLOR, alpha=0.75,
            edgecolor='none', label=f'SB (n={len(sb_vals):,})')
    ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _annotate_median(ax, vals, color, y_frac, label, ha='left'):
    """Dashed vertical line at the median, labelled."""
    m = float(np.median(vals))
    ax.axvline(m, color=color, linestyle='--', linewidth=1.6, zorder=5)
    pad = ' ' if ha == 'left' else ''
    suffix = '' if ha == 'left' else ' '
    ax.text(m, ax.get_ylim()[1] * y_frac,
            f'{pad}{label} median{suffix}\n{pad}{m:,.0f}{suffix}',
            color=color, fontsize=FONT_SIZE_SMALL - 1, va='top', ha=ha)


def _add_cv_guides(ax):
    """Vertical guides where Poisson counting error hits set thresholds."""
    for n, lab in CV_GUIDES:
        ax.axvline(n, color='#888888', linestyle=':', linewidth=1.2, zorder=1)
        ax.text(n, ax.get_ylim()[1] * 0.02, f' {lab}',
                color='#555555', fontsize=FONT_SIZE_SMALL - 2,
                rotation=90, va='bottom', ha='right')


def plot_profiling_depth_markers(kraken_data=None, save_path=None):
    """Three stacked histograms: profiling depth, Tobamovirus reads, PMMoV reads."""
    df = load_all_kraken_data() if kraken_data is None else kraken_data

    mu = df[df['sequencing_lab'] == 'MU']
    sb = df[df['sequencing_lab'] == 'SB']

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))

    # (a) profiling subsample size
    ax = axes[0]
    _log_hist(ax, mu['n_reads_profiled'], sb['n_reads_profiled'], 6e5, 2e7, nbins=45)
    ax.set_xlabel('Reads in taxonomic profiling subsample', fontsize=FONT_SIZE_BASE)
    ax.set_ylabel('Samples', fontsize=FONT_SIZE_BASE)
    _annotate_median(ax, mu['n_reads_profiled'], MU_COLOR, 0.95, 'MU', ha='right')
    _annotate_median(ax, sb['n_reads_profiled'], SB_COLOR, 0.60, 'SB')
    ax.legend(fontsize=FONT_SIZE_SMALL, frameon=False, loc='upper right')

    # (b) and (c) marker reads, shared x range
    lo, hi = 1, 2e6
    for ax, col, name in [(axes[1], TOBAMOVIRUS_COL, 'Tobamovirus genus'),
                          (axes[2], PMMOV_COL, 'PMMoV')]:
        _log_hist(ax, mu[col].clip(lower=lo), sb[col].clip(lower=lo), lo, hi)
        ax.set_xlabel(f'{name} reads in profiling subsample', fontsize=FONT_SIZE_BASE)
        ax.set_ylabel('Samples', fontsize=FONT_SIZE_BASE)
        _add_cv_guides(ax)
        _annotate_median(ax, mu[col], MU_COLOR, 0.95, 'MU')
        _annotate_median(ax, sb[col], SB_COLOR, 0.60, 'SB')
        n_zero = int((df[col] == 0).sum())
        if n_zero:
            ax.text(0.02, 0.72, f'{n_zero} samples with zero reads',
                    transform=ax.transAxes, fontsize=FONT_SIZE_SMALL - 1,
                    color='#B22222')

    for ax, letter in zip(axes, 'abc'):
        ax.text(-0.09, 1.02, letter, transform=ax.transAxes,
                fontsize=FONT_SIZE_LARGE, fontweight='bold', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f'Saved figure to {save_path}')
    return fig, axes


def print_marker_depth_statistics(kraken_data=None):
    """Print the read-count statistics quoted in the Results text."""
    df = load_all_kraken_data() if kraken_data is None else kraken_data
    print('Profiling subsample size (reads):')
    for lab, g in df.groupby('sequencing_lab'):
        s = g['n_reads_profiled']
        print(f'  {lab}: median {s.median():,.0f}  IQR {s.quantile(.25):,.0f}-'
              f'{s.quantile(.75):,.0f}  n={len(g)}')
    print('\nMarker reads in the profiling subsample:')
    for name, col in [('Tobamovirus genus', TOBAMOVIRUS_COL), ('PMMoV', PMMOV_COL)]:
        s = df[col]
        cv5 = 100 / np.sqrt(max(s.quantile(.05), 1))
        print(f'  {name}: median {s.median():,.0f}, 5th pct {s.quantile(.05):,.0f} '
              f'(counting error {cv5:.1f}%), min {s.min():,.0f}, '
              f'{int((s == 0).sum())} zero')
        for lab, g in df.groupby('sequencing_lab'):
            t = g[col]
            print(f'      {lab}: median {t.median():,.0f}  5th pct {t.quantile(.05):,.0f}')


if __name__ == '__main__':
    print_marker_depth_statistics()
    plot_profiling_depth_markers(save_path='../figures/profiling_depth_markers.png')
