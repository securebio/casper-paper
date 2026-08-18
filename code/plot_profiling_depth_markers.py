#!/usr/bin/env python3
"""
Profiling depth and normalization marker read counts (supplementary figure).

Shows, by sequencing laboratory:
- (a) the size of the Kraken taxonomic profiling subsample
- (b) Tobamovirus genus reads recovered in that subsample

Dotted vertical lines mark the per-laboratory medians.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from plot_config import *  # noqa: F401,F403
from data_loaders import load_all_kraken_data

MU_COLOR = '#1f77b4'
SB_COLOR = '#ff7f0e'

TOBAMOVIRUS_COL = 'n_reads_tobamovirus_12234_clade_non_rrna'
PMMOV_COL = 'n_reads_pmmov_12239_clade_non_rrna'


def _log_hist(ax, mu_vals, sb_vals, lo, hi, nbins=40):
    """Overlaid MU/SB histogram on a log-spaced bin grid, with median lines."""
    bins = np.logspace(np.log10(lo), np.log10(hi), nbins)
    ax.hist(mu_vals, bins=bins, color=MU_COLOR, alpha=0.75,
            edgecolor='none', label=f'MU (n={len(mu_vals):,})')
    ax.hist(sb_vals, bins=bins, color=SB_COLOR, alpha=0.75,
            edgecolor='none', label=f'SB (n={len(sb_vals):,})')
    for vals, color in [(mu_vals, MU_COLOR), (sb_vals, SB_COLOR)]:
        ax.axvline(float(np.median(vals)), color=color, linestyle=':',
                   linewidth=1.8, zorder=5)
    ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_profiling_depth_markers(kraken_data=None, save_path=None):
    """Two stacked histograms: profiling depth and Tobamovirus reads."""
    df = load_all_kraken_data() if kraken_data is None else kraken_data

    mu = df[df['sequencing_lab'] == 'MU']
    sb = df[df['sequencing_lab'] == 'SB']

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    # (a) profiling subsample size
    ax = axes[0]
    _log_hist(ax, mu['n_reads_profiled'], sb['n_reads_profiled'], 6e5, 2e7, nbins=45)
    ax.set_xlabel('Read pairs in shallow taxonomic profiling subsample',
                  fontsize=FONT_SIZE_BASE)
    ax.set_ylabel('Samples', fontsize=FONT_SIZE_BASE)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='#888888', linestyle=':', linewidth=1.8))
    labels.append('Median')
    ax.legend(handles, labels, fontsize=FONT_SIZE_SMALL, frameon=False,
              loc='upper center')

    # (b) Tobamovirus genus reads recovered in that subsample
    ax = axes[1]
    lo, hi = 1, 2e6
    _log_hist(ax, mu[TOBAMOVIRUS_COL].clip(lower=lo),
              sb[TOBAMOVIRUS_COL].clip(lower=lo), lo, hi)
    ax.set_xlabel('Tobamovirus genus read pairs in shallow taxonomic profiling subsample',
                  fontsize=FONT_SIZE_BASE)
    ax.set_ylabel('Samples', fontsize=FONT_SIZE_BASE)

    for ax, letter in zip(axes, 'ab'):
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
