#!/usr/bin/env python
"""
Export supplementary literature review table for the CASPER manuscript.

Reads the pre-processed supplementary literature review table and prints
summary statistics. The table is already available in the supplementary_tables
directory; this script is provided for reference and verification.
"""

import pandas as pd
from pathlib import Path

# Paths relative to this file
_CODE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CODE_DIR.parent
_SUPP_DIR = _REPO_ROOT / "tables"

LIT_REVIEW_TSV = _SUPP_DIR / "lit_review_table.tsv"


def load_lit_review_table():
    """Load the supplementary literature review table.

    Returns:
        DataFrame with literature review data.
    """
    if not LIT_REVIEW_TSV.exists():
        raise FileNotFoundError(f"Literature review table not found: {LIT_REVIEW_TSV}")

    return pd.read_csv(LIT_REVIEW_TSV, sep='\t')


def print_summary(verbose=False):
    """Print summary statistics for the literature review table.

    Args:
        verbose: If True, print per-study details.
    """
    df = load_lit_review_table()

    # Count studies with bp stats
    has_median = df['median_gb_untargeted'].notna() & (df['median_gb_untargeted'] != '')
    has_total = df['total_gb_untargeted'].notna() & (df['total_gb_untargeted'] != '')
    n_bp = (has_median | has_total).sum()

    print(f"Literature review: {len(df)} studies ({n_bp} with bp stats)")

    if verbose:
        for _, row in df.iterrows():
            author = row.get('author', '')
            date = row.get('date_paper_published', '')
            instrument = row.get('instrument', '')
            median = row.get('median_gb_untargeted', '')
            total_u = row.get('total_gb_untargeted', '')
            bp_flag = "" if (pd.isna(median) or median == '') and (pd.isna(total_u) or total_u == '') else " [bp included]"
            print(f"  {author} ({date}): instrument={instrument}, "
                  f"median_gb={median}, total_gb={total_u}{bp_flag}")
