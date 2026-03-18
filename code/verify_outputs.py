#!/usr/bin/env python3
"""
Verify that public CASPER code produces tabular outputs matching reference tables.

Compares:
  1. NWSS correlation table (Supplementary Table S4)
  2. Clinical correlation table (Supplementary Table S6)

Reference tables in data/supplementary_tables/ were generated from the internal
analysis notebook. Small differences are expected due to data export rounding,
different NWSS data vintages, or MMWR week boundary effects.

Usage:
    python verify_outputs.py
"""

import sys
import os
import io
import re
import contextlib
import numpy as np
import pandas as pd

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loaders import (
    load_mgs_pathogen_data,
    load_nwss_data,
    load_mgs_nwss_matches,
    load_clinical_positives_tests_data,
)
from plot_mgs_nwss_supplementary_all import (
    generate_nwss_correlation_display_tables,
    _build_site_info_dict,
    NWSS_PATHOGEN_MAP,
)
from plot_mgs_nwss_panel import generate_clinical_correlation_display_tables

# Paths
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_CODE_DIR), "data")
_SUPP_DIR = os.path.join(os.path.dirname(_CODE_DIR), "tables")

R_TOLERANCE = 0.01
P_TOLERANCE = 0.01

PATHOGENS = ["SARS-CoV-2", "Influenza A", "RSV"]
NORMS = ["Raw", "PMMoV", "ToBRFV"]


def _r_p_columns():
    """Return list of (R_col, p_col) pairs."""
    cols = []
    for p in PATHOGENS:
        for n in NORMS:
            cols.append((f"{p} R ({n})", f"{p} p ({n})"))
    return cols


def _parse_sectioned_table(source, from_file=True):
    """Parse a correlation table preserving MU/SB section context.

    Handles two formats:
      1. Old: section header rows (``**MU-sequenced**``, ``**SB-sequenced**``)
      2. New: explicit ``sequencing_lab`` column (MU / SB)

    Returns DataFrame with a ``section`` column and ``site_clean`` column.
    """
    if from_file:
        df = pd.read_csv(source, sep="\t")
    else:
        df = source.copy()

    # Detect format: new format has a sequencing_lab column without header rows
    has_section_headers = df["Sampling site"].astype(str).str.startswith("**").any()
    has_lab_column = "sequencing_lab" in df.columns

    if has_lab_column and not has_section_headers:
        # New format: sequencing_lab column
        df["section"] = df["sequencing_lab"]
    else:
        # Old format: section header rows
        section = "MU"
        sections = []
        is_header = []
        for _, row in df.iterrows():
            site = str(row.get("Sampling site", ""))
            if site == "**MU-sequenced**":
                section = "MU"
                is_header.append(True)
            elif site == "**SB-sequenced**":
                section = "SB"
                is_header.append(True)
            else:
                is_header.append(False)
            sections.append(section)

        df["section"] = sections
        df = df[~pd.Series(is_header).values].copy()

    # Convert numeric columns
    for r_col, p_col in _r_p_columns():
        if r_col in df.columns:
            df[r_col] = pd.to_numeric(df[r_col], errors="coerce")
        if p_col in df.columns:
            df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
    if "N (weeks)" in df.columns:
        df["N (weeks)"] = pd.to_numeric(df["N (weeks)"], errors="coerce")
    for c in df.columns:
        if "freq (days)" in c:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean site names: strip (MU)/(SB) suffixes if present
    df["site_clean"] = df["Sampling site"].apply(
        lambda x: re.sub(r"\s*\((MU|SB)\)$", "", x) if pd.notna(x) else x
    )

    return df


def _compare_sectioned_tables(ref_df, gen_df, freq_col_prefix):
    """Compare reference and generated tables with section awareness.

    Returns (pass_count, fail_count, messages).
    """
    passes = 0
    fails = 0
    messages = []
    r_deltas = []  # Track (site, col, delta) for summary

    ref_keys = set(zip(ref_df["section"], ref_df["site_clean"]))
    gen_keys = set(zip(gen_df["section"], gen_df["site_clean"]))

    missing = sorted(ref_keys - gen_keys)
    extra = sorted(gen_keys - ref_keys)

    if missing:
        fails += len(missing)
        messages.append(f"  Missing site-section pairs ({len(missing)}):")
        for sec, site in missing:
            messages.append(f"    [{sec}] {site}")
    if extra:
        messages.append(f"  Extra site-section pairs ({len(extra)}):")
        for sec, site in extra:
            messages.append(f"    [{sec}] {site}")

    common_keys = ref_keys & gen_keys
    if not common_keys:
        fails += 1
        messages.append("  No common site-section pairs to compare")
        return passes, fails, messages

    for section, site in sorted(common_keys):
        ref_row = ref_df[(ref_df["section"] == section) & (ref_df["site_clean"] == site)].iloc[0]
        gen_row = gen_df[(gen_df["section"] == section) & (gen_df["site_clean"] == site)].iloc[0]
        label = f"[{section}] {site}"

        # Compare N (weeks)
        ref_n = ref_row.get("N (weeks)")
        gen_n = gen_row.get("N (weeks)")
        if pd.notna(ref_n) and pd.notna(gen_n):
            if int(ref_n) != int(gen_n):
                fails += 1
                messages.append(f"  FAIL: {label} N (weeks): ref={int(ref_n)}, gen={int(gen_n)}")
            else:
                passes += 1

        # Compare R and p values
        for r_col, p_col in _r_p_columns():
            if r_col not in ref_df.columns or r_col not in gen_df.columns:
                continue

            ref_r = ref_row.get(r_col)
            gen_r = gen_row.get(r_col)
            ref_p = ref_row.get(p_col)
            gen_p = gen_row.get(p_col)

            # Both NaN = match
            if pd.isna(ref_r) and pd.isna(gen_r):
                passes += 1
                continue

            # One NaN, other not
            if pd.isna(ref_r) != pd.isna(gen_r):
                fails += 1
                ref_str = "NaN" if pd.isna(ref_r) else f"{ref_r:.4f}"
                gen_str = "NaN" if pd.isna(gen_r) else f"{gen_r:.4f}"
                messages.append(f"  FAIL: {label} {r_col}: ref={ref_str}, gen={gen_str}")
                continue

            # Compare R
            delta_r = abs(ref_r - gen_r)
            if delta_r > R_TOLERANCE:
                fails += 1
                r_deltas.append((label, r_col, delta_r))
                messages.append(
                    f"  FAIL: {label} {r_col}: ref={ref_r:.4f}, gen={gen_r:.4f}, "
                    f"delta={delta_r:.4f}"
                )
            else:
                passes += 1

            # Compare p
            if pd.notna(ref_p) and pd.notna(gen_p):
                delta_p = abs(ref_p - gen_p)
                if delta_p > P_TOLERANCE:
                    fails += 1
                    messages.append(
                        f"  FAIL: {label} {p_col}: ref={ref_p:.6f}, gen={gen_p:.6f}, "
                        f"delta={delta_p:.6f}"
                    )
                else:
                    passes += 1

        # Compare frequency columns
        for freq_prefix in [f"Median {freq_col_prefix} freq (days)", "Median MGS freq (days)"]:
            if freq_prefix in ref_df.columns and freq_prefix in gen_df.columns:
                ref_f = ref_row.get(freq_prefix)
                gen_f = gen_row.get(freq_prefix)
                if pd.notna(ref_f) and pd.notna(gen_f):
                    if abs(ref_f - gen_f) > 1:
                        fails += 1
                        messages.append(
                            f"  FAIL: {label} {freq_prefix}: ref={ref_f}, gen={gen_f}"
                        )
                    else:
                        passes += 1

    # Summary of R value deltas
    if r_deltas:
        messages.append("")
        messages.append("  R-value delta summary (only failures):")
        messages.append(f"    Max delta: {max(d for _, _, d in r_deltas):.4f}")
        messages.append(f"    Median delta: {np.median([d for _, _, d in r_deltas]):.4f}")
        messages.append(f"    Sites with largest deltas:")
        for label, col, delta in sorted(r_deltas, key=lambda x: -x[2])[:5]:
            messages.append(f"      {label} {col}: {delta:.4f}")

    return passes, fails, messages


def verify_nwss_correlations():
    """Verify NWSS correlation table."""
    print("=" * 70)
    print("NWSS Correlation Table (Supplementary Table S4)")
    print("=" * 70)

    ref_path = os.path.join(_SUPP_DIR, "nwss_correlation_table.tsv")
    if not os.path.exists(ref_path):
        print("  SKIP: Reference file not found")
        return 0, 0

    ref_df = _parse_sectioned_table(ref_path)

    # Generate from public code
    print("  Loading data...")
    nwss_data = load_nwss_data()
    matches = load_mgs_nwss_matches()

    # Use _build_site_info_dict for proper seq_lab detection
    site_info_dict = _build_site_info_dict(matches)

    mgs_data_dict = {}
    for pathogen_key in ["sars-cov-2", "influenza_a", "rsv"]:
        pathogen_config = NWSS_PATHOGEN_MAP[pathogen_key]
        mgs_data = load_mgs_pathogen_data(taxids=pathogen_config["taxids"])
        if not mgs_data.empty:
            mgs_data_dict[pathogen_key] = mgs_data

    print("  Computing correlations...")
    with contextlib.redirect_stdout(io.StringIO()):
        tables = generate_nwss_correlation_display_tables(
            mgs_data_dict, nwss_data, site_info_dict
        )

    gen_df = _parse_sectioned_table(tables["raw"], from_file=False)

    passes, fails, messages = _compare_sectioned_tables(ref_df, gen_df, "PCR")

    for msg in messages:
        print(msg)

    status = "PASS" if fails == 0 else "FAIL"
    print(f"\n  Result: {status} ({passes} checks passed, {fails} failed)")
    return passes, fails


def verify_clinical_correlations():
    """Verify clinical correlation table."""
    print("\n" + "=" * 70)
    print("Clinical Correlation Table (Supplementary Table S6)")
    print("=" * 70)

    ref_path = os.path.join(_SUPP_DIR, "clinical_correlation_table.tsv")
    if not os.path.exists(ref_path):
        print("  SKIP: Reference file not found")
        return 0, 0

    ref_df = _parse_sectioned_table(ref_path)

    # Generate from public code
    print("  Loading data...")
    mgs_data_dict = {}
    for pathogen_key in ["sars-cov-2", "influenza_a", "rsv"]:
        pathogen_config = NWSS_PATHOGEN_MAP[pathogen_key]
        mgs_data = load_mgs_pathogen_data(taxids=pathogen_config["taxids"])
        if not mgs_data.empty:
            mgs_data_dict[pathogen_key] = mgs_data

    clinical_data = load_clinical_positives_tests_data()

    print("  Computing correlations...")
    with contextlib.redirect_stdout(io.StringIO()):
        tables = generate_clinical_correlation_display_tables(
            mgs_data_dict, clinical_data
        )

    gen_df = _parse_sectioned_table(tables["raw"], from_file=False)

    passes, fails, messages = _compare_sectioned_tables(ref_df, gen_df, "clinical")

    for msg in messages:
        print(msg)

    status = "PASS" if fails == 0 else "FAIL"
    print(f"\n  Result: {status} ({passes} checks passed, {fails} failed)")
    return passes, fails


def main():
    print("CASPER Public Code Output Verification")
    print("=" * 70)
    print(f"Tolerance: |dR| < {R_TOLERANCE}, |dp| < {P_TOLERANCE}")
    print()

    total_passes = 0
    total_fails = 0

    p, f = verify_nwss_correlations()
    total_passes += p
    total_fails += f

    p, f = verify_clinical_correlations()
    total_passes += p
    total_fails += f

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    total = total_passes + total_fails
    print(f"  Total checks: {total}")
    print(f"  Passed: {total_passes} ({100*total_passes/total:.1f}%)" if total > 0 else "  Passed: 0")
    print(f"  Failed: {total_fails} ({100*total_fails/total:.1f}%)" if total > 0 else "  Failed: 0")

    if total_fails == 0:
        print("\n  ALL CHECKS PASSED")
    else:
        print(f"\n  {total_fails} CHECK(S) FAILED")
        print("  Note: Some differences are expected between internal and public data exports.")

    return 0 if total_fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
