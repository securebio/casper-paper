#!/usr/bin/env python3
"""
Data loading functions for reproducing CASPER manuscript figures.

Reads from the public flat-file dataset shipped with this repository.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — resolve relative to this file so it works from any CWD
# ---------------------------------------------------------------------------

_CODE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CODE_DIR.parent
_DATA_DIR = _REPO_ROOT / "data"
_SUPP_DIR = _REPO_ROOT / "tables"
_SRA_DIR = _DATA_DIR / "sra_statistics"


# ============================================================================
# Pathogen constants (shared across NWSS + clinical comparisons)
# ============================================================================

NWSS_PATHOGEN_MAP = {
    "sars-cov-2": {
        "label": "SARS-CoV-2",
        "taxids": [2697049],
        "nwss_col": "pcr_target_avg_conc_lin",
    },
    "influenza_a": {
        "label": "Influenza A",
        "taxids": [197911],
        "nwss_col": "pcr_target_avg_conc_lin",
    },
    "rsv": {
        "label": "RSV",
        "taxids": [3049954],
        "nwss_col": "pcr_target_avg_conc_lin",
    },
}

# CASPER BioProject accessions on SRA
CASPER_BIOPROJECTS = ["PRJNA1247874", "PRJNA1198001"]


# ============================================================================
# Core per-sample data loaders
# ============================================================================

def _load_csv(filename):
    """Load a CSV from the data directory."""
    path = _DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_all_site_data(**kwargs):
    """Load virus_hits_summary — per-sample read pair totals and virus hit stats.

    Merges in sequencing_lab from libraries_metadata.
    """
    df = _load_csv("virus_hits_summary.csv")
    meta_path = _DATA_DIR / "sample_metadata.csv"
    if not meta_path.exists():
        meta_path = _DATA_DIR / "libraries_metadata.csv"
    meta = pd.read_csv(meta_path)
    if "date" in meta.columns:
        meta["date"] = pd.to_datetime(meta["date"])
    if "sequencing_lab" not in df.columns and "sequencing_lab" in meta.columns:
        df = df.merge(
            meta[["sra_accession", "sequencing_lab"]].drop_duplicates(),
            on="sra_accession", how="left",
        )
    return df


def load_sample_metadata(**kwargs):
    """Load sample_metadata — per-sample sequencing and collection metadata."""
    # Accept both old and new filenames
    for fname in ("sample_metadata.csv", "libraries_metadata.csv"):
        path = _DATA_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
    raise FileNotFoundError("Neither sample_metadata.csv nor libraries_metadata.csv found")


def load_all_kraken_data(**kwargs):
    """Load kraken_summary — domain-level taxonomic composition per sample."""
    return _load_csv("kraken_summary.csv")


def load_all_vv_family_data(exclude_families=None, **kwargs):
    """Load vertebrate-virus family-level relative abundance.

    Args:
        exclude_families: List of family names to exclude (default: Dicistroviridae).
    """
    if exclude_families is None:
        exclude_families = ["Dicistroviridae"]

    df = _load_csv("relative_abundance_vv_families.csv")
    if exclude_families:
        df = df[~df["name"].isin(exclude_families)]
    return df


def load_all_relative_abundance(**kwargs):
    """Load merged relative abundance (mj + custom targets)."""
    return _load_csv("relative_abundance.csv")


def load_all_virus_host_data(**kwargs):
    """Load virus_host_summary — virus reads by host category per sample.

    Creates simplified n_reads_host_* columns from the detailed
    n_reads_virus_*_exclusive columns for downstream analysis.
    """
    df = _load_csv("virus_host_summary.csv")

    # Map detailed exclusive columns to simplified host categories
    _HOST_MAP = {
        "vertebrate": ["n_reads_virus_vertebrate_7742_exclusive"],
        "invertebrate": ["n_reads_virus_invertebrate_33208m7742_exclusive"],
        "plant": ["n_reads_virus_plant_33090_exclusive"],
        "bacteria": ["n_reads_virus_bacteria_2_exclusive"],
        "metagenome": ["n_reads_virus_metagenome_408169_exclusive"],
        "other": [
            "n_reads_virus_fungi_4751_exclusive",
            "n_reads_virus_archaea_2157_exclusive",
            "n_reads_virus_other_eukaryote_2759mMFP_exclusive",
            "n_reads_virus_virus_10239_exclusive",
        ],
        "unknown": [
            "n_reads_virus_host_root_1_exclusive",
            "n_reads_virus_unclassified_2787823m408169_exclusive",
            "n_reads_virus_other_entries_2787854_exclusive",
            "n_reads_virus_unknown_exclusive",
            "n_reads_virus_not_annotated",
        ],
    }
    for host_cat, source_cols in _HOST_MAP.items():
        present = [c for c in source_cols if c in df.columns]
        if present:
            df[f"n_reads_host_{host_cat}"] = df[present].sum(axis=1)

    return df


def load_all_qc_data(**kwargs):
    """Load qc_basic_stats — FastQC summary statistics per sample."""
    return _load_csv("qc_basic_stats.csv")


def load_sample_age_data(**kwargs):
    """Load sample turnaround time (sample_age) from sample metadata."""
    meta = load_sample_metadata()
    if "sample_age" not in meta.columns:
        raise ValueError("sample_age column not found in metadata")
    return meta[meta["sample_age"].notna()].copy()


def load_all_quality_data(stage="cleaned", **kwargs):
    """Load per-sample quality score distributions.

    Returns a DataFrame with median quality scores per sample.
    Falls back to qc_basic_stats if detailed quality data is unavailable.
    """
    # Try detailed quality distribution file first
    path = _DATA_DIR / "qc_quality_distribution.csv"
    if path.exists():
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if stage and "stage" in df.columns:
            df = df[df["stage"] == stage]
        return df

    # Fallback: derive from basic stats
    qc = load_all_qc_data()
    return qc


# ============================================================================
# Pathogen-level loader (for NWSS / clinical comparisons)
# ============================================================================

def load_mgs_pathogen_data(taxids=None, **kwargs):
    """Load MGS pathogen abundance for specific taxids.

    Args:
        taxids: List of NCBI taxonomy IDs to filter to (None = all).

    Returns:
        DataFrame with columns including site_name, sra_accession, date,
        taxid, ra_clade_pmmov_norm, etc.
    """
    df = load_all_relative_abundance()
    if taxids is not None:
        df = df[df["taxid"].isin(taxids)]
    return df


# ============================================================================
# NWSS data loaders
# ============================================================================

def load_nwss_data(**kwargs):
    """Load pre-matched NWSS wastewater PCR data.

    Returns:
        dict mapping pathogen key -> DataFrame with NWSS measurements.
    """
    path = _DATA_DIR / "nwss_pcr_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"NWSS data not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    result = {}
    for pathogen_key in df["pathogen"].unique():
        result[pathogen_key] = df[df["pathogen"] == pathogen_key].copy()

    return result


def load_nwss_site_matches():
    """Load the mapping between CASPER sites and NWSS sewershed IDs.

    Returns:
        DataFrame with columns: site_name, sewershed_id.
    """
    path = _SUPP_DIR / "sewershed_ids.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Sewershed ID mapping not found: {path}")

    df = pd.read_csv(path, sep="\t")
    # Rename columns for consistency
    col_map = {"Site": "site_name", "NWSS Sewershed ID": "sewershed_id"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    return df


def load_mgs_nwss_matches(**kwargs):
    """Load MGS-to-NWSS site matches.

    Returns a DataFrame compatible with the plotting scripts' expectations:
    columns include site_name, state, sewershed_ids (as list).
    """
    sewershed_df = load_nwss_site_matches()
    site_meta = load_site_metadata()

    # Merge to get state info
    merged = sewershed_df.merge(
        site_meta[["site_name", "state"]].drop_duplicates(),
        on="site_name",
        how="left",
    )

    # Convert sewershed_id to list format (for compatibility)
    merged["sewershed_ids"] = merged["sewershed_id"].apply(
        lambda x: [int(x)] if pd.notna(x) else []
    )

    return merged


def filter_to_dominant_source(site_nwss):
    """Keep only the most frequently reporting NWSS data source."""
    if "source" not in site_nwss.columns or site_nwss.empty:
        return site_nwss
    source_counts = site_nwss["source"].value_counts()
    if len(source_counts) <= 1:
        return site_nwss
    dominant = source_counts.index[0]
    return site_nwss[site_nwss["source"] == dominant].copy()


# ============================================================================
# Clinical data loader
# ============================================================================

def load_clinical_positives_tests_data(filter_test_types=True,
                                       drop_last_n_weeks=0, **kwargs):
    """Load clinical respiratory virus testing data.

    The public export contains pre-filtered weekly positive counts and total
    tests from Massachusetts General Hospital.

    Args:
        filter_test_types: Ignored (pre-filtered in export). Kept for API compat.
        drop_last_n_weeks: Drop last N weeks (0 = keep all).

    Returns:
        DataFrame pivoted with date index and pathogen columns containing
        positive counts, matching the internal loader's output format.
    """
    path = _DATA_DIR / "clinical_testing_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Clinical data not found: {path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    if drop_last_n_weeks > 0 and not df.empty:
        unique_dates = sorted(df["date"].unique())
        if len(unique_dates) > drop_last_n_weeks:
            dates_to_drop = unique_dates[-drop_last_n_weeks:]
            df = df[~df["date"].isin(dates_to_drop)]

    # Pivot to wide format: date + pathogen columns
    pivot = df.pivot_table(
        index="date", columns="pathogen", values="positives", aggfunc="sum"
    ).reset_index()
    pivot = pivot.fillna(0)

    return pivot


# ============================================================================
# Site metadata
# ============================================================================

def load_site_metadata():
    """Load site-level metadata (coordinates, population, type, region)."""
    return pd.read_csv(_DATA_DIR / "site_metadata.csv")


def load_site_coordinates(**kwargs):
    """Load site coordinates for mapping.

    Returns:
        DataFrame with site_name, lat, lon, state, region, etc.
    """
    return load_site_metadata()


def get_sequencing_lab(site_name, sra_accession=None):
    """Look up sequencing lab for a site/sample.

    For most sites there is only one lab. For Boston, samples from both
    MU and SB exist — use the sample_metadata to disambiguate.
    """
    meta = load_sample_metadata()
    if sra_accession is not None:
        row = meta[meta["sra_accession"] == sra_accession]
        if not row.empty:
            return row["sequencing_lab"].iloc[0]

    site_labs = meta[meta["site_name"] == site_name]["sequencing_lab"].unique()
    if len(site_labs) == 1:
        return site_labs[0]
    return list(site_labs)


# ============================================================================
# SRA statistics loaders
# ============================================================================

def load_sra_timeline(by_collection_date=False, **kwargs):
    """Load SRA monthly timeline data.

    Adds a ``date`` column (first of month) derived from year/month columns.
    """
    if by_collection_date:
        fname = "sra_monthly_timeline_by_collection.csv"
    else:
        fname = "sra_monthly_timeline.csv"
    path = _SRA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"SRA timeline not found: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns and "year" in df.columns and "month" in df.columns:
        df["date"] = pd.to_datetime(
            df[["year", "month"]].assign(day=1)
        )
    return df


def load_sra_summary_statistics(**kwargs):
    """Load SRA summary statistics."""
    path = _SRA_DIR / "sra_summary_statistics.csv"
    if not path.exists():
        raise FileNotFoundError(f"SRA summary not found: {path}")
    return pd.read_csv(path)


# ============================================================================
# Taxonomic composition utilities
# ============================================================================

def prepare_taxonomic_fractions_rrna_separated(kraken_data):
    """Calculate taxonomic fractions with rRNA as a separate category.

    Creates fractions where ribosomal is its own category and all other
    categories use non-rRNA reads only.
    """
    df = kraken_data.copy()

    df["fraction_ribosomal"] = (
        df["fraction_classified_1_clade_rrna"]
        + df["fraction_unclassified_0_clade_rrna"]
    )
    df["fraction_bacteria_non_rrna"] = df["fraction_bacteria_2_clade_non_rrna"]
    df["fraction_archaea_non_rrna"] = df["fraction_archaea_2157_clade_non_rrna"]
    df["fraction_eukaryota_non_rrna"] = df["fraction_eukaryota_2759_clade_non_rrna"]
    df["fraction_virus_non_rrna"] = df["fraction_viruses_10239_clade_non_rrna"]
    df["fraction_unclassified_non_rrna"] = (
        df["fraction_unclassified_0_clade_non_rrna"]
        + df["fraction_root_direct_1_non_rrna"]
        + df["fraction_cellular_organisms_131567_direct_non_rrna"]
    )

    # Print QC-filtered fraction stats if column is available
    if "fraction_qc_filtered" in df.columns:
        qc = df["fraction_qc_filtered"]
        print(f"  QC-filtered fraction (combined): mean={qc.mean():.4f}  median={qc.median():.4f}  "
              f"range=[{qc.min():.4f}, {qc.max():.4f}]  (n={len(qc)})")
        if "sequencing_lab" in df.columns:
            for lab, g in df.groupby("sequencing_lab"):
                q = g["fraction_qc_filtered"]
                print(f"    {str(lab):6s}  mean={q.mean():.4f}  median={q.median():.4f}  "
                      f"range=[{q.min():.4f}, {q.max():.4f}]  (n={len(q)})")

    # Renormalize so categories sum to 1 (excludes QC-filtered reads and
    # minor categories like unclassified_entries/other_entries)
    cat_cols = [
        "fraction_ribosomal", "fraction_bacteria_non_rrna",
        "fraction_archaea_non_rrna", "fraction_eukaryota_non_rrna",
        "fraction_virus_non_rrna", "fraction_unclassified_non_rrna",
    ]
    cat_sum = df[cat_cols].sum(axis=1)
    for col in cat_cols:
        df[col] = df[col] / cat_sum

    return df


def aggregate_taxonomic_composition(merged_data, groupby=None):
    """Calculate read-count-weighted taxonomic composition.

    Scales kraken fractions by total_read_pairs, then sums across samples.
    """
    df = merged_data.copy()
    if df.empty:
        return None

    categories = ["ribosomal", "bacteria", "archaea", "eukaryota", "virus", "unclassified"]

    for cat in categories:
        frac_col = f"fraction_{cat}" if cat == "ribosomal" else f"fraction_{cat}_non_rrna"
        df[f"reads_{cat}"] = df[frac_col] * df["total_read_pairs"]

    if groupby is None:
        total_reads = {cat: df[f"reads_{cat}"].sum() for cat in categories}
        total = sum(total_reads.values())
        proportions = {k: (v / total) if total > 0 else 0 for k, v in total_reads.items()}
        return {
            "proportions": proportions,
            "read_counts": total_reads,
            "total_reads": total,
            "n_samples": len(df),
            "n_sites": df["site_name"].nunique() if "site_name" in df.columns else 1,
        }
    else:
        results = []
        for group_val, group_df in df.groupby(groupby):
            total_reads = {cat: group_df[f"reads_{cat}"].sum() for cat in categories}
            total = sum(total_reads.values())
            proportions = {k: (v / total) if total > 0 else 0 for k, v in total_reads.items()}
            proportions[groupby] = group_val
            proportions["n_samples"] = len(group_df)
            results.append(proportions)
        return pd.DataFrame(results)


def aggregate_taxonomic_composition_for_site(merged_data, site_name):
    """Calculate read-count-weighted taxonomic composition for one site."""
    site_data = merged_data[merged_data["site_name"] == site_name]
    return aggregate_taxonomic_composition(site_data)


# ============================================================================
# Virus host composition utilities
# ============================================================================

def prepare_host_fractions(virus_host_data):
    """Extract host category columns from virus_host_summary.

    Combines metagenome + unknown into unknown_ambiguous.
    """
    df = virus_host_data.copy()

    host_cols = [c for c in df.columns if c.startswith("n_reads_host_")]
    if not host_cols:
        raise ValueError("No n_reads_host_* columns found in virus_host_data")

    # Combine metagenome and unknown
    if "n_reads_host_metagenome" in df.columns and "n_reads_host_unknown" in df.columns:
        df["n_reads_host_unknown_ambiguous"] = (
            df["n_reads_host_metagenome"] + df["n_reads_host_unknown"]
        )

    return df


def aggregate_virus_host_composition(virus_host_data, virus_data, groupby=None):
    """Calculate read-count-weighted virus host composition.

    Uses exclusive host assignment columns, scaled by total_read_pairs/n_reads_profiled.
    """
    df = virus_host_data.copy()
    if df.empty:
        return None

    # Merge to get total_read_pairs
    merge_cols = ["sra_accession"] if "sra_accession" in df.columns else ["site_name", "date"]
    trp_cols = merge_cols + ["total_read_pairs"]
    trp_cols = [c for c in trp_cols if c in virus_data.columns]
    df = df.merge(
        virus_data[trp_cols].drop_duplicates(),
        on=[c for c in merge_cols if c in virus_data.columns],
        how="inner",
        suffixes=("", "_vd"),
    )
    # Use virus_data's total_read_pairs if both exist
    if "total_read_pairs_vd" in df.columns:
        df["total_read_pairs"] = df["total_read_pairs_vd"]
        df = df.drop(columns=["total_read_pairs_vd"])

    # Merge n_reads_profiled from kraken_summary if not present
    if "n_reads_profiled" not in df.columns:
        kraken = _load_csv("kraken_summary.csv")
        if kraken is not None and "n_reads_profiled" in kraken.columns:
            kr_merge = ["sra_accession"] if "sra_accession" in df.columns else ["site_name", "date"]
            kr_cols = [c for c in kr_merge if c in kraken.columns] + ["n_reads_profiled"]
            df = df.merge(
                kraken[kr_cols].drop_duplicates(),
                on=[c for c in kr_merge if c in kraken.columns],
                how="left",
            )

    source_categories = ["vertebrate", "invertebrate", "bacteria", "plant", "metagenome", "other", "unknown"]
    output_categories = ["vertebrate", "invertebrate", "bacteria", "plant", "other", "unknown_ambiguous"]

    scale_factor = df["total_read_pairs"] / df["n_reads_profiled"].replace(0, np.nan)
    scale_factor = scale_factor.fillna(0)

    for cat in source_categories:
        col = f"n_reads_host_{cat}"
        if col in df.columns:
            df[f"n_reads_host_{cat}_scaled"] = df[col] * scale_factor

    if groupby is None:
        total_reads = {}
        for cat in output_categories:
            if cat == "unknown_ambiguous":
                total_reads[cat] = (
                    df.get("n_reads_host_metagenome_scaled", pd.Series(0)).sum()
                    + df.get("n_reads_host_unknown_scaled", pd.Series(0)).sum()
                )
            else:
                total_reads[cat] = df.get(f"n_reads_host_{cat}_scaled", pd.Series(0)).sum()
        total = sum(total_reads.values())
        proportions = {k: (v / total) if total > 0 else 0 for k, v in total_reads.items()}

        raw_counts = {}
        for cat in output_categories:
            if cat == "unknown_ambiguous":
                raw_counts[cat] = (
                    df.get("n_reads_host_metagenome", pd.Series(0)).sum()
                    + df.get("n_reads_host_unknown", pd.Series(0)).sum()
                )
            else:
                raw_counts[cat] = df.get(f"n_reads_host_{cat}", pd.Series(0)).sum()

        return {
            "proportions": proportions,
            "read_counts": total_reads,
            "raw_counts": raw_counts,
            "total_virus_reads": total,
            "n_samples": len(df),
            "n_sites": df["site_name"].nunique() if "site_name" in df.columns else 1,
        }
    else:
        results = []
        for group_val, group_df in df.groupby(groupby):
            total_reads = {}
            for cat in output_categories:
                if cat == "unknown_ambiguous":
                    total_reads[cat] = (
                        group_df.get("n_reads_host_metagenome_scaled", pd.Series(0)).sum()
                        + group_df.get("n_reads_host_unknown_scaled", pd.Series(0)).sum()
                    )
                else:
                    total_reads[cat] = group_df.get(f"n_reads_host_{cat}_scaled", pd.Series(0)).sum()
            total = sum(total_reads.values())
            proportions = {k: (v / total) if total > 0 else 0 for k, v in total_reads.items()}
            proportions[groupby] = group_val
            proportions["n_samples"] = len(group_df)
            results.append(proportions)
        return pd.DataFrame(results)


# ============================================================================
# VV family utilities
# ============================================================================

def get_top_families(vv_family_data, n=10, exclude_families=None):
    """Get top N virus families by total clade counts."""
    if exclude_families is None:
        exclude_families = ["Dicistroviridae"]

    df = vv_family_data.copy()
    if exclude_families:
        df = df[~df["name"].isin(exclude_families)]

    family_totals = df.groupby("name")["clade_counts"].sum().sort_values(ascending=False)
    return family_totals.head(n).index.tolist()


def get_family_colors(families):
    """Generate color mapping for virus families."""
    import matplotlib.pyplot as plt

    # Use Set2 + Pastel1 for enough colors
    set2 = plt.cm.Set2.colors
    pastel1 = plt.cm.Pastel1.colors
    all_colors = list(set2) + list(pastel1)

    colors = {}
    for i, fam in enumerate(families):
        colors[fam] = all_colors[i % len(all_colors)]
    colors["Other"] = (0.85, 0.85, 0.85)

    return colors


def calculate_vv_fraction_per_library(vv_data, virus_data):
    """Compute VV fraction (total VV reads / total read pairs) per sample."""
    merge_key = "sra_accession" if "sra_accession" in vv_data.columns else "site_name"

    vv_totals = vv_data.groupby(merge_key)["clade_counts"].sum().reset_index()
    vv_totals = vv_totals.rename(columns={"clade_counts": "total_vv_reads"})

    keep_cols = [merge_key, "total_read_pairs"]
    if "site_name" in virus_data.columns and "site_name" not in keep_cols:
        keep_cols.append("site_name")
    trp = virus_data[keep_cols].drop_duplicates()
    merged = vv_totals.merge(trp, on=merge_key, how="inner")
    merged["vv_fraction"] = merged["total_vv_reads"] / merged["total_read_pairs"]

    return merged


# ============================================================================
# Manuscript statistics helpers
# ============================================================================

def print_sample_site_stats(site_data):
    """Print sample and site count statistics."""
    n_samples = site_data["sra_accession"].nunique() if "sra_accession" in site_data.columns else len(site_data)
    n_sites = site_data["site_name"].nunique()
    total_reads = site_data["total_read_pairs"].sum()

    print(f"Total samples: {n_samples:,}")
    print(f"Total sites: {n_sites}")
    print(f"Total read pairs: {total_reads:,.0f}")

    # Per-lab breakdown
    if "sequencing_lab" in site_data.columns:
        for lab in sorted(site_data["sequencing_lab"].unique()):
            lab_data = site_data[site_data["sequencing_lab"] == lab]
            n = lab_data["sra_accession"].nunique() if "sra_accession" in lab_data.columns else len(lab_data)
            ns = lab_data["site_name"].nunique()
            print(f"  {lab}: {n:,} samples from {ns} sites")

    # Median read pairs
    if "sra_accession" in site_data.columns:
        per_lib = site_data.drop_duplicates("sra_accession")
    else:
        per_lib = site_data
    median_rp = per_lib["total_read_pairs"].median()
    min_rp = per_lib["total_read_pairs"].min()
    max_rp = per_lib["total_read_pairs"].max()
    print(f"Median read pairs per sample: {median_rp:,.0f} ({median_rp/1e9:.2f}B), range [{min_rp:,.0f}, {max_rp:,.0f}]")


def print_sra_statistics():
    """Print CASPER's contribution to SRA wastewater sequencing."""
    stats = load_sra_summary_statistics()
    print("SRA statistics:")
    print(stats.to_string(index=False))


def print_lit_review_statistics():
    """Print literature review comparison statistics."""
    path = _SUPP_DIR / "lit_review_table.tsv"
    if path.exists():
        df = pd.read_csv(path, sep="\t")
        print(f"Literature review: {len(df)} studies")
        with_median = df[df["median_gb_untargeted"].notna()].copy()
        casper = with_median[with_median["bioproject"].isin(CASPER_BIOPROJECTS)]
        non_casper = with_median[~with_median["bioproject"].isin(CASPER_BIOPROJECTS)]
        if not casper.empty and not non_casper.empty:
            casper_median = casper.loc[casper["bioproject"] == "PRJNA1247874", "median_gb_untargeted"].iloc[0]
            next_highest = non_casper["median_gb_untargeted"].max()
            fold = casper_median / next_highest
            next_author = non_casper.loc[non_casper["median_gb_untargeted"].idxmax(), "author"]
            print(f"  CASPER median depth: {casper_median:.1f} Gb vs next highest {next_author}: {next_highest:.1f} Gb ({fold:.0f}-fold higher)")
    else:
        print("Literature review table not found")


def print_cost_statistics():
    """Print sequencing cost statistics."""
    path = _SUPP_DIR / "sequencing_cost_table.tsv"
    if path.exists():
        df = pd.read_csv(path, sep="\t")
        print(f"Cost data: {len(df)} entries")
    else:
        print("Cost table not found")


def print_rrna_statistics(kraken_data):
    """Print rRNA fraction statistics."""
    df = kraken_data.copy()
    median_rrna = df["rrna_fraction"].median()
    mean_rrna = df["rrna_fraction"].mean()
    print(f"rRNA fraction: median={median_rrna:.3f}, mean={mean_rrna:.3f}")

    if "sequencing_lab" in df.columns:
        for lab in sorted(df["sequencing_lab"].unique()):
            lab_data = df[df["sequencing_lab"] == lab]
            med = lab_data["rrna_fraction"].median()
            print(f"  {lab}: median rRNA={med:.3f} (n={len(lab_data)})")


def print_taxonomic_composition_statistics(kraken_data):
    """Print median and range for all major taxonomic categories.

    Uses renormalized fractions from prepare_taxonomic_fractions_rrna_separated
    so categories sum to 1.
    """
    df = prepare_taxonomic_fractions_rrna_separated(kraken_data)

    categories = [
        ("rRNA (ribosomal)", "fraction_ribosomal"),
        ("Unclassified (non-rRNA)", "fraction_unclassified_non_rrna"),
        ("Bacteria (non-rRNA)", "fraction_bacteria_non_rrna"),
        ("Viruses (non-rRNA)", "fraction_virus_non_rrna"),
        ("Eukaryota (non-rRNA)", "fraction_eukaryota_non_rrna"),
        ("Archaea (non-rRNA)", "fraction_archaea_non_rrna"),
    ]

    print("Taxonomic composition (renormalized fractions):")
    for label, col in categories:
        vals = df[col] * 100
        print(f"  {label}: median={vals.median():.1f}%  range=[{vals.min():.2f}%, {vals.max():.1f}%]")


def print_turnaround_stats(sample_age_data):
    """Print sample turnaround time statistics."""
    ages = sample_age_data["sample_age"].dropna()
    print(f"Sample turnaround (days): median={ages.median():.0f}, "
          f"mean={ages.mean():.1f}, range={ages.min():.0f}-{ages.max():.0f}")

    if "sequencing_lab" in sample_age_data.columns:
        for lab in sorted(sample_age_data["sequencing_lab"].unique()):
            lab_ages = sample_age_data[sample_age_data["sequencing_lab"] == lab]["sample_age"]
            print(f"  {lab}: median={lab_ages.median():.0f} days (n={len(lab_ages)})")


def print_quality_statistics(quality_data, stage="raw"):
    """Print sequencing quality statistics."""
    col = f"{stage}_mean_mean_seq_len" if f"{stage}_mean_mean_seq_len" in quality_data.columns else None
    if col:
        vals = quality_data[col].dropna()
        print(f"Read length ({stage}): median={vals.median():.0f}, range={vals.min():.0f}-{vals.max():.0f}")


def print_read_length_statistics(qc_data):
    """Print read length statistics."""
    print_quality_statistics(qc_data, stage="raw")


def generate_cost_table(save_path=None):
    """Generate manuscript cost table with derived columns.

    Reads the full sequencing_cost_table.tsv (Cost per Mb from NHGRI),
    filters to Jan 2007 onward, and adds 'Cost per median CASPER sample'.
    Saves as sequencing_cost_table_manuscript.tsv.

    The original file is kept intact for use by plot_sampling_timeline_sra_combined.
    """
    path = _SUPP_DIR / "sequencing_cost_table.tsv"
    if not path.exists():
        print("Cost table not found")
        return pd.DataFrame()

    # Get median CASPER sample size from lit review table
    lit_path = _SUPP_DIR / "lit_review_table.tsv"
    lit_df = pd.read_csv(lit_path, sep="\t")
    casper_median_gb = lit_df.loc[
        lit_df["bioproject"] == "PRJNA1247874", "median_gb_untargeted"
    ].iloc[0]
    casper_median_mb = casper_median_gb * 1e3

    df = pd.read_csv(path, sep="\t")

    # Filter to Jan 2007 onward, keep only first datapoint per year
    df["_date"] = pd.to_datetime(df["Date"], format="%b %Y")
    df = df[df["_date"] >= "2007-01-01"].copy()
    df = df.sort_values("_date")
    df["_year"] = df["_date"].dt.year
    df = df.drop_duplicates(subset="_year", keep="first")
    df = df.drop(columns=["_date", "_year"])

    # Parse 'Cost per Mb' to numeric dollars
    def parse_cost(s):
        s = str(s).strip().replace("$", "").replace(",", "")
        return float(s)

    cost_per_mb = df["Cost per Mb"].apply(parse_cost)
    cost_per_sample = cost_per_mb * casper_median_mb

    # Format cost per sample with $ and appropriate suffix
    def format_cost(val):
        if val >= 1e6:
            return f"${val/1e6:.2f}M"
        elif val >= 1e3:
            return f"${val/1e3:.2f}k"
        elif val >= 1:
            return f"${val:.2f}"
        else:
            return f"${val:.3f}"

    df["Cost per median CASPER sample"] = cost_per_sample.apply(format_cost)

    # Reorder so Source is rightmost
    df = df[["Date", "Cost per Mb", "Cost per median CASPER sample", "Source"]]

    if save_path:
        # Save manuscript version alongside the original
        out = Path(save_path)
        manuscript_path = out.parent / (out.stem + "_manuscript" + out.suffix)
        df.to_csv(manuscript_path, sep="\t", index=False)
        print(f"Saved manuscript cost table to {manuscript_path}")
    return df
