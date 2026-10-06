#!/usr/bin/env python3
"""Regenerate every figure in figures/ and every table in tables/.

Run from anywhere:

    python code/run_all.py              # everything
    python code/run_all.py --list       # show the stages
    python code/run_all.py fig-4 tbl-s4 # named stages only

This produces the same files as rendering code/index.qmd, without Quarto.
Each stage prints its name and the files it wrote.
"""

import argparse
import contextlib
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
FIGURES = REPO_ROOT / "figures"
TABLES = REPO_ROOT / "tables"
BENCHMARK = REPO_ROOT / "benchmark"

sys.path.insert(0, str(CODE_DIR))

from data_loaders import (  # noqa: E402
    NWSS_PATHOGEN_MAP,
    load_all_kraken_data,
    load_all_qc_data,
    load_all_quality_data,
    load_all_site_data,
    load_all_virus_host_data,
    load_all_vv_family_data,
    load_clinical_positives_tests_data,
    load_mgs_nwss_matches,
    load_mgs_pathogen_data,
    load_nwss_data,
    load_sample_age_data,
    load_sample_metadata,
    generate_cost_table,
)


def load_inputs():
    """Read the datasets the stages share."""
    return {
        "site_data": load_all_site_data(),
        "kraken_data": load_all_kraken_data(),
        "qc_data": load_all_qc_data(),
        "virus_host_data": load_all_virus_host_data(),
        "vv_family_data": load_all_vv_family_data(),
        "clinical_data": load_clinical_positives_tests_data(),
    }


def pathogen_series(keys):
    """MGS abundance per pathogen key, skipping targets with no data."""
    out = {}
    for key in keys:
        data = load_mgs_pathogen_data(taxids=NWSS_PATHOGEN_MAP[key]["taxids"])
        if not data.empty:
            out[key] = data
    return out


# --- Main figures and Table 1 ------------------------------------------------

def sampling_sites(d):
    from plot_sample_summary import generate_table1_sampling_sites

    generate_table1_sampling_sites(
        d["site_data"], save_path=str(TABLES / "sampling_sites.tsv"))


def sampling_map(d):
    from plot_sampling_map import (
        add_cluster_offsets, load_site_coordinates, plot_us_sampling_map,
    )

    plot_us_sampling_map(add_cluster_offsets(load_site_coordinates()),
                         save_path=str(FIGURES / "sampling_map.png"))


def population_served(d):
    from plot_population_served import load_population_data, plot_population_served

    plot_population_served(load_population_data(),
                           save_path=str(FIGURES / "population_served.png"))


def sampling_timeline(d):
    from plot_sampling_timeline_sra_combined import plot_sampling_timeline_sra_combined

    plot_sampling_timeline_sra_combined(
        save_path=str(FIGURES / "sampling_timeline_sra_combined.png"))


def taxonomic_panel(d):
    from plot_taxonomic_panel_combined import plot_taxonomic_panel_combined

    plot_taxonomic_panel_combined(
        d["kraken_data"], d["virus_host_data"], d["site_data"], d["site_data"],
        save_path=str(FIGURES / "taxonomic_panel_combined.png"))


def vv_pathogen(d):
    from plot_vv_pathogen_combined import plot_vv_pathogen_combined

    plot_vv_pathogen_combined(save_path=str(FIGURES / "vv_pathogen_combined.png"),
                              log_scale=False, normalize_by_site_median=True)


def pcr_clinical_lineage(d):
    from plot_mgs_pcr_clinical_lineage_panel import build

    build(output=str(FIGURES / "mgs_pcr_clinical_lineage_panel.png"))


# --- Supplementary figures ---------------------------------------------------

def read_pairs(d):
    from plot_read_pairs_timeseries_all import plot_read_pairs_timeseries_all

    plot_read_pairs_timeseries_all(
        d["site_data"], save_path=str(FIGURES / "read_pairs_timeseries_all.png"))


def sra_untargeted(d):
    from plot_sra_untargeted_supplement import plot_sra_untargeted_supplement

    plot_sra_untargeted_supplement(
        save_path=str(FIGURES / "sra_untargeted_supplement.png"))


def sample_age(d):
    from plot_sample_age import plot_sample_age_histogram

    plot_sample_age_histogram(
        load_sample_age_data(), save_path=str(FIGURES / "sample_age_histogram.png"))


def taxonomic_composition(d):
    from plot_taxonomic_composition_timeseries_all import (
        plot_taxonomic_composition_timeseries_all,
    )

    plot_taxonomic_composition_timeseries_all(
        d["kraken_data"],
        save_path=str(FIGURES / "taxonomic_composition_timeseries_all.png"))


def profiling_depth(d):
    from plot_profiling_depth_markers import plot_profiling_depth_markers

    plot_profiling_depth_markers(
        d["kraken_data"], save_path=str(FIGURES / "profiling_depth_markers.png"))


def vv_family_timeseries(d):
    from plot_vv_family_timeseries_all import plot_vv_family_timeseries_all

    plot_vv_family_timeseries_all(
        d["vv_family_data"], save_path=str(FIGURES / "vv_family_timeseries_all.png"))


def vv_pathogen_log(d):
    from plot_vv_pathogen_log import plot_vv_pathogen_log

    plot_vv_pathogen_log(save_path=str(FIGURES / "vv_pathogen_log.png"))


def vv_pathogen_read_counts(d):
    from plot_vv_pathogen_read_counts import plot_vv_pathogen_read_counts

    plot_vv_pathogen_read_counts(
        save_path=str(FIGURES / "vv_pathogen_read_counts.png"))


def pcr_supplementary(d):
    """Also writes tables/mgs_pcr_supplementary_all_correlations.tsv."""
    from plot_mgs_pcr_supplementary_all import (
        _build_site_info_dict, plot_mgs_pcr_supplementary_all,
    )

    d["nwss_data_all"] = load_nwss_data()
    d["site_info_all"] = _build_site_info_dict(load_mgs_nwss_matches())
    d["mgs_all"] = pathogen_series(["sars-cov-2", "influenza_a", "rsv"])
    plot_mgs_pcr_supplementary_all(
        d["mgs_all"], d["nwss_data_all"], d["site_info_all"],
        save_path=str(FIGURES / "mgs_pcr_supplementary_all.png"))


def clinical_testing(d):
    from plot_clinical_testing_supplement import (
        PATHOGENS_TO_PLOT, load_clinical_testing_data,
        plot_clinical_testing_supplement,
    )

    d["mgs_clinical"] = pathogen_series(PATHOGENS_TO_PLOT)
    plot_clinical_testing_supplement(
        mgs_data_dict=d["mgs_clinical"],
        clinical_time_series=load_clinical_positives_tests_data(filter_test_types=True),
        clinical_testing_data=load_clinical_testing_data(),
        metadata=load_sample_metadata(),
        save_path=str(FIGURES / "clinical_testing_supplement.png"))


def gc_content(d):
    from plot_gc_content_timeseries_all import plot_gc_content_timeseries_all

    plot_gc_content_timeseries_all(
        d["qc_data"], d["site_data"],
        save_path=str(FIGURES / "gc_content_timeseries_all.png"))


def quality_score(d):
    from plot_quality_score_timeseries_all import plot_quality_score_timeseries_all

    plot_quality_score_timeseries_all(
        load_all_quality_data(stage="raw"), d["site_data"],
        save_path=str(FIGURES / "quality_score_timeseries_all.png"))


def qc_pass_rate(d):
    from plot_qc_pass_rate_timeseries_all import plot_qc_pass_rate_timeseries_all

    merged = d["kraken_data"].merge(
        d["qc_data"][["sra_accession", "raw_n_read_pairs", "cleaned_n_read_pairs"]],
        on="sra_accession", how="left")
    merged["fraction_qc_filtered"] = 1 - (
        merged["cleaned_n_read_pairs"] / merged["raw_n_read_pairs"])
    plot_qc_pass_rate_timeseries_all(
        merged, d["site_data"],
        save_path=str(FIGURES / "qc_pass_rate_timeseries_all.png"))


def read_length(d):
    from plot_read_length_timeseries_all import plot_read_length_timeseries_all

    plot_read_length_timeseries_all(
        load_all_qc_data(), d["site_data"],
        save_path=str(FIGURES / "read_length_timeseries_all.png"))


def protocol_comparison(d):
    """Also writes tables/boston_protocol_comparison{,_raw}.tsv."""
    from plot_boston_protocol_comparison import (
        PATHOGEN_KEYS, SITES, plot_boston_protocol_comparison,
    )

    matches = load_mgs_nwss_matches()
    sewersheds = sorted({
        int(sid)
        for _, row in matches[matches["site_name"].isin(SITES)].iterrows()
        for sid in row["sewershed_ids"]
    })
    nwss = {key: df[df["sewershed_id"].isin(sewersheds)].copy()
            for key, df in load_nwss_data().items()}
    plot_boston_protocol_comparison(
        pathogen_series(PATHOGEN_KEYS), nwss, d["clinical_data"],
        save_path=str(FIGURES / "boston_protocol_comparison.png"),
        table_path=str(TABLES / "boston_protocol_comparison.tsv"))


def vv_fraction_methods(d):
    from plot_vv_fraction_kraken_vs_pipeline import plot_vv_fraction_kraken_vs_pipeline

    plot_vv_fraction_kraken_vs_pipeline(
        save_path=str(FIGURES / "vv_fraction_kraken_vs_pipeline.png"))


def ribodepletion(d):
    from plot_ribodepletion_effect import (
        plot_ribodepletion_effect, prepare_ribodepletion_data,
    )

    d["ribodepletion_data"] = prepare_ribodepletion_data(
        d["kraken_data"], d["site_data"], d["vv_family_data"])
    plot_ribodepletion_effect(d["ribodepletion_data"],
                              save_path=str(FIGURES / "ribodepletion_effect.png"))


def pipeline_benchmark(d):
    sys.path.insert(0, str(BENCHMARK))
    from assignment_evaluation.plot_composition import plot_pipeline_benchmark

    plot_pipeline_benchmark(save_path=str(FIGURES / "pipeline_benchmark.png"))


# --- Supplementary tables ----------------------------------------------------

def cost_table(d):
    generate_cost_table(save_path=str(TABLES / "sequencing_cost_table_manuscript.tsv"))


def pcr_correlations(d):
    from plot_mgs_pcr_supplementary_all import generate_nwss_correlation_display_tables

    generate_nwss_correlation_display_tables(
        d["mgs_all"], d["nwss_data_all"], d["site_info_all"],
        save_path=str(TABLES / "nwss_correlation_table.tsv"))


def clinical_correlations(d):
    from plot_mgs_pcr_clinical_lineage_panel import (
        generate_clinical_correlation_display_tables,
    )

    generate_clinical_correlation_display_tables(
        d["mgs_clinical"], d["clinical_data"],
        save_path=str(TABLES / "clinical_correlation_table.tsv"))


def detection_table(d):
    from export_detection_table import build_detection_table

    build_detection_table().to_csv(
        TABLES / "pathogen_detection.tsv", sep="\t", index=False)


def ribodepletion_table(d):
    from export_ribodepletion_table import build_ribodepletion_table

    build_ribodepletion_table(d["ribodepletion_data"]).to_csv(
        TABLES / "ribodepletion_omitted_samples.tsv", sep="\t", index=False)


def benchmark_tables(d):
    sys.path.insert(0, str(BENCHMARK))
    from export_benchmark_table import build_benchmark_table, build_family_table

    build_benchmark_table().to_csv(
        TABLES / "pipeline_benchmark.tsv", sep="\t", index=False)
    build_family_table().to_csv(
        TABLES / "pipeline_benchmark_by_family.tsv", sep="\t", index=False)


# Order matches code/index.qmd. Stages marked needs_benchmark are skipped when
# the benchmark directory is absent.
STAGES = [
    ("tbl-1", sampling_sites, False),
    ("fig-1a", sampling_map, False),
    ("fig-1b", population_served, False),
    ("fig-2", sampling_timeline, False),
    ("fig-3", taxonomic_panel, False),
    ("fig-4", vv_pathogen, False),
    ("fig-5", pcr_clinical_lineage, False),
    ("fig-s1", read_pairs, False),
    ("fig-s2", sra_untargeted, False),
    ("fig-s3", sample_age, False),
    ("fig-s4", taxonomic_composition, False),
    ("fig-s5", profiling_depth, False),
    ("fig-s6", vv_family_timeseries, False),
    ("fig-s7", vv_pathogen_log, False),
    ("fig-s-read-counts", vv_pathogen_read_counts, False),
    ("fig-s8", pcr_supplementary, False),
    ("fig-s9", clinical_testing, False),
    ("fig-s10", gc_content, False),
    ("fig-s11", quality_score, False),
    ("fig-s12", qc_pass_rate, False),
    ("fig-s13", read_length, False),
    ("fig-s-protocol-comparison", protocol_comparison, False),
    ("fig-s-vv-fraction-methods", vv_fraction_methods, False),
    ("fig-s-pipeline-benchmark", pipeline_benchmark, True),
    ("fig-s-ribodepletion", ribodepletion, False),
    ("tbl-s2", cost_table, False),
    ("tbl-s4", pcr_correlations, False),
    ("tbl-s6", clinical_correlations, False),
    ("tbl-s-pathogen-detection", detection_table, False),
    ("tbl-s-protocol-comparison", protocol_comparison, False),
    ("tbl-s-pipeline-benchmark", benchmark_tables, True),
    ("tbl-s-ribodepletion-omitted", ribodepletion_table, False),
]

# Stages whose output another stage already wrote.
ALIASES = {"tbl-s-protocol-comparison"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stages", nargs="*", help="stage names; default is all")
    parser.add_argument("--list", action="store_true", help="list stage names")
    parser.add_argument("--verbose", action="store_true",
                        help="show the statistics each stage prints")
    args = parser.parse_args(argv)

    if args.list:
        for name, _, needs_benchmark in STAGES:
            mark = " (needs benchmark/)" if needs_benchmark else ""
            print(f"{name}{mark}")
        return 0

    selected = [s for s in STAGES if not args.stages or s[0] in args.stages]
    unknown = set(args.stages) - {s[0] for s in STAGES}
    if unknown:
        parser.error(f"unknown stage(s): {', '.join(sorted(unknown))}")

    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)

    print("Loading data")
    shared = load_inputs()

    failures = []
    for name, stage, needs_benchmark in selected:
        if name in ALIASES and not args.stages:
            continue
        if needs_benchmark and not BENCHMARK.is_dir():
            print(f"{name}: skipped, {BENCHMARK.name}/ is not present")
            continue
        print(f"{name}: running")
        try:
            if args.verbose:
                stage(shared)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    stage(shared)
        except Exception as exc:  # keep going so one failure does not hide others
            failures.append((name, exc))
            print(f"{name}: FAILED - {type(exc).__name__}: {exc}")
        finally:
            plt.close("all")

    if failures:
        print(f"\n{len(failures)} stage(s) failed:")
        for name, exc in failures:
            print(f"  {name}: {type(exc).__name__}: {exc}")
        return 1

    print(f"\nWrote figures to {FIGURES} and tables to {TABLES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
