"""Command-line entry point: score a pipeline run against the panels."""

#=======================================================================
# Imports
#=======================================================================

import argparse
import gzip
import json
from pathlib import Path

import polars as pl

from assignment_evaluation.evaluate import read_all_virus_hits
from assignment_evaluation.score import (
    READS_PER_BILLION,
    attribute_hits,
    false_positive_taxa,
    restrict_to_run,
    summarise_negative,
    summarise_resolution,
    summarise_resolution_by_family,
)
from assignment_evaluation.taxonomy import read_ncbi_taxonomy

#=======================================================================
# CLI
#=======================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m assignment_evaluation.cli",
        description="Score an mgs-workflow run against the benchmark panels.",
    )
    parser.add_argument(
        "--hits", type=Path, nargs="+", required=True,
        help="virus_hits_final.tsv.gz from the run (any number of files).",
    )
    parser.add_argument("--manifest", type=Path, default=Path("panels/panel_manifest.tsv"))
    parser.add_argument("--allocation", type=Path, default=Path("panels/read_allocation.tsv"))
    parser.add_argument(
        "--taxa", type=Path, default=Path("panels/positive_taxa.tsv"),
        help="Positive taxa table, for the per-family breakdown.",
    )
    parser.add_argument(
        "--read-counts", type=Path, required=True,
        help=(
            "read_counts.tsv.gz from the run. Identifies which panels the pipeline actually "
            "received, so a panel that was not run is not scored as though it were."
        ),
    )
    parser.add_argument("--taxonomy-nodes", type=Path, required=True)
    parser.add_argument("--taxonomy-names", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--bt2-score-threshold", type=float, required=True,
        help="The run's viral-hit calling threshold, recorded with the results.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    read = lambda path: pl.read_csv(path, separator="\t", infer_schema_length=0)  # noqa: E731
    manifest, allocation = read(args.manifest), read(args.allocation)
    with gzip.open(args.read_counts, "rt") as handle:
        read_counts = pl.read_csv(
            handle.read().encode(), separator="\t", infer_schema_length=0,
        )
    allocation = restrict_to_run(allocation, read_counts)

    print(f"reading {len(args.hits)} hits file(s) ...")
    hits = read_all_virus_hits(args.hits)
    print(f"  {hits.height:,} viral hits")

    print("loading taxonomy (~20 s, ~2.3 GiB) ...")
    taxonomy = read_ncbi_taxonomy(args.taxonomy_nodes, args.taxonomy_names)

    attributed = attribute_hits(hits, manifest)
    resolution = summarise_resolution(attributed, allocation, taxonomy)
    taxa = read(args.taxa)
    families = dict(
        zip(taxa["species_taxid"].cast(pl.Int64), taxa["family_name"], strict=True),
    )
    by_family = summarise_resolution_by_family(
        attributed, allocation, manifest, families, taxonomy,
    )
    scored_negative = "negative" in set(allocation["panel"])
    negative = (
        summarise_negative(attributed, allocation, manifest)
        if scored_negative
        else pl.DataFrame()
    )
    landed_on = false_positive_taxa(attributed, taxonomy)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[str, pl.DataFrame]] = [
        ("resolution", resolution),
        ("resolution_by_family", by_family),
        ("false_positive_taxa", landed_on),
    ]
    if scored_negative:
        outputs.append(("negative_per_genome", negative))
    for name, frame in outputs:
        frame.write_csv(args.out_dir / f"{name}.csv")

    observed = (
        float(negative["expected_per_billion"].fill_null(0).sum()) if scored_negative else None
    )
    bound = (
        float(negative["bound_per_billion"].fill_null(0).sum()) if scored_negative else None
    )
    (args.out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "hits_files": [str(path) for path in args.hits],
                "n_hits": hits.height,
                "bt2_score_threshold": args.bt2_score_threshold,
                "taxonomy_nodes": str(args.taxonomy_nodes),
                "spurious_per_billion_observed": observed,
                "spurious_per_billion_bound": bound,
            },
            indent=2,
        ),
    )

    print("\nhow precisely each read was identified (% of reads simulated):")
    print(
        resolution.with_columns(
            (100 * pl.col("n_reads") / pl.col("n_allocated")).round(2).alias("pct"),
        ).pivot(on="category", index="panel", values="pct"),
    )
    if scored_negative:
        print(
            f"\nnegative panel: {int(negative['n_false_positive'].sum()):,} false positives "
            f"over {int(negative['n_pairs'].cast(pl.Int64).sum()):,} pairs",
        )
        print(
            f"  spurious VV pairs per {READS_PER_BILLION:,} wastewater pairs: "
            f"{observed:,.0f} observed, {bound:,.0f} upper bound",
        )
    else:
        print("\nnegative panel was not part of this run; specificity is not reported")
    if landed_on.height:
        print("\nfalse positives were assigned to:")
        print(landed_on.head(10))
    print(f"\nwrote summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
