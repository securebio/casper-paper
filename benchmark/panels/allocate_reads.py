"""Decide how many read pairs to simulate from each genome: 10,000 pairs for a
positive genome, 500,000 for a negative one, split across its accessions by length.
"""

#=======================================================================
# Imports
#=======================================================================

import argparse
from pathlib import Path

import polars as pl

#=======================================================================
# Constants
#=======================================================================

# Pairs per genome in the positive panels. At a detection rate near 99% this gives a standard
# error of roughly 0.1 percentage points, which is finer than any difference worth reporting.
DEFAULT_POSITIVE_PAIRS = 10_000

# Pairs per genome in the negative panel, fifty times the positive budget because the quantity
# is a rate near zero rather than one near 99%. Observing no false positive from a genome at
# this depth bounds its per-read rate at 3/500,000 = 6e-6 (rule of three), and the panel as a
# whole at a correspondingly tighter figure once the per-taxon rates are weighted by abundance.
DEFAULT_NEGATIVE_PAIRS = 500_000

PANEL_BUDGET_ARG = {
    "in_index": "positive_pairs",
    "out_of_index": "positive_pairs",
    "negative": "negative_pairs",
}

#=======================================================================
# Allocation
#=======================================================================


def allocate(
    manifest: pl.DataFrame,
    positive_pairs: int,
    negative_pairs: int,
) -> pl.DataFrame:
    """Assign a read-pair count to every genome in the panel manifest."""
    for column in ("length", "genome"):
        if column not in manifest.columns:
            raise ValueError(f"manifest rows need a {column!r} column to size allocations")
    budgets = {"positive_pairs": positive_pairs, "negative_pairs": negative_pairs}
    panels = set(manifest["panel"].unique())
    unknown = panels - set(PANEL_BUDGET_ARG)
    if unknown:
        raise ValueError(f"manifest carries unknown panel(s) {sorted(unknown)}")

    # Joined rather than mapped inline so the budget table is explicit in the output and a
    # panel with no budget fails loudly instead of silently allocating nothing.
    budget_rows = pl.DataFrame(
        {
            "panel": sorted(panels),
            "taxon_pairs": [budgets[PANEL_BUDGET_ARG[panel]] for panel in sorted(panels)],
        },
        schema={"panel": pl.Utf8, "taxon_pairs": pl.Int64},
    )
    lengths = pl.col("length").cast(pl.Int64)
    allocation = (
        manifest.with_columns(lengths.alias("length"))
        .join(budget_rows, on="panel", how="inner")
        # The genome is the unit that receives a budget; grouping on the panel as well keeps
        # the two positive arms independent, since a species appears in both.
        .with_columns(lengths.sum().over(["panel", "genome"]).alias("genome_length"))
        .with_columns(
            (pl.col("taxon_pairs") * pl.col("length") / pl.col("genome_length"))
            .round()
            .cast(pl.Int64)
            .clip(lower_bound=1)
            .alias("n_pairs"),
        )
    )
    return allocation.select("genome_id", "panel", "name", "n_pairs")


#=======================================================================
# CLI
#=======================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="allocate_reads",
        description="Assign read-pair counts to every genome in the study.",
    )
    parser.add_argument("--manifest", type=Path, default=Path("panels/panel_manifest.tsv"))
    parser.add_argument("--positive-pairs", type=int, default=DEFAULT_POSITIVE_PAIRS)
    parser.add_argument("--negative-pairs", type=int, default=DEFAULT_NEGATIVE_PAIRS)
    parser.add_argument("--out", type=Path, default=Path("panels/read_allocation.tsv"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = pl.read_csv(args.manifest, separator="\t", infer_schema_length=0)
    allocation = allocate(manifest, args.positive_pairs, args.negative_pairs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    allocation.write_csv(args.out, separator="\t")
    print(f"wrote {allocation.height} allocations to {args.out}")
    print(
        allocation.group_by("panel").agg(
            pl.len().alias("genomes"),
            pl.col("n_pairs").sum().alias("total_pairs"),
            pl.col("n_pairs").min().alias("min"),
            pl.col("n_pairs").median().alias("median"),
            pl.col("n_pairs").max().alias("max"),
        ).sort("panel"),
    )
    print(f"study total: {int(allocation['n_pairs'].sum()):,} read pairs")


if __name__ == "__main__":
    main()
