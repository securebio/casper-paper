"""Combine the three panels into one accession-to-organism ground-truth table."""

#=======================================================================
# Imports
#=======================================================================

import argparse
from pathlib import Path

import polars as pl

#=======================================================================
# Assembly
#=======================================================================


def fasta_lengths(path: Path) -> dict[str, int]:
    """Sequence length per accession, so the manifest can carry it.

    The in-index arm's lengths are not in the index metadata, which records assemblies rather
    than sequences, so they are read from the FASTA the arm is simulated from -- the same file,
    so the two cannot drift.
    """
    lengths: dict[str, int] = {}
    current = None
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                current = line[1:].split()[0]
                lengths[current] = 0
            elif current is not None:
                lengths[current] += len(line.strip())
    return lengths


def in_index_rows(
    panel: pl.DataFrame, index_metadata: pl.DataFrame, lengths: dict[str, int],
) -> pl.DataFrame:
    """In-index genomes, taking the strain-level taxid from the index's own metadata."""
    taxids = index_metadata.select(
        "genome_id", "taxid", pl.col("species_taxid").alias("index_species_taxid"),
    ).unique(subset=["genome_id"])
    missing = sorted(set(panel["genome_id"]) - set(lengths))
    if missing:
        raise SystemExit(
            f"{len(missing)} in-index genomes are absent from the sequence FASTA, so their "
            f"read allocation cannot be sized: {missing[:5]}",
        )
    return (
        panel.with_columns(
            pl.col("genome_id").replace_strict(lengths).cast(pl.Int64).alias("length"),
        )
        .join(taxids, on="genome_id", how="left")
        .select(
            "genome_id",
            pl.lit("in_index").alias("panel"),
            "taxid",
            # The index's species wins over the panel file's, which comes from the RefSeq
            # assembly summary. The two occasionally disagree -- LC381845.1 is "TTV-like mini
            # virus" to the index and Torque teno virus to RefSeq -- and the index's answer is
            # the one the pipeline's LCA reads, so scoring against RefSeq's would manufacture
            # a species-level error out of a correct call.
            pl.coalesce(pl.col("index_species_taxid"), pl.col("species_taxid"))
            .alias("species_taxid"),
            # The species name is the grouping key the evaluator averages within, so it must
            # be populated on every row: a null would collapse every unlabelled genome into
            # one group and silently corrupt the species-averaged rates.
            pl.col("species_name").alias("name"),
            pl.lit(value=True).alias("expect_vv"),
            pl.lit(None, dtype=pl.Float64).alias("read_fraction"),
            "length",
            pl.col("assembly_accession").alias("genome"),
        )
    )


def out_of_index_rows(panel: pl.DataFrame) -> pl.DataFrame:
    """Out-of-index genomes; taxid comes from the GenBank record."""
    return panel.select(
        "genome_id",
        pl.lit("out_of_index").alias("panel"),
        "taxid",
        "species_taxid",
        pl.col("species_name").alias("name"),
        pl.lit(value=True).alias("expect_vv"),
        pl.lit(None, dtype=pl.Float64).alias("read_fraction"),
        pl.col("length").cast(pl.Int64).alias("length"),
        # One isolate per segmented species, so the species taxid names that genome; for a
        # non-segmented species each of the two records is its own genome.
        pl.when(pl.col("n_segments_expected").cast(pl.Int64) > 1)
        .then(pl.col("species_taxid"))
        .otherwise(pl.col("genome_id"))
        .alias("genome"),
    )


def negative_rows(panel: pl.DataFrame) -> pl.DataFrame:
    """Negative genomes; the taxon is the species, and no VV assignment is correct."""
    return panel.select(
        "genome_id",
        pl.lit("negative").alias("panel"),
        pl.col("species_taxid").alias("taxid"),
        "species_taxid",
        "name",
        pl.lit(value=False).alias("expect_vv"),
        pl.col("read_fraction").cast(pl.Float64),
        pl.col("length").cast(pl.Int64).alias("length"),
        pl.col("assembly_accession").alias("genome"),
    )


#=======================================================================
# CLI
#=======================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_panel_manifest",
        description="Combine the three panels into one ground-truth table.",
    )
    parser.add_argument("--in-index", type=Path, default=Path("panels/in_index_genomes.tsv"))
    parser.add_argument(
        "--out-of-index", type=Path, default=Path("panels/out_of_index_genomes.tsv"),
    )
    parser.add_argument("--negative", type=Path, default=Path("panels/negative_genomes.tsv"))
    parser.add_argument(
        "--in-index-fasta", type=Path, default=Path("refs/in_index.fasta"),
        help="Sequences the in-index arm is simulated from, for its lengths.",
    )
    parser.add_argument(
        "--genome-metadata",
        type=Path,
        default=Path("refs/virus-genome-metadata-gid.tsv.gz"),
    )
    parser.add_argument("--out", type=Path, default=Path("panels/panel_manifest.tsv"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    read = lambda path: pl.read_csv(path, separator="\t", infer_schema_length=0)  # noqa: E731

    manifest = pl.concat(
        [
            in_index_rows(
                read(args.in_index), read(args.genome_metadata),
                fasta_lengths(args.in_index_fasta),
            ),
            out_of_index_rows(read(args.out_of_index)),
            negative_rows(read(args.negative)),
        ],
        how="vertical",
    )

    # The read name is the only link back to this table, so an accession appearing in two
    # panels would make a read's origin ambiguous and silently corrupt every rate.
    duplicated = manifest.filter(pl.col("genome_id").is_duplicated())
    if duplicated.height:
        raise SystemExit(
            f"genome_id appears in more than one panel: "
            f"{sorted(set(duplicated['genome_id']))[:5]}",
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(args.out, separator="\t")
    print(f"wrote {manifest.height} genomes to {args.out}")
    print(manifest.group_by("panel").agg(pl.len().alias("n_genomes")).sort("panel"))


if __name__ == "__main__":
    main()
