"""Fetch the in-index panel's unmasked sequences from GenBank."""

#=======================================================================
# Imports
#=======================================================================

import argparse
import os
from pathlib import Path

import polars as pl
from ncbi import Eutils, parse_fasta

#=======================================================================
# CLI
#=======================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_panel_sequences",
        description="Fetch the in-index arm's genome sequences from GenBank.",
    )
    parser.add_argument("--panel", type=Path, default=Path("panels/in_index_genomes.tsv"))
    parser.add_argument("--out-fasta", type=Path, default=Path("refs/in_index.fasta"))
    parser.add_argument("--email", default=os.environ.get("NCBI_EMAIL"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    panel = pl.read_csv(args.panel, separator="\t", infer_schema_length=0)
    accessions = panel["genome_id"].to_list()
    print(f"fetching {len(accessions)} in-index genomes from GenBank ...")

    client = Eutils(email=args.email, api_key=os.environ.get("NCBI_API_KEY"))
    records = parse_fasta(client.efetch_fasta(accessions))

    missing = [accession for accession in accessions if accession not in records]
    if missing:
        raise SystemExit(f"GenBank returned no sequence for: {missing[:5]}")

    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
    with args.out_fasta.open("w") as handle:
        for accession in accessions:
            handle.write(f">{accession}\n{records[accession]}\n")

    total = sum(len(records[a]) for a in accessions)
    ambiguous = sum(
        sum(1 for base in records[a].upper() if base not in "ACGT") for a in accessions
    )
    print(f"wrote {len(accessions)} genomes ({total:,} bp) to {args.out_fasta}")
    print(f"  ambiguous bases in the GenBank records: {ambiguous:,} ({ambiguous/total:.3%})")


if __name__ == "__main__":
    main()
