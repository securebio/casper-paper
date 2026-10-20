"""Simulate the reads for each panel and write mgs-workflow samplesheets."""

#=======================================================================
# Imports
#=======================================================================

import argparse
import csv
import gzip
from dataclasses import asdict
from pathlib import Path

import polars as pl

from assignment_evaluation.provenance import write_metadata
from assignment_evaluation.simulate import (
    SimulationConfig,
    normalise_bases,
    simulate,
)

#=======================================================================
# Constants
#=======================================================================

# Each panel's gzipped sequences, named relative to the run directory's genomes/.
PANEL_FASTA = {
    "in_index": Path("in_index.fasta.gz"),
    "out_of_index": Path("out_of_index.fasta.gz"),
    "negative": Path("negative_genomes.fasta.gz"),
}

#=======================================================================
# Helpers
#=======================================================================


def resolve_fasta(path: Path, scratch: Path) -> Path:
    """A plain-text, sequencer-alphabet FASTA path, prepared in ``scratch``.

    Two transformations, both needed before InSilicoSeq sees the file. It reads input through
    Biopython, which does not transparently decompress, so a gzipped panel is expanded. And
    every base outside ACGTN is replaced with N: reference genomes carry IUPAC ambiguity codes
    that no sequencer emits, and BBMerge in the profiling branch aborts the run on the first
    one it encounters.

    The published genomes are left as fetched, so what is stored stays faithful to GenBank and
    only the simulator's input is normalised.
    """
    prepared = scratch / (path.name.removesuffix(".gz") + ".prepared.fasta")
    if prepared.exists():
        return prepared
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"  preparing {path.name} (decompress, normalise to ACGTN) ...")
    opener = gzip.open if path.suffix == ".gz" else open
    replaced = 0
    with opener(path, "rt") as source, prepared.open("w") as target:  # type: ignore[operator]
        for line in source:
            if line.startswith(">"):
                target.write(line)
                continue
            sequence = line.strip()
            cleaned = normalise_bases(sequence)
            replaced += sum(
                1 for a, b in zip(sequence.upper(), cleaned, strict=True) if a != b
            )
            target.write(cleaned + "\n")
    if replaced:
        print(f"    replaced {replaced:,} ambiguity codes with N")
    return prepared


def write_samplesheet(out_dir: Path, path: Path) -> None:
    """Write mgs-workflow's paired-end samplesheet from whatever panels are on disk.

    Built by scanning the output directory rather than from the panels simulated in this
    invocation: the negative panel takes hours and is naturally run on its own, and writing
    only the current run's rows would silently drop the panels generated earlier.
    """
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample", "fastq_1", "fastq_2"))
        for panel in PANEL_FASTA:
            r1 = out_dir / f"{panel}_R1.fastq.gz"
            r2 = out_dir / f"{panel}_R2.fastq.gz"
            if r1.exists() and r2.exists():
                writer.writerow((panel, str(r1.resolve()), str(r2.resolve())))


#=======================================================================
# CLI
#=======================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simulate_panels",
        description="Simulate reads for each panel with InSilicoSeq.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory holding genomes/ and panels/; reads are written to its reads/.",
    )
    parser.add_argument(
        "--panels",
        nargs="+",
        default=list(PANEL_FASTA),
        help="Which panels to simulate (default: all three).",
    )
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # The allocation travels with the run, so a re-simulation cannot silently pick up a
    # regenerated one from the repository.
    allocation = pl.read_csv(
        args.run_dir / "panels" / "read_allocation.tsv", separator="\t", infer_schema_length=0,
    )
    out_dir = args.run_dir / "reads"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig(cpus=args.cpus, seed=args.seed)

    for panel in args.panels:
        subset = allocation.filter(pl.col("panel") == panel)
        if subset.height == 0:
            raise SystemExit(f"no genomes allocated to panel {panel!r}")
        pairs = {
            row["genome_id"]: int(row["n_pairs"]) for row in subset.iter_rows(named=True)
        }
        total = sum(pairs.values())
        print(f"{panel}: {len(pairs)} genomes, {total:,} read pairs")

        fasta = resolve_fasta(args.run_dir / "genomes" / PANEL_FASTA[panel], out_dir / "fasta")
        r1, r2 = simulate(fasta, pairs, out_dir / panel, config)
        print(f"  wrote {r1.name}, {r2.name}")

    samplesheet = out_dir / "samplesheet.csv"
    write_samplesheet(out_dir, samplesheet)
    print(f"\nsamplesheet: {samplesheet}")
    print(samplesheet.read_text().rstrip())

    write_metadata(
        {
            "reads": {
                panel: int(
                    allocation.filter(pl.col("panel") == panel)["n_pairs"]
                    .cast(pl.Int64)
                    .sum(),
                )
                for panel in args.panels
            },
            "simulation": asdict(config),
        },
        args.run_dir / "run_metadata.json",
    )


if __name__ == "__main__":
    main()
