"""Measure each out-of-index genome's distance from the reference database with
minimap2.
"""

#=======================================================================
# Imports
#=======================================================================

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# Requires the minimap2 binary on PATH (tested with 2.30).

#=======================================================================
# Constants
#=======================================================================

# Genomes are measured against the masked viral FASTA from the pipeline's reference index
# (version 20250825), the same sequences the aligner searches. The index is not public, so the
# path must be given explicitly; `fetch_database` takes a local file or an s3:// URI. The
# output, `panels/out_of_index_divergence.tsv`, is committed, so rebuilding is optional.

# See the module docstring on what asm20 targets. The short-read presets would be wrong
# regardless: these are genome-to-genome comparisons.
PRESET = "asm20"

# Index batch size. Smaller batches keep peak memory down on a laptop at the cost of extra
# passes; minimap2 emits hits per batch and the best is taken across all of them.
BATCH_SIZE = "1G"

#=======================================================================
# Types
#=======================================================================


@dataclass
class BestHit:
    """The closest database genome found for one panel genome."""

    reference_id: str
    identity: float
    matching_bases: int
    aligned_span: int
    query_length: int

    @property
    def query_coverage(self) -> float:
        """Fraction of the panel genome the alignment spans."""
        return self.aligned_span / self.query_length if self.query_length else 0.0


#=======================================================================
# Alignment
#=======================================================================


def fetch_database(source: str, destination: Path) -> Path:
    """Download the database FASTA if it lives on S3; minimap2 reads gzip directly."""
    if not source.startswith("s3://"):
        return Path(source)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {source} ...")
        subprocess.run(
            ["aws", "s3", "cp", source, str(destination), "--no-progress"], check=True,
        )
    return destination


def align(query_fasta: Path, database: Path, preset: str, batch_size: str) -> list[str]:
    """Run minimap2 with the panel genomes as queries, returning PAF lines."""
    command = [
        "minimap2", "-x", preset, "-c", "-I", batch_size, "--secondary=no",
        str(database), str(query_fasta),
    ]
    print(f"aligning: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"minimap2 failed:\n{result.stderr[-2000:]}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def best_hits(paf_lines: list[str]) -> dict[str, BestHit]:
    """Reduce PAF to each query's best hit, ranked by matching bases.

    PAF columns used: 1 query, 2 query length, 3-4 query start/end, 6 target,
    10 matching bases, 11 alignment block length.
    """
    best: dict[str, BestHit] = {}
    for line in paf_lines:
        fields = line.split("\t")
        query, query_length = fields[0], int(fields[1])
        span = int(fields[3]) - int(fields[2])
        target, matches, block = fields[5], int(fields[9]), int(fields[10])
        if block <= 0:
            continue
        current = best.get(query)
        if current is None or matches > current.matching_bases:
            best[query] = BestHit(
                reference_id=target,
                identity=matches / block,
                matching_bases=matches,
                aligned_span=span,
                query_length=query_length,
            )
    return best


def read_query_lengths(query_fasta: Path) -> dict[str, int]:
    """Every query's id and length, so genomes with no alignment are still reported."""
    lengths: dict[str, int] = {}
    name: str | None = None
    total = 0
    for raw in query_fasta.read_text().splitlines():
        line = raw.strip()
        if line.startswith(">"):
            if name is not None:
                lengths[name] = total
            name, total = line[1:].split()[0], 0
        elif line:
            total += len(line)
    if name is not None:
        lengths[name] = total
    return lengths


def measure(query_fasta: Path, database: Path, preset: str, batch_size: str) -> pl.DataFrame:
    """Find each panel genome's closest relative in the reference database."""
    lengths = read_query_lengths(query_fasta)
    hits = best_hits(align(query_fasta, database, preset, batch_size))
    return pl.DataFrame(
        [
            {
                "genome_id": genome_id,
                "length": length,
                "nearest_reference": hits[genome_id].reference_id
                if genome_id in hits
                else None,
                "identity": round(hits[genome_id].identity, 5)
                if genome_id in hits
                else None,
                "query_coverage": round(hits[genome_id].query_coverage, 5)
                if genome_id in hits
                else None,
                "matching_bases": hits[genome_id].matching_bases
                if genome_id in hits
                else 0,
            }
            for genome_id, length in sorted(lengths.items())
        ],
    ).sort("identity", descending=True, nulls_last=True)


#=======================================================================
# CLI
#=======================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="measure_divergence",
        description="Measure panel genomes' distance from the reference database.",
    )
    parser.add_argument("--query-fasta", type=Path, required=True)
    # No default: the reference index is not public, so the database has to be named
    # explicitly. Either a local path or an s3:// URI.
    parser.add_argument("--index-fasta", required=True)
    parser.add_argument(
        "--database-cache", type=Path, default=Path("refs/virus-genomes-masked.fasta.gz"),
    )
    parser.add_argument("--preset", default=PRESET)
    parser.add_argument("--batch-size", default=BATCH_SIZE)
    parser.add_argument(
        "--out", type=Path, default=Path("panels/out_of_index_divergence.tsv"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    database = fetch_database(args.index_fasta, args.database_cache)
    table = measure(args.query_fasta, database, args.preset, args.batch_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(args.out, separator="\t")

    hit = table.filter(pl.col("identity").is_not_null())
    print(f"\nwrote {table.height} genomes to {args.out}")
    print(f"  with an alignment to the database: {hit.height}/{table.height}")
    if hit.height:
        print("\nidentity to nearest database genome:")
        print(hit["identity"].describe())
        pl.Config.set_tbl_rows(12)
        print("\nmost divergent:")
        print(
            hit.sort("identity").head(12).select(
                "genome_id", "length", "nearest_reference", "identity", "query_coverage",
            ),
        )


if __name__ == "__main__":
    main()
