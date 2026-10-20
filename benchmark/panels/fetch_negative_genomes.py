"""Fetch one RefSeq reference genome per negative-panel taxon."""

#=======================================================================
# Imports
#=======================================================================

import argparse
import gzip
import io
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import polars as pl
from ncbi import parse_fasta

#=======================================================================
# Constants
#=======================================================================

SUMMARY_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/{division}/assembly_summary.txt"

# Divisions the panel's taxa live in. Human is handled separately (see below), so the large
# vertebrate_mammalian summary is never downloaded.
DIVISIONS = ("bacteria", "archaea", "viral")

# NCBI's own quality gates.
REQUIRED_FIELDS = {
    "version_status": "latest",
    "genome_rep": "Full",
}
CLEAN_EXCLUSION = ("", "na")

# Assembly levels accepted, best first. Scaffold and Contig are excluded: only an assembly's
# an assembly's longest record, which for a fragmented assembly is an arbitrary contig rather
# than a genome. Chromosome level is accepted because 11 panel taxa have nothing better, and a
# finished chromosome counts as a genome here.
ACCEPTED_LEVELS = ("Complete Genome", "Chromosome")

# Any `excluded_from_refseq` flag disqualifies an assembly, and "derived from metagenome"
# matters most here. A MAG can carry chimeric or contaminating sequence, and viral sequence
# wrongly assembled into a bacterial bin would be scored as a pipeline false positive when it
# is really a defect of the reference -- exactly the confusion a specificity panel must avoid.
CATEGORY_RANK = {"reference genome": 0, "representative genome": 1}

# NCBI's FTP host resets long-running connections; retry rather than lose the run.
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_BACKOFF_SECONDS = 3.0

HUMAN_TAXID = "9606"

# The full GRCh38 primary assembly: autosomes 1-22, X, Y and the mitochondrion. Human gets the
# same fixed budget as any other taxon, so sampling the whole assembly only decides which human
# sequence those reads come from. Sequence liable to be miscalled as viral is spread unevenly
# across chromosomes, so sparse genome-wide sampling estimates the rate better than deep
# sampling of a few chromosomes.
HUMAN_ASSEMBLY = "GCF_000001405.40"
HUMAN_FASTA_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/"
    "GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
)
# Primary assembly only. The full file also carries unplaced scaffolds, patches and alternate
# haplotypes, which are redundant with the chromosomes and would treble the record count.
HUMAN_CHROMOSOMES = tuple(
    f"NC_{number:06d}.{version}"
    for number, version in (
        (1, 11), (2, 12), (3, 12), (4, 12), (5, 10), (6, 12), (7, 14), (8, 11),
        (9, 12), (10, 11), (11, 10), (12, 12), (13, 11), (14, 9), (15, 10), (16, 10),
        (17, 11), (18, 10), (19, 10), (20, 11), (21, 9), (22, 11), (23, 11), (24, 10),
    )
) + ("NC_012920.1",)

#=======================================================================
# Assembly summaries
#=======================================================================


def load_summary(division: str, cache_dir: Path) -> pl.DataFrame:
    """Read (downloading and caching if needed) a RefSeq division's assembly summary."""
    path = cache_dir / f"asm_refseq_{division}.txt"
    if not path.exists():
        url = SUMMARY_URL.format(division=division)
        print(f"  downloading {division} assembly summary ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=600) as response:  # noqa: S310
            path.write_bytes(response.read())
    return pl.read_csv(
        path, separator="\t", skip_rows=1, infer_schema_length=0, quote_char=None,
    ).rename({"#assembly_accession": "assembly_accession"})


def passes_quality(row: dict[str, str]) -> bool:
    """Whether an assembly clears NCBI's standard quality gates."""
    if any(row.get(field) != value for field, value in REQUIRED_FIELDS.items()):
        return False
    if str(row.get("assembly_level", "")) not in ACCEPTED_LEVELS:
        return False
    return str(row.get("excluded_from_refseq", "")).strip().lower() in CLEAN_EXCLUSION


def best_assembly(candidates: list[dict[str, str]]) -> dict[str, str] | None:
    """The canonical assembly among candidates: reference, then representative, then longest."""
    usable = [row for row in candidates if passes_quality(row)]
    if not usable:
        return None
    return min(
        usable,
        key=lambda row: (
            ACCEPTED_LEVELS.index(str(row["assembly_level"])),
            CATEGORY_RANK.get(str(row.get("refseq_category", "")), 2),
            row.get("assembly_accession", ""),
        ),
    )


#=======================================================================
# Sequence download
#=======================================================================


def genome_fasta_url(ftp_path: str) -> str:
    """The ``*_genomic.fna.gz`` URL for an assembly's FTP directory."""
    base = ftp_path.replace("ftp://", "https://").rstrip("/")
    return f"{base}/{base.split('/')[-1]}_genomic.fna.gz"


def download_to(url: str, cache_path: Path) -> None:
    """Fetch a large FASTA to disk, retrying the connection resets NCBI's host is prone to."""
    last_error: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            partial = cache_path.with_name(cache_path.name + ".part")
            with urllib.request.urlopen(url, timeout=1800) as response, partial.open("wb") as out:  # noqa: S310
                shutil.copyfileobj(response, out, length=1 << 20)
        except (OSError, urllib.error.URLError) as error:  # noqa: PERF203
            last_error = error
            time.sleep(DOWNLOAD_BACKOFF_SECONDS * (attempt + 1))
        else:
            partial.rename(cache_path)
            return
    raise RuntimeError(f"could not download {url}: {last_error}")


def copy_selected(path: Path, wanted: set[str], out: io.TextIOBase) -> dict[str, int]:
    """Copy named records from a gzipped multi-FASTA into ``out``, returning their lengths.

    Streams line by line, so memory stays at one FASTA line however large the records are.
    """
    lengths: dict[str, int] = {}
    current: str | None = None
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                accession = line[1:].split()[0]
                current = accession if accession in wanted else None
                if current is not None:
                    lengths[current] = 0
                    out.write(f">{current}\n")
            elif current is not None:
                out.write(line)
                lengths[current] += len(line.rstrip("\n"))
    return lengths


def longest_record(url: str, cache_path: Path) -> tuple[str, str]:
    """An assembly's longest record as ``(id, sequence)``, downloading only if not cached.

    Each genome is cached on its own because the panel is ~800 Mb over 200 downloads from
    NCBI's FTP host, which resets connections often enough that a single-shot run does not
    finish; caching makes a rerun resume rather than restart. Retries cover the resets
    themselves -- note ``ConnectionResetError`` is a plain ``OSError`` and *not* a
    ``URLError``, so catching the latter alone lets it through.
    """
    if cache_path.exists():
        with gzip.open(cache_path, "rt") as handle:
            records = parse_fasta(handle.read())
        if records:
            record_id = next(iter(records))
            return record_id, records[record_id]

    last_error: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=600) as response:  # noqa: S310
                raw = response.read()
            with gzip.open(io.BytesIO(raw), "rt") as handle:
                records = parse_fasta(handle.read())
            if not records:
                raise ValueError(f"no records in {url}")
            record_id = max(records, key=lambda key: len(records[key]))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(cache_path, "wt") as handle:
                handle.write(f">{record_id}\n{records[record_id]}\n")
            return record_id, records[record_id]
        except OSError as error:
            last_error = error
            time.sleep(DOWNLOAD_BACKOFF_SECONDS * (attempt + 1))
    raise OSError(f"{url} failed after {DOWNLOAD_ATTEMPTS} attempts: {last_error}")


#=======================================================================
# CLI
#=======================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_negative_genomes",
        description="Fetch a reference genome for each negative-panel taxon.",
    )
    parser.add_argument("--taxa", type=Path, default=Path("panels/kraken_abundance.tsv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("refs"))
    parser.add_argument(
        "--out-table", type=Path, default=Path("panels/negative_genomes.tsv"),
    )
    parser.add_argument(
        "--out-fasta", type=Path, default=Path("refs/negative_genomes.fasta.gz"),
    )
    parser.add_argument("--limit", type=int, help="Only process the first N taxa (testing).")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    taxa = pl.read_csv(args.taxa, separator="\t", infer_schema_length=0)
    if args.limit:
        taxa = taxa.head(args.limit)

    print("loading RefSeq assembly summaries ...")
    by_taxid: dict[str, list[dict[str, str]]] = {}
    wanted = set(taxa["species_taxid"])
    for division in DIVISIONS:
        summary = load_summary(division, args.cache_dir)
        for row in summary.filter(pl.col("species_taxid").is_in(list(wanted))).iter_rows(
            named=True,
        ):
            by_taxid.setdefault(str(row["species_taxid"]), []).append(row)

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
    partial_fasta = args.out_fasta.with_name(args.out_fasta.name + ".part")
    with gzip.open(partial_fasta, "wt", compresslevel=1) as out:
        for record in taxa.iter_rows(named=True):
            taxid, name = str(record["species_taxid"]), str(record["name"])
            if taxid == HUMAN_TAXID:
                # One bulk download rather than 25 efetch calls: the assembly FASTA is ~950 MB
                # gzipped and efetch is not built to stream chromosome-sized records.
                human_cache = args.cache_dir / "negative_genomes" / "GRCh38.fna.gz"
                if not human_cache.exists():
                    human_cache.parent.mkdir(parents=True, exist_ok=True)
                    print(f"  downloading GRCh38 ({HUMAN_ASSEMBLY}) ...")
                    download_to(HUMAN_FASTA_URL, human_cache)
                chromosomes = set(HUMAN_CHROMOSOMES)
                lengths = copy_selected(human_cache, chromosomes, out)
                missing_chromosomes = sorted(chromosomes - set(lengths))
                if missing_chromosomes:
                    raise SystemExit(
                        f"GRCh38 download is missing {len(missing_chromosomes)} expected "
                        f"chromosomes, e.g. {missing_chromosomes[:3]}",
                    )
                for accession in HUMAN_CHROMOSOMES:
                    rows.append(
                        {
                            "genome_id": accession, "species_taxid": taxid, "name": name,
                            "assembly_accession": HUMAN_ASSEMBLY,
                            "length": lengths[accession],
                            "read_fraction": record["read_fraction"],
                            "source": "human_chromosome",
                        },
                    )
                print(f"  {name[:44]:<44} {len(HUMAN_CHROMOSOMES)} GRCh38 chromosomes")
                continue

            assembly = best_assembly(by_taxid.get(taxid, []))
            if assembly is None:
                missing.append(f"{name} ({taxid})")
                continue
            cache_path = args.cache_dir / "negative_genomes" / f"{assembly['assembly_accession']}.fna.gz"
            try:
                record_id, sequence = longest_record(
                    genome_fasta_url(assembly["ftp_path"]), cache_path,
                )
            except (OSError, ValueError, KeyError) as error:
                missing.append(f"{name} ({taxid}): {error}")
                continue
            out.write(f">{record_id}\n{sequence}\n")
            rows.append(
                {
                    "genome_id": record_id, "species_taxid": taxid, "name": name,
                    "assembly_accession": assembly["assembly_accession"],
                    "length": len(sequence), "read_fraction": record["read_fraction"],
                    "source": f"refseq_{assembly.get('refseq_category') or 'na'}",
                },
            )
            print(f"  {name[:44]:<44} {record_id:<18} {len(sequence):>11,} bp")

        if missing:
            print(f"\n{len(missing)} taxa without a qualifying assembly:")
            for item in missing[:20]:
                print(f"   {item}")

    partial_fasta.rename(args.out_fasta)
    table = pl.DataFrame(rows)
    table.write_csv(args.out_table, separator="\t")
    total = int(table["length"].cast(pl.Int64).sum())
    # read_fraction is a property of the *taxon*, and one taxon can contribute several
    # genomes (human contributes 25 sequences), so coverage must be summed over distinct
    # taxa. Summing rows would count human's share twice.
    covered = float(
        table.unique(subset=["species_taxid"])["read_fraction"].cast(pl.Float64).sum(),
    )
    asked = float(
        taxa.unique(subset=["species_taxid"])["read_fraction"].cast(pl.Float64).sum(),
    )
    print(f"\nwrote {table.height} genomes ({total:,} bp) to {args.out_table}")
    print(
        f"abundance covered: {covered:.2%} of all profiled reads, "
        f"{covered / asked:.1%} of the taxa that were asked for",
    )


if __name__ == "__main__":
    main()
