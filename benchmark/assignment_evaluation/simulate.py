"""Generate simulated reads with InSilicoSeq: 2x151 bp paired-end, NovaSeq
error model, fragment length 250 +/- 50 bp.
"""

#=======================================================================
# Imports
#=======================================================================

import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

#=======================================================================
# Constants
#=======================================================================

# InSilicoSeq's NovaSeq model emits 151 bp reads; CASPER is 2x150.
READ_LENGTH = 151

# Trailing "_<read index>_<cpu chunk>" appended by InSilicoSeq to the record id.
_READ_SUFFIX = re.compile(r"_\d+_\d+$")

#=======================================================================
# Types
#=======================================================================


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters passed to InSilicoSeq."""

    executable: str = "iss"
    model: str = "novaseq"
    # Drawn from CASPER's own fragment-length distribution (median 171 over 319,000 real VV
    # alignments), shifted up so that essentially no draw falls below the 151 bp read length.
    # 250 sits near the 65th percentile, so mates overlap by ~50 bp as in the real data.
    fragment_length: int = 250
    fragment_length_sd: int = 50
    seed: int = 20260826
    cpus: int = 4
    compress: bool = True

    def __post_init__(self) -> None:
        if self.fragment_length < READ_LENGTH:
            raise ValueError(
                f"fragment_length {self.fragment_length} is below the read length "
                f"{READ_LENGTH}; InSilicoSeq would place the reverse read at a negative "
                f"offset and emit chimeric reads",
            )
        if self.fragment_length_sd < 0:
            raise ValueError("fragment_length_sd must be non-negative")
        if self.cpus <= 0:
            raise ValueError("cpus must be positive")


#=======================================================================
# Sequence preparation
#=======================================================================

# Bases a sequencer can emit. Reference genomes may carry any IUPAC ambiguity code, but an
# instrument reports only these, so reads containing R/Y/K/M/S/W/B/D/H/V are not something
# the pipeline would ever see.
_SEQUENCER_BASES = frozenset("ACGTN")


def normalise_bases(sequence: str) -> str:
    """Upper-case a sequence and replace every non-ACGTN base with N.

    Simulating straight from a GenBank record puts IUPAC ambiguity codes into the reads, and
    BBMerge -- which the profiling branch runs before Kraken2 -- asserts on the first one it
    sees ("A non-ACGTN base was detected"), taking down the whole run. The codes are rare
    rare enough to be a rounding error against the panels' billions of bases, but one is
    sufficient.

    Replacing them with N is also the more faithful choice: an Illumina instrument emits only
    ACGT and N, so an ambiguity code in a read is an artifact of simulating from a curated
    reference rather than anything a sequencer produces.
    """
    return "".join(
        base if base in _SEQUENCER_BASES else "N" for base in sequence.upper()
    )


#=======================================================================
# Ground truth
#=======================================================================


def source_accession(seq_id: str) -> str:
    """Recover the source genome accession from an InSilicoSeq read name.

    ``NZ_CP032147.1_13_1`` -> ``NZ_CP032147.1``. Only the two trailing numeric fields are
    stripped, so accessions containing underscores and dots survive intact. Any ``/1`` or
    ``/2`` mate suffix is removed first, since aligners drop it from the SAM QNAME.
    """
    without_mate = seq_id.rsplit("/", 1)[0] if seq_id.endswith(("/1", "/2")) else seq_id
    stripped = _READ_SUFFIX.sub("", without_mate)
    if stripped == without_mate:
        raise ValueError(f"{seq_id!r} is not an InSilicoSeq read name")
    return stripped


#=======================================================================
# Running InSilicoSeq
#=======================================================================


def resolve_executable(name: str) -> str:
    """Absolute path to the InSilicoSeq entry point.

    Resolved rather than shelled out to by name: the tool is installed into the project's
    virtualenv, which is not on ``PATH`` for a subprocess started from a script run with the
    venv's interpreter.
    """
    found = shutil.which(name)
    if found is None:
        # The venv's bin directory sits alongside the running interpreter.
        candidate = Path(sys.executable).parent / name
        if candidate.exists():
            return str(candidate)
        raise RuntimeError(
            f"{name!r} not found on PATH or beside {sys.executable}; "
            f"install InSilicoSeq (pip install InSilicoSeq)",
        )
    return found


def write_readcount_file(pairs_per_genome: Mapping[str, int], path: Path) -> None:
    """Write InSilicoSeq's ``--readcount_file``, converting read *pairs* into reads."""
    if not pairs_per_genome:
        raise ValueError("no genomes to simulate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for genome_id, pairs in sorted(pairs_per_genome.items()):
            if pairs <= 0:
                raise ValueError(f"{genome_id} was allocated {pairs} pairs")
            handle.write(f"{genome_id}\t{pairs * 2}\n")


def build_command(
    genomes_fasta: Path,
    readcount_file: Path,
    out_prefix: Path,
    config: SimulationConfig,
) -> list[str]:
    """The ``iss generate`` command line for one panel."""
    # Deliberately never ``--compress``: InSilicoSeq gzips single-threaded, measured at
    # 0.57 MB/s on a 32-core box, which for a 31 GB mate is ~3 hours each. ``compress_pair``
    # does the same job with pigz at ~47 MB/s. See ``simulate``.
    return [
        resolve_executable(config.executable), "generate",
        "--genomes", str(genomes_fasta),
        "--readcount_file", str(readcount_file),
        "--model", config.model,
        "--fragment-length", str(config.fragment_length),
        "--fragment-length-sd", str(config.fragment_length_sd),
        "--seed", str(config.seed),
        "--cpus", str(config.cpus),
        "--output", str(out_prefix),
    ]


def simulate(
    genomes_fasta: Path,
    pairs_per_genome: Mapping[str, int],
    out_prefix: Path,
    config: SimulationConfig | None = None,
) -> tuple[Path, Path]:
    """Run InSilicoSeq for one panel, returning the R1 and R2 paths."""
    settings = config or SimulationConfig()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    readcount_file = out_prefix.with_name(f"{out_prefix.name}_readcounts.tsv")
    write_readcount_file(pairs_per_genome, readcount_file)

    result = subprocess.run(
        build_command(genomes_fasta, readcount_file, out_prefix, settings),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"iss generate failed:\n{result.stderr.strip()}")

    r1 = out_prefix.with_name(f"{out_prefix.name}_R1.fastq")
    r2 = out_prefix.with_name(f"{out_prefix.name}_R2.fastq")
    for path in (r1, r2):
        if not path.exists():
            raise RuntimeError(f"iss generate did not write {path}")
    if not settings.compress:
        return r1, r2
    return compress_pair(r1, r2, settings.cpus)


def compress_pair(r1: Path, r2: Path, cpus: int) -> tuple[Path, Path]:
    """Gzip both mates, in parallel where pigz is available.

    InSilicoSeq's own ``--compress`` is single-threaded and, on a 31 GB mate, takes hours
    while 29 of 30 cores idle. pigz produces an ordinary gzip stream that anything reading
    ``.fastq.gz`` accepts, so this changes only how long it takes. gzip is the fallback for
    machines without pigz, where the panels are small enough not to matter.
    """
    compressor = shutil.which("pigz")
    command = [compressor, "-p", str(cpus)] if compressor else [shutil.which("gzip") or "gzip"]
    result = subprocess.run([*command, str(r1), str(r2)], check=False, capture_output=True,
                            text=True)
    if result.returncode != 0:
        raise RuntimeError(f"compressing reads failed:\n{result.stderr.strip()}")
    out = (r1.with_suffix(".fastq.gz"), r2.with_suffix(".fastq.gz"))
    for path in out:
        if not path.exists():
            raise RuntimeError(f"compression did not write {path}")
    return out
