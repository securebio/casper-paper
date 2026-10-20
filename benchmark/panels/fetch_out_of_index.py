"""Fetch genomes of the panel species that the reference index does not contain."""

#=======================================================================
# Imports
#=======================================================================

import argparse
import collections
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from ncbi import Eutils, parse_fasta, usable_window_fraction

#=======================================================================
# Constants
#=======================================================================

# Records whose titles carry any of these are not natural, finished genomes.
EXCLUDE_TITLE_PATTERN = re.compile(
    r"UNVERIFIED|patent|synthetic|vector|recombinant|chimeric|clone|"
    r"partial|mutant|engineered|expression|plasmid|"
    # "nearly complete genome" contains "complete genome", so the include pattern would
    # otherwise admit truncated records the length band is too loose to catch.
    r"nearly complete|near-complete|near complete|incomplete",
    re.IGNORECASE,
)

# Titles must look like a finished sequence. "complete cds" is accepted because that is how
# segmented viruses label a whole segment -- influenza A's records are "segment N ... gene,
# complete cds" and nothing else. Admitting it is safe only because the length band below is
# what actually enforces "genome-sized": a single gene out of a larger genome cannot pass it.
INCLUDE_TITLE_PATTERN = re.compile(
    r"complete genome|complete sequence|complete cds", re.IGNORECASE,
)

# Accepted length band, as a fraction of the species' index representative length.
LENGTH_TOLERANCE = 0.25

# Minimum share of read-length windows that must be free of ambiguous bases. Expressed as
# usable windows rather than as an ambiguity rate because that is what the generator consumes;
# see ncbi.usable_window_fraction.
MIN_USABLE_WINDOW_FRACTION = 0.95
READ_WINDOW = 150

# How many out-of-index genomes to take per species, and how many candidates to consider.
# One, matching the in-index arm: a second genome doubled a non-segmented species' reads while
# a segmented species still contributed a single isolate, so the arm weighted species by genome
# architecture rather than by anything the benchmark measures.
DEFAULT_PER_SPECIES = 1
SEARCH_RETMAX = 200

# Complete segmented genomes are sparse among recent depositions -- rotavirus A needs about
# 600 results before a full 11-segment isolate appears -- so segmented species search deeper.
SEGMENTED_RETMAX = 800

DEFAULT_SEED = 20260826

#=======================================================================
# Types
#=======================================================================


@dataclass(frozen=True)
class Candidate:
    """One GenBank nucleotide record considered for the out-of-index arm."""

    accession: str
    species_taxid: str
    taxid: str
    organism: str
    title: str
    length: int
    create_date: str


#=======================================================================
# Filtering
#=======================================================================


def strip_version(accession: str) -> str:
    """``NC_001474.2`` -> ``NC_001474``, so a re-versioned index genome is still recognised."""
    return accession.rsplit(".", 1)[0]


def title_is_acceptable(title: str) -> bool:
    """Whether a record title reads as a natural, finished genome."""
    return bool(INCLUDE_TITLE_PATTERN.search(title)) and not EXCLUDE_TITLE_PATTERN.search(
        title,
    )


def length_is_acceptable(length: int, reference_length: int) -> bool:
    """Whether a candidate's length sits within the tolerance band of the index representative.

    With no reference length the band has nothing to anchor to, so every candidate passes. That
    happens when the species is absent from the index entirely, and for a segmented genome,
    where a whole-genome length would reject every individual segment. No listed species is
    absent from the index. The guard rejects a candidate for want of a
    reference would silently drop the taxon rather than report it as uncovered.
    """
    if reference_length <= 0:
        return True
    low = reference_length * (1 - LENGTH_TOLERANCE)
    high = reference_length * (1 + LENGTH_TOLERANCE)
    return low <= length <= high


def summary_to_candidate(summary: dict[str, Any], species_taxid: str) -> Candidate | None:
    """Convert an esummary document to a candidate, or ``None`` if it is unusable."""
    accession = str(summary.get("accessionversion") or "")
    length = int(summary.get("slen") or 0)
    if not accession or length <= 0:
        return None
    return Candidate(
        accession=accession,
        species_taxid=species_taxid,
        taxid=str(summary.get("taxid") or ""),
        organism=str(summary.get("organism") or ""),
        title=str(summary.get("title") or ""),
        length=length,
        create_date=str(summary.get("createdate") or ""),
    )


def filter_candidates(
    candidates: list[Candidate],
    index_accessions: set[str],
    reference_length: int,
) -> list[Candidate]:
    """Apply the metadata-only filters (everything but the ambiguity check)."""
    return [
        candidate
        for candidate in candidates
        if strip_version(candidate.accession) not in index_accessions
        and title_is_acceptable(candidate.title)
        and length_is_acceptable(candidate.length, reference_length)
    ]

#=======================================================================
# Per-species selection
#=======================================================================


def search_term(
    species_taxid: str,
    reference_length: int,
    released_after: str | None = None,
) -> str:
    """The nuccore query for one species: its whole subtree, finished genomes, right length.

    ``[Organism:exp]`` explodes the taxon, so strain- and isolate-level records under the
    species are included -- which is where most non-index genomes live.

    The length band is applied *in the query* rather than to the results. Filtering afterwards
    spends the whole result budget on whatever was uploaded most recently, which for a species
    with millions of records is not a sample of the species: influenza A returns 200 consecutive
    2.2 kb PB1 segments, none of which can match a 1.4 kb segment-6 reference. Asking NCBI for
    the right size makes every returned record a real candidate.
    """
    term = (
        f"txid{species_taxid}[Organism:exp] AND "
        f'("complete genome"[Title] OR "complete sequence"[Title] OR "complete cds"[Title])'
    )
    # No reference length means no band to ask NCBI for; take any complete record instead.
    if reference_length > 0:
        low = int(reference_length * (1 - LENGTH_TOLERANCE))
        high = int(reference_length * (1 + LENGTH_TOLERANCE)) + 1
        term += f" AND {low}:{high}[SLEN]"
    if released_after is not None:
        term += f" AND {released_after}:3000[PDAT]"
    return term


def isolate_key(title: str) -> str:
    """The part of a record title that identifies the isolate rather than the segment.

    Segments of one genome share a title prefix and differ only after it: "Influenza A virus
    (A/Minnesota/ISC-1070/2026(H3N2)) **segment 4 haemagglutinin (HA) gene**". Cutting at the
    first segment, gene or RNA designation leaves a key shared by exactly the sequences that
    belong together. NCBI's ``organism`` field cannot do this job -- it returns a bare
    "Influenza A virus" for every record -- and esummary exposes no BioSample to group on.
    """
    head = re.split(r",|\bsegment\b|\bRNA\b\s*\d|\bgene\b", title, maxsplit=1)[0]
    return head.strip().rstrip(" ,;")


def choose_segmented(
    eligible: list[Candidate], segments: int, seed: int, species_taxid: str,
) -> tuple[list[Candidate], str]:
    """One isolate's complete segment set, or nothing.

    A segmented species must contribute a whole genome, as the in-index arm does: every
    segmented assembly in the index carries exactly as many sequences as it declares, so a
    partial set here would mean the two arms describe different molecules and the comparison
    for that species would not be like for like. Taking unrelated segments from different
    isolates would be worse still. A species with no complete isolate among its candidates
    therefore yields nothing and is reported as uncovered, which is the honest description: the
    panel has no out-of-index genome for it, rather than a fragment standing in for one.
    """
    groups: dict[str, list[Candidate]] = collections.defaultdict(list)
    for candidate in eligible:
        groups[isolate_key(candidate.title)].append(candidate)
    complete = sorted(
        (key for key, members in groups.items() if len(members) >= segments),
    )
    if complete:
        rng = random.Random(f"{seed}:{species_taxid}:isolate")
        chosen = groups[rng.choice(complete)]
        return sorted(chosen, key=lambda c: c.accession)[:segments], "ok"
    best = max(groups.values(), key=len, default=[])
    if not best:
        return [], "no isolate groups"
    return [], f"no complete isolate (best {len(best)}/{segments} segments)"


def choose_for_species(
    client: Eutils,
    species_taxid: str,
    reference_length: int,
    index_accessions: set[str],
    per_species: int,
    seed: int,
    released_after: str | None = None,
    segments: int = 1,
) -> tuple[list[Candidate], str]:
    """Pick out-of-index genomes for one species, returning them and a status string.

    Candidates surviving the metadata filters are shuffled with a species-specific seeded RNG
    and then checked for ambiguity one batch at a time, so a genome full of Ns costs a fetch
    but not the species. Sampling rather than taking the top hit keeps the choice out of the
    hands; seeding it keeps the panel reproducible.
    """
    # A segmented species has no single reference length: its segments span a wide range, and
    # a band drawn around any one of them would reject the rest. Completeness of the isolate
    # is the constraint that replaces it.
    if segments > 1:
        reference_length = 0
    retmax = SEGMENTED_RETMAX if segments > 1 else SEARCH_RETMAX
    uids = client.esearch(
        search_term(species_taxid, reference_length, released_after), retmax=retmax,
    )
    if not uids:
        return [], "no search hits"
    summaries = client.esummary(uids)
    candidates = [
        candidate
        for summary in summaries
        if (candidate := summary_to_candidate(summary, species_taxid)) is not None
    ]
    eligible = filter_candidates(candidates, index_accessions, reference_length)
    if not eligible:
        return [], f"0/{len(candidates)} candidates passed metadata filters"

    if segments > 1:
        return choose_segmented(eligible, segments, seed, species_taxid)

    rng = random.Random(f"{seed}:{species_taxid}")
    rng.shuffle(eligible)

    chosen: list[Candidate] = []
    seen_dates: set[str] = set()
    for start in range(0, len(eligible), per_species * 4):
        batch = eligible[start : start + per_species * 4]
        sequences = parse_fasta(client.efetch_fasta([c.accession for c in batch]))
        for candidate in batch:
            sequence = sequences.get(candidate.accession)
            if sequence is None:
                continue
            if usable_window_fraction(sequence, READ_WINDOW) < MIN_USABLE_WINDOW_FRACTION:
                continue
            # Consecutive accessions from one submission are often the same isolate sequenced
            # repeatedly, which would make a species' picks near-duplicates of each other.
            # The deposit date is a coarse but dependency-free proxy for the submission.
            if candidate.create_date in seen_dates:
                continue
            seen_dates.add(candidate.create_date)
            chosen.append(candidate)
            if len(chosen) == per_species:
                return chosen, "ok"
    status = "ok" if chosen else f"all {len(eligible)} candidates too ragged"
    return chosen, status


#=======================================================================
# CLI
#=======================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_out_of_index",
        description="Fetch high-quality virus genomes absent from the pipeline's index.",
    )
    parser.add_argument("--species", type=Path, default=Path("panels/positive_species.tsv"))
    parser.add_argument(
        "--genome-metadata",
        type=Path,
        default=Path("refs/virus-genome-metadata-gid.tsv.gz"),
        help="Index metadata, used to reject anything already indexed.",
    )
    parser.add_argument(
        "--out-table", type=Path, default=Path("panels/out_of_index_genomes.tsv"),
    )
    parser.add_argument("--out-fasta", type=Path, default=Path("refs/out_of_index.fasta"))
    parser.add_argument("--per-species", type=int, default=DEFAULT_PER_SPECIES)
    parser.add_argument(
        "--in-index", type=Path, default=Path("panels/in_index_genomes.tsv"),
        help="In-index arm, read only for each species' segment count.",
    )
    parser.add_argument(
        "--released-after",
        help=(
            "Only consider records deposited after this date (YYYY/MM/DD). Defaults to the "
            "day after the index's newest genome, so a hit cannot predate the index. "
            "Pass 'none' to select purely on absence from the index."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--email",
        default=os.environ.get("NCBI_EMAIL"),
        help="Contact address sent to NCBI (defaults to $NCBI_EMAIL). Strongly encouraged.",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N species (testing).")
    return parser


def _resolve_cutoff(requested: str | None, index_metadata: pl.DataFrame) -> str | None:
    """The deposit-date floor for candidates, defaulting to just after the index's newest genome.

    Index 20250825's newest ``seq_rel_date`` is 2025-03-10, so anything deposited after that
    could not have been in the build. The accession anti-join already guarantees absence from
    the index; the date is the criterion a reader can check without the index metadata, and the two
    together make the claim hard to argue with.
    """
    if requested is not None:
        return None if requested.lower() == "none" else requested
    newest = index_metadata["seq_rel_date"].max()
    if not isinstance(newest, str):
        return None
    year, month, day = (int(part) for part in newest.split("-"))
    return f"{year:04d}/{month:02d}/{day + 1:02d}"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.email:
        print("warning: no --email/$NCBI_EMAIL; NCBI asks that requests identify a contact")

    species = pl.read_csv(args.species, separator="\t", infer_schema_length=0)
    if args.limit:
        species = species.head(args.limit)
    index_metadata = pl.read_csv(
        args.genome_metadata, separator="\t", infer_schema_length=0,
    )
    index_accessions = {strip_version(g) for g in index_metadata["genome_id"]}

    released_after = _resolve_cutoff(args.released_after, index_metadata)
    if released_after:
        print(f"restricting to records deposited after {released_after}")

    client = Eutils(email=args.email, api_key=os.environ.get("NCBI_API_KEY"))

    # One batched call gets every index representative's length, which anchors each species'
    # accepted length band.
    # Species absent from the index have no representative, so there is nothing to look up.
    representatives = [
        accession for accession in species["representative_genome_id"].to_list()
        if accession and str(accession).strip() not in {"", "na", "None"}
    ]
    print(f"fetching lengths for {len(representatives)} index representatives ...")
    lengths = {
        str(summary.get("accessionversion")): int(summary.get("slen") or 0)
        for summary in client.esummary(representatives)
    }

    # How many sequences a complete genome of each species has, taken from the in-index arm
    # so both arms agree on what "complete" means.
    segments_for: dict[str, int] = {}
    if args.in_index.exists():
        in_index = pl.read_csv(args.in_index, separator="\t", infer_schema_length=0)
        segments_for = {
            str(row["species_taxid"]): int(row["n_sequences"])
            for row in in_index.iter_rows(named=True)
        }

    rows: list[dict[str, object]] = []
    for record in species.iter_rows(named=True):
        species_taxid = str(record["species_taxid"])
        reference_length = lengths.get(str(record["representative_genome_id"]), 0)
        chosen, status = choose_for_species(
            client,
            species_taxid,
            reference_length,
            index_accessions,
            args.per_species,
            args.seed,
            released_after,
            segments_for.get(species_taxid, 1),
        )
        label = record.get("species_name") or species_taxid
        print(f"  {label[:44]:<44} ref={reference_length:>7}  got {len(chosen)}  [{status}]")
        for candidate in chosen:
            rows.append(
                {
                    "genome_id": candidate.accession,
                    "species_taxid": species_taxid,
                    "taxid": candidate.taxid,
                    "organism_name": candidate.organism,
                    "title": candidate.title,
                    "length": candidate.length,
                    "create_date": candidate.create_date,
                    "reference_length": reference_length,
                    "species_name": record.get("species_name"),
                    "reference_density": record.get("reference_density"),
                    # The selector's own verdict for this species: "ok", or the shortfall when
                    # no isolate carried a full segment set. Read from the species row before,
                    # which has no such column, so every row was silently null and an
                    # incomplete segment set was indistinguishable from a complete one.
                    "selection_reason": status,
                    "n_segments_expected": segments_for.get(species_taxid, 1),
                },
            )

    empty = [str(r["species_taxid"]) for r in species.iter_rows(named=True)
             if not any(str(row["species_taxid"]) == str(r["species_taxid"]) for row in rows)]
    if empty:
        print(
            f"\n{len(empty)} species yielded no out-of-index genome: {', '.join(empty)}\n"
            "  (no qualifying record has been deposited for them since the index was built)",
        )
    if not rows:
        raise SystemExit("no out-of-index genomes were selected")

    # Re-fetch the winners together so the FASTA is written in one pass and in table order.
    accessions = [str(row["genome_id"]) for row in rows]
    print(f"fetching {len(accessions)} selected sequences ...")
    fetched = parse_fasta(client.efetch_fasta(accessions))
    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
    with args.out_fasta.open("w") as handle:
        for accession in accessions:
            handle.write(f">{accession}\n{fetched[accession]}\n")

    table = pl.DataFrame(rows)
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(args.out_table, separator="\t")
    print(f"wrote {table.height} genomes to {args.out_table} and {args.out_fasta}")


if __name__ == "__main__":
    main()
