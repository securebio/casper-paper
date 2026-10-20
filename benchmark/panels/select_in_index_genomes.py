"""Choose the in-index genome for each positive-panel species: the species' RefSeq
reference assembly, resolved into the index.
"""

#=======================================================================
# Imports
#=======================================================================

import argparse
from pathlib import Path

import polars as pl

#=======================================================================
# Selection
#=======================================================================

UNLINKED = {"", "na", "NA", "None"}


def refseq_linked(index_metadata: pl.DataFrame) -> pl.DataFrame:
    """Index rows whose assembly is also published by RefSeq."""
    return index_metadata.filter(
        pl.col("gbrs_paired_asm").is_not_null()
        & ~pl.col("gbrs_paired_asm").is_in(list(UNLINKED)),
    )


def choose_assembly(candidates: pl.DataFrame, representative: str | None) -> str | None:
    """The assembly representing a species: the representative's, else the largest."""
    if representative:
        holding = candidates.filter(pl.col("genome_id") == representative)
        if holding.height:
            return str(holding["assembly_accession"][0])
    if not candidates.height:
        return None
    ranked = (
        candidates.group_by("assembly_accession")
        .agg(pl.len().alias("n_sequences"))
        .sort(["n_sequences", "assembly_accession"], descending=[True, False])
    )
    return str(ranked["assembly_accession"][0])


def descendant_ids(taxid: int, parents: dict[int, int], universe: set[int]) -> set[int]:
    """Every taxid in ``universe`` that sits at or below ``taxid``."""
    under = set()
    for candidate in universe:
        walk, seen = candidate, set()
        while walk not in seen:
            seen.add(walk)
            if walk == taxid:
                under.add(candidate)
                break
            parent = parents.get(walk)
            if parent is None or parent == walk:
                break
            walk = parent
    return under


def select(
    taxa: pl.DataFrame,
    species: pl.DataFrame,
    index_metadata: pl.DataFrame,
    parents: dict[int, int],
) -> tuple[pl.DataFrame, list[str]]:
    """One complete genome per listed taxon, plus the taxa that could not be represented."""
    linked = refseq_linked(index_metadata)
    representative = dict(
        zip(species["species_taxid"], species["representative_genome_id"], strict=True),
    )
    names = dict(zip(species["species_taxid"], species["species_name"], strict=True))
    rows: dict[str, dict[str, object]] = {}
    dropped: list[str] = []
    for record in taxa.iter_rows(named=True):
        species_taxid = str(record["species_taxid"])
        in_species = linked.filter(pl.col("species_taxid") == species_taxid)
        if not in_species.height:
            dropped.append(f"{record['ncbi_scientific_name']} ({species_taxid})")
            continue
        universe = {int(t) for t in in_species["taxid"] if str(t).isdigit()}
        under = descendant_ids(int(record["taxid"]), parents, universe)
        # Genomes of the taxon itself where they exist, otherwise the species stands in.
        candidates = (
            in_species.filter(pl.col("taxid").cast(pl.Int64).is_in(list(under)))
            if under else in_species
        )
        assembly = choose_assembly(candidates, representative.get(species_taxid))
        if assembly is None:
            dropped.append(f"{record['ncbi_scientific_name']} ({species_taxid})")
            continue
        chosen = candidates.filter(pl.col("assembly_accession") == assembly).sort("genome_id")
        for genome_id in chosen["genome_id"]:
            rows.setdefault(genome_id, {
                "genome_id": genome_id,
                "species_taxid": species_taxid,
                "species_name": names.get(species_taxid, ""),
                "assembly_accession": assembly,
                "n_sequences": chosen.height,
            })
    return pl.DataFrame(list(rows.values())), dropped


#=======================================================================
# CLI
#=======================================================================


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="select_in_index_genomes",
        description="Select one complete reference genome per panel species.",
    )
    parser.add_argument("--taxa", type=Path, default=Path("panels/positive_taxa.tsv"))
    parser.add_argument("--species", type=Path, default=Path("panels/positive_species.tsv"))
    parser.add_argument(
        "--taxonomy-nodes", type=Path, default=Path("refs/taxonomy-nodes.dmp"),
    )
    parser.add_argument(
        "--genome-metadata", type=Path,
        default=Path("refs/virus-genome-metadata-gid.tsv.gz"),
    )
    parser.add_argument("--out", type=Path, default=Path("panels/in_index_genomes.tsv"))
    args = parser.parse_args(argv)

    read = lambda path: pl.read_csv(path, separator="\t", infer_schema_length=0)  # noqa: E731
    parents: dict[int, int] = {}
    with args.taxonomy_nodes.open() as handle:
        for line in handle:
            fields = [field.strip() for field in line.split("|")]
            parents[int(fields[0])] = int(fields[1])
    panel, dropped = select(
        read(args.taxa), read(args.species), read(args.genome_metadata), parents,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.write_csv(args.out, separator="\t")

    if dropped:
        print(f"dropped {len(dropped)} taxa with no RefSeq-linked assembly in the index:")
        for name in dropped:
            print(f"   {name}")
    print(f"wrote {panel.height} genomes over {panel['species_taxid'].n_unique()} species")
    multi = (
        panel.group_by("species_name")
        .agg(pl.col("n_sequences").first())
        .filter(pl.col("n_sequences") > 1)
        .sort("n_sequences", descending=True)
    )
    print(f"  segmented genomes kept whole: {multi.height}")
    for record in multi.head(8).iter_rows(named=True):
        print(f"     {record['species_name'][:44]:<44} {record['n_sequences']:>2} sequences")


if __name__ == "__main__":
    main()
