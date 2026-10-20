"""Define the positive vertebrate-infecting virus set as the union of published
priority lists.
"""

#=======================================================================
# Imports
#=======================================================================

import argparse
import csv
from collections import defaultdict
from pathlib import Path

#=======================================================================
# The list
#=======================================================================

# (query, sources). The query is matched against every NCBI name class.
POSITIVE_TAXA: tuple[tuple[str, str], ...] = (
    # --- wastewater surveillance targets ---
    ("Severe acute respiratory syndrome coronavirus 2", "nwss,scan,nndss,who,vpd,casper"),
    ("Influenza A virus", "nwss,scan,nndss,who,vpd,casper"),
    ("Influenza B virus", "scan,vpd,casper"),
    ("Influenza C virus", "casper"),
    # Influenza A is one taxon; its subtypes partition the same accessions. NWSS interest in
    # A(H5) is carried by the nwss tag on the influenza A row.
    # One RSV row: the index carries no RefSeq-linked assembly under either subtype taxid, so
    # per-subtype rows would both inherit the subgroup A reference genome.
    ("Human respiratory syncytial virus", "nwss,scan,vpd,casper"),
    ("Measles morbillivirus", "nwss,nndss,vpd"),
    ("Monkeypox virus", "nwss,nndss,who"),
    ("Human metapneumovirus", "scan,casper"),
    ("Human respirovirus 1", "scan,casper"),
    ("Human orthorubulavirus 2", "scan,casper"),
    ("Human respirovirus 3", "scan,casper"),
    ("Human orthorubulavirus 4", "scan,casper"),
    ("Enterovirus D68", "scan,casper"),
    ("Enterovirus C", "gpei,nndss,vpd,casper"),
    # --- hepatitis ---
    ("Hepatitis A virus", "nndss,vpd,casper"),
    ("Hepatitis B virus", "nndss,vpd,casper"),
    ("Hepacivirus hominis", "nndss,casper"),
    ("Hepatitis delta virus", "nndss"),
    # --- other vaccine-preventable ---
    ("Mumps orthorubulavirus", "nndss,vpd"),
    ("Rubella virus", "nndss,vpd"),
    ("Human alphaherpesvirus 3", "nndss,vpd,casper"),
    ("Rotavirus A", "vpd,casper"),
    ("Alphapapillomavirus 9", "vpd"),
    ("Yellow fever virus", "nndss,vpd,who"),
    ("Japanese encephalitis virus", "nndss,vpd"),
    ("Lyssavirus rabies", "nndss,vpd"),
    ("Tick-borne encephalitis virus", "vpd"),
    # --- arboviral, notifiable ---
    ("Dengue virus", "nndss,who,vpd"),
    ("Zika virus", "nndss,who"),
    ("Chikungunya virus", "nndss,who"),
    ("Eastern equine encephalitis virus", "nndss"),
    ("Western equine encephalitis virus", "nndss"),
    ("St. Louis encephalitis virus", "nndss"),
    ("Powassan virus", "nndss"),
    ("La Crosse virus", "nndss"),
    ("Jamestown Canyon virus", "nndss"),
    ("Oropouche orthobunyavirus", "who"),
    # --- viral haemorrhagic fever and WHO priority ---
    ("Zaire ebolavirus", "nndss,who"),
    ("Sudan ebolavirus", "nndss,who"),
    ("Marburg marburgvirus", "nndss,who"),
    ("Lassa mammarenavirus", "nndss,who"),
    ("Orthonairovirus haemorrhagiae", "nndss,who"),
    ("Henipavirus nipahense", "who"),
    ("Henipavirus hendraense", "who"),
    ("Phlebovirus riftense", "who"),
    ("Middle East respiratory syndrome-related coronavirus", "nndss,who"),
    ("Severe acute respiratory syndrome-related coronavirus", "nndss,who"),
    ("Mammarenavirus machupoense", "nndss,who"),
    ("Mammarenavirus juninense", "nndss,who"),
    ("Mammarenavirus guanaritoense", "nndss,who"),
    ("Mammarenavirus chapareense", "nndss,who"),
    ("Mammarenavirus lujoense", "nndss,who"),
    ("Variola virus", "nndss,who"),
    ("Orthohantavirus andesense", "nndss,who"),
    ("Orthohantavirus sinnombreense", "nndss,who"),
    ("Orthohantavirus hantanense", "who"),
    # --- HIV ---
    ("Human immunodeficiency virus 1", "nndss,casper"),
    ("Human immunodeficiency virus 2", "nndss,casper"),
    # --- enteric, reported and clinically panelled ---
    # One norovirus row. NCBI nests "Norovirus GII" beneath "Norwalk virus", so the single
    # entry covers both genogroups by lineage, and the release does not report them apart.
    ("Norwalk virus", "casper"),
    ("Sapporo virus", "casper"),
    ("Mamastrovirus hominis", "casper"),
    ("Mamastrovirus melbournense", "casper"),
    ("Human mastadenovirus F", "casper"),
    ("Human mastadenovirus A", "casper"),
    ("Human mastadenovirus B", "casper"),
    ("Human mastadenovirus C", "casper"),
    ("Human mastadenovirus D", "casper"),
    ("Primate bocaparvovirus 1", "casper"),
    ("Aichivirus A", "casper"),
    # No picobirnavirus entry: the reference database's own inclusion rule excludes
    # Picobirnaviridae, so the family carries no genomes to select from.
    ("Enterovirus A", "casper"),
    ("Enterovirus B", "casper"),
    # No Enterovirus D row: it is the parent of the enterovirus D68 entry above, and the
    # index holds one RefSeq-linked assembly for the species, so both rows would draw the
    # same genome in either arm.
    ("Rhinovirus A", "casper"),
    ("Rhinovirus B", "casper"),
    ("Rhinovirus C", "casper"),
    ("Rotavirus B", "casper"),
    ("Rotavirus C", "casper"),
    # --- respiratory coronaviruses ---
    ("Human coronavirus 229E", "casper"),
    ("Human coronavirus NL63", "casper"),
    ("Betacoronavirus 1", "casper"),
    ("Human coronavirus HKU1", "casper"),
    # --- herpesviruses and other persistent infections ---
    ("Human alphaherpesvirus 1", "casper"),
    ("Human alphaherpesvirus 2", "nndss,casper"),
    ("Human gammaherpesvirus 4", "casper"),
    ("Human betaherpesvirus 5", "casper"),
    ("Human betaherpesvirus 6A", "casper"),
    ("Human betaherpesvirus 6B", "casper"),
    ("Human betaherpesvirus 7", "casper"),
    ("Human gammaherpesvirus 8", "casper"),
    ("Betapolyomavirus hominis", "casper"),
    ("Erythroparvovirus primate1", "casper"),
    ("Molluscum contagiosum virus", "casper"),
    ("Pegivirus hominis", "casper"),
    ("Pegivirus columbiaense", "casper"),
)

#=======================================================================
# Resolution
#=======================================================================


def load_names(path: Path) -> tuple[dict[str, set[int]], dict[int, str]]:
    """Index every NCBI name class, and the scientific name per taxid."""
    lookup: dict[str, set[int]] = defaultdict(set)
    scientific: dict[int, str] = {}
    with path.open() as handle:
        for line in handle:
            parts = [field.strip() for field in line.split("|")]
            taxid, name, name_class = int(parts[0]), parts[1], parts[3]
            lookup[name.lower()].add(taxid)
            if name_class == "scientific name":
                scientific[taxid] = name
    return lookup, scientific


def load_nodes(path: Path) -> tuple[dict[int, str], dict[int, int]]:
    """Rank and parent per taxid."""
    ranks, parents = {}, {}
    with path.open() as handle:
        for line in handle:
            parts = [field.strip() for field in line.split("|")]
            taxid = int(parts[0])
            parents[taxid] = int(parts[1])
            ranks[taxid] = parts[2]
    return ranks, parents


def ancestor_at(
    taxid: int, rank: str, ranks: dict[int, str], parents: dict[int, int],
) -> int | None:
    """The ancestor at ``rank``, or the taxon itself if it already sits there."""
    seen = set()
    while taxid not in seen:
        seen.add(taxid)
        if ranks.get(taxid) == rank:
            return taxid
        parent = parents.get(taxid)
        if parent is None or parent == taxid:
            return None
        taxid = parent
    return None


def species_of(taxid: int, ranks: dict[int, str], parents: dict[int, int]) -> int | None:
    """The species-rank ancestor, or the taxon itself if it is one.

    Most of these names resolve to nodes NCBI places *below* species -- the ICTV binomial
    renaming left the familiar names ("SARS-CoV-2", "Rotavirus A") as unranked children of the
    new species nodes. The panel is built at species level, so both are recorded: the taxon the
    published list names, and the species it belongs to.
    """
    return ancestor_at(taxid, "species", ranks, parents)


def load_manifest(
    path: Path, listed: set[int], parents: dict[int, int],
) -> dict[int, dict[str, list[str]]]:
    """Accessions already selected, keyed by the most specific listed taxon each falls under.

    A genome is attributed to the *deepest* listed taxon in its lineage, not to every listed
    ancestor. NCBI nests some listed taxa inside others -- "Norovirus GII" sits beneath
    "Norwalk virus", and enterovirus D68 beneath enterovirus D -- so attaching to every
    ancestor put the GII genome on the GI row as well, and the panel appeared to hold two
    norovirus genomes per genotype when it holds one each.

    Keying on species instead would be worse still: four species in the panel carry two listed
    taxa, and every genome of those species would land on both rows.
    """
    by_taxon: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with path.open() as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            if record["panel"] == "negative":
                continue
            taxid, seen = int(record["taxid"]), set()
            while taxid not in seen:
                seen.add(taxid)
                if taxid in listed:
                    by_taxon[taxid][record["panel"]].append(record["genome_id"])
                    break
                parent = parents.get(taxid)
                if parent is None or parent == taxid:
                    break
                taxid = parent
    return by_taxon


def attach_above(
    taxid: int,
    species: int | None,
    manifest_rows: list[dict[str, str]],
    ranks: dict[int, str],
    parents: dict[int, int],
) -> dict[str, list[str]]:
    """Genomes sitting above ``taxid`` but inside the same species."""
    if species is None:
        return {}
    found: dict[str, list[str]] = defaultdict(list)
    for record in manifest_rows:
        genome_taxid = int(record["taxid"])
        if genome_taxid == taxid:
            continue
        if ancestor_at(taxid, "species", ranks, parents) != ancestor_at(
            genome_taxid, "species", ranks, parents,
        ):
            continue
        # Keep it only if the genome's taxon is an ancestor of the listed one.
        walk, seen = taxid, set()
        while walk not in seen:
            seen.add(walk)
            if walk == genome_taxid:
                found[record["panel"]].append(record["genome_id"])
                break
            parent = parents.get(walk)
            if parent is None or parent == walk:
                break
            walk = parent
    return found


def write_species(
    rows: list[dict[str, object]], index_species: Path, out: Path,
) -> None:
    """Emit the species list the selection steps consume, one row per distinct species.

    A left join, deliberately. The taxa list is defined against published priority lists and
    NCBI taxonomy, not against the reference database, so a species the database does not hold
    appears here with empty index metadata rather than vanishing during the join, and the gap
    is then visible in the emitted table. No listed taxon is in that position at present: the
    two that were -- West Nile virus, absent from index 20250825 altogether, and
    *Mamastrovirus virginiaense*, present but with no RefSeq-linked assembly -- are excluded
    from the list, because a taxon with no in-index genome cannot contribute the matched pair
    of arms the panel is built on.
    """
    catalogue: dict[str, dict[str, str]] = {}
    if index_species.exists():
        with index_species.open() as handle:
            for record in csv.DictReader(handle, delimiter="\t"):
                catalogue[record["species_taxid"]] = record

    seen, species_rows = set(), []
    for row in rows:
        taxid = str(row["species_taxid"])
        if not taxid or taxid in seen:
            continue
        seen.add(taxid)
        metadata = catalogue.get(taxid, {})
        species_rows.append({
            "species_taxid": taxid,
            "species_name": row["species_name"],
            "family_name": row["family_name"],
            "in_reference_index": "yes" if metadata else "no",
            **{
                field: metadata.get(field, "")
                for field in (
                    "n_index_genomes", "representative_genome_id", "representative_relation",
                    "earliest_seq_rel_date", "latest_seq_rel_date", "species_rank",
                    "taxid_genus", "taxid_family", "infection_status_human",
                    "infection_status_vertebrate", "reference_density",
                )
            },
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(species_rows[0]))
        writer.writeheader()
        writer.writerows(species_rows)
    absent = sum(1 for row in species_rows if row["in_reference_index"] == "no")
    print(f"  {len(species_rows)} species -> {out}  ({absent} absent from the index)")


def attach_within_species(
    taxid: int,
    species: int | None,
    manifest_rows: list[dict[str, str]],
    ranks: dict[int, str],
    parents: dict[int, int],
) -> dict[str, list[str]]:
    """Genomes of the same species that sit in a sibling branch, and that nothing else claimed.

    NCBI sometimes keeps a legacy name and a current one as siblings under one species rather
    than nesting them: *Orthoebolavirus zairense* holds both "Ebola virus" and "Zaire
    ebolavirus", and the index's reference genome descends from the first while the published
    priority lists name the second. Neither is an ancestor of the other, so a lineage test in
    either direction misses, and the row looks unselected when a perfectly good genome exists.

    Last resort, reached only when a taxon has no genome of its own and none above it: a
    broader listed taxon whose only genomes sit in a narrower one, as enterovirus D does when
    the sole RefSeq-linked assembly of the species is the enterovirus D68 one.
    """
    if species is None:
        return {}
    found: dict[str, list[str]] = defaultdict(list)
    for record in manifest_rows:
        if ancestor_at(int(record["taxid"]), "species", ranks, parents) == species:
            found[record["panel"]].append(record["genome_id"])
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="positive_taxa")
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("panels/positive_taxa.tsv"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("panels/panel_manifest.tsv"),
        help="Panel manifest to draw already-selected accessions from, if it exists.",
    )
    parser.add_argument(
        "--index-species", type=Path, default=Path("panels/index_species.tsv"),
        help="Species catalogue derived from the reference index, for metadata only.",
    )
    parser.add_argument(
        "--species-out", type=Path, default=Path("panels/positive_species.tsv"),
        help="Derived species list the two selection steps consume.",
    )
    args = parser.parse_args(argv)

    lookup, scientific = load_names(args.names)
    ranks, parents = load_nodes(args.nodes)
    manifest_rows: list[dict[str, str]] = []
    if args.manifest.exists():
        with args.manifest.open() as handle:
            manifest_rows = [
                record for record in csv.DictReader(handle, delimiter="\t")
                if record["panel"] != "negative"
            ]
    listed_taxids = {
        sorted(lookup[query.lower()])[0]
        for query, _sources in POSITIVE_TAXA if lookup.get(query.lower())
    }
    selected = (
        load_manifest(args.manifest, listed_taxids, parents)
        if args.manifest.exists() else {}
    )

    rows, unresolved, ambiguous = [], [], []
    for query, _sources in POSITIVE_TAXA:
        hits = lookup.get(query.lower(), set())
        if not hits:
            unresolved.append(query)
            continue
        if len(hits) > 1:
            ambiguous.append((query, sorted(hits)))
        taxid = sorted(hits)[0]
        species = species_of(taxid, ranks, parents)
        family = ancestor_at(taxid, "family", ranks, parents)
        # A cascade, each step used only where the one before it found nothing. A taxon with
        # its own genome must not also inherit a relative's: "Norovirus GII" sits beneath
        # "Norwalk virus" in NCBI's tree, so an unconditional ancestor rule handed the GI
        # genome to the GII row and doubled both.
        arms = {panel: list(ids) for panel, ids in selected.get(taxid, {}).items()}
        for fallback in (attach_above, attach_within_species):
            for panel in ("in_index", "out_of_index"):
                if arms.get(panel):
                    continue
                found = fallback(taxid, species, manifest_rows, ranks, parents).get(panel)
                if found:
                    arms[panel] = list(found)
        rows.append({
            "taxid": taxid,
            "ncbi_scientific_name": scientific.get(taxid, ""),
            "ncbi_rank": ranks.get(taxid, ""),
            "species_taxid": species or "",
            "species_name": scientific.get(species, "") if species else "",
            "family_name": scientific.get(family, "") if family else "",
            "in_index_genomes": ",".join(arms.get("in_index", ())),
            "out_of_index_genomes": ",".join(arms.get("out_of_index", ())),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t",
            fieldnames=["taxid", "ncbi_scientific_name", "ncbi_rank",
                        "species_taxid", "species_name", "family_name",
                        "in_index_genomes", "out_of_index_genomes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with_genomes = sum(
        1 for row in rows if row["in_index_genomes"] or row["out_of_index_genomes"]
    )
    write_species(rows, args.index_species, args.species_out)

    print(f"resolved {len(rows)} of {len(POSITIVE_TAXA)} entries -> {args.out}")
    print(f"  entries with at least one panel genome selected: {with_genomes}")
    print(f"  entries still needing genome selection:          {len(rows) - with_genomes}")
    if ambiguous:
        print(f"\nambiguous ({len(ambiguous)}), first taxid taken:")
        for query, taxids in ambiguous:
            print(f"   {query}: {taxids}")
    if unresolved:
        print(f"\nUNRESOLVED ({len(unresolved)}):")
        for query in unresolved:
            print(f"   {query}")


if __name__ == "__main__":
    main()
