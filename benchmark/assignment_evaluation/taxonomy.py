"""Load the NCBI taxonomy dumps and place one taxon relative to another."""

#=======================================================================
# Imports
#=======================================================================

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

#=======================================================================
# Types
#=======================================================================


@dataclass(frozen=True)
class TaxonNode:
    """One taxon: its id, parent id, NCBI rank word, and scientific name."""

    taxid: int
    parent: int
    rank: str
    name: str


# Sentinel for the up/down coordinates when no tree relationship is defined.


# The major ranks, root-ward to tip-ward. Anything else is either a sub-level of one of these
# (see _NCBI_SUBRANKS) or unranked.
_MAJOR_RANKS: frozenset[str] = frozenset(
    {
        "root",
        "superkingdom",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    },
)

# NCBI sub-ranks -> the major rank enclosing them, so a sub-level reports the resolution it
# actually achieved.
_NCBI_SUBRANKS: Mapping[str, str] = MappingProxyType(
    {
        "subspecies": "species",
        "serotype": "species",
        "serogroup": "species",
        "genotype": "species",
        "strain": "species",
        "isolate": "species",
        "biotype": "species",
        "species group": "genus",
        "species subgroup": "genus",
        "subgenus": "genus",
        "subfamily": "family",
        "suborder": "order",
        "subclass": "class",
        "subphylum": "phylum",
        "subkingdom": "kingdom",
        # Realms are the viral tree's sub-root clades, above every major rank tracked here.
        "realm": "clade",
        "subrealm": "clade",
        "clade": "clade",
    },
)

def ancestor_at_rank(taxonomy: "Taxonomy", taxid: int, rank: str) -> int | None:
    """The ancestor of ``taxid`` at ``rank``, itself if it is already there, else ``None``.

    ``None`` means the taxon sits *above* the rank -- a genus node has no species ancestor --
    which is a distinct outcome from naming the wrong one, and the two must not be pooled.

    This is what makes rates comparable between genomes. The tree-topology categories measure
    steps from the source genome's own taxid, and NCBI places viral taxids at inconsistent
    depth: across the out-of-index arm the true taxid sits 0 to 3 steps below its species node
    depending on the organism. A one-step call is therefore the species for one genome and the
    genus for another, so `parent` and `ancestor` are not one thing. Projecting both truth and
    call onto a named rank removes that, at the cost of the finer detail relate() preserves.
    """
    for ancestor in taxonomy.lineage(taxid):
        node = taxonomy.nodes.get(ancestor)
        if node is not None and node.rank == rank:
            return ancestor
    return None


@dataclass(frozen=True)
class Taxonomy:
    """An immutable taxonomy tree keyed by taxid."""

    nodes: Mapping[int, TaxonNode]

    def lineage(self, taxid: int) -> tuple[int, ...]:
        """Return the lineage from ``taxid`` (inclusive) up to the root."""
        result: list[int] = []
        seen: set[int] = set()
        current = taxid
        while True:
            node = self.nodes.get(current)
            if node is None:
                raise KeyError(f"unknown taxid {current}")
            result.append(current)
            if node.parent == current:  # root points to itself
                return tuple(result)
            if current in seen:
                raise ValueError(f"cycle in taxonomy at taxid {current}")
            seen.add(current)
            current = node.parent

    def is_ancestor_or_equal(self, ancestor: int, taxid: int) -> bool:
        """Return whether ``ancestor`` lies on the lineage of ``taxid`` (or equals it)."""
        return ancestor in self.lineage(taxid)

#=======================================================================
# Parsing
#=======================================================================


def parse_ncbi_taxonomy(nodes_lines: Iterable[str], names_lines: Iterable[str]) -> Taxonomy:
    """Parse NCBI ``nodes.dmp`` + ``names.dmp`` (``\\t|\\t``-separated) into a Taxonomy.

    This is the tree the viral LCA itself uses; ranks are words (e.g. ``species``). Only
    scientific names are kept.
    """
    names: dict[int, str] = {}
    for raw in names_lines:
        line = raw.rstrip("\n")
        if not line:
            continue
        fields = line.split("\t|\t")
        if len(fields) > 3 and fields[3].rstrip("\t|") == "scientific name":
            names[int(fields[0])] = fields[1]

    nodes: dict[int, TaxonNode] = {}
    for raw in nodes_lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        fields = line.split("\t|\t")
        if len(fields) < 3:
            raise ValueError(f"malformed nodes.dmp line: {line!r}")
        taxid = int(fields[0])
        nodes[taxid] = TaxonNode(
            taxid=taxid,
            parent=int(fields[1]),
            rank=fields[2],
            name=names.get(taxid, ""),
        )
    return Taxonomy(MappingProxyType(nodes))


def read_ncbi_taxonomy(nodes_path: Path, names_path: Path) -> Taxonomy:
    """Read NCBI ``nodes.dmp`` and ``names.dmp`` files into a Taxonomy.

    Both handles are streamed (the parser consumes names before nodes), so only the parsed
    name/node mappings are retained — never the raw text of either dump.
    """
    with nodes_path.open() as nodes_handle, names_path.open() as names_handle:
        return parse_ncbi_taxonomy(nodes_handle, names_handle)


#=======================================================================
# Classification and ranks
#=======================================================================



def major_rank(taxonomy: Taxonomy, taxid: int) -> str:
    """Return the human-readable major rank of a taxon.

    A sub-level reports the major rank that encloses it (``subspecies`` -> ``species``), and
    anything unrecognised -- including NCBI's very common ``no rank`` -- is ``"unknown"``.
    """
    node = taxonomy.nodes.get(taxid)
    if node is None:
        raise KeyError(f"unknown taxid {taxid}")
    if node.parent == taxid:
        return "root"  # the root is marked by self-parenthood; its own rank is "no rank"
    if node.rank in _MAJOR_RANKS:
        return node.rank
    return _NCBI_SUBRANKS.get(node.rank, "unknown")


#=======================================================================
# Ranks lost
#=======================================================================

# The ranks that count as a level of resolution. "root" is excluded: it is the tree's origin,
# not a rank anyone resolves to.
_RANKED: frozenset[str] = _MAJOR_RANKS - {"root"}


