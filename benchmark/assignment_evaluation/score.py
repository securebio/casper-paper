"""Score a pipeline run against the panels, per panel and per genome."""

#=======================================================================
# Imports
#=======================================================================

from collections.abc import Mapping, Sequence

import polars as pl

from assignment_evaluation.simulate import source_accession
from assignment_evaluation.taxonomy import (
    Taxonomy,
    ancestor_at_rank,
)

#=======================================================================
# Attribution
#=======================================================================


def attribute_hits(hits: pl.DataFrame, manifest: pl.DataFrame) -> pl.DataFrame:
    """Attach each hit's source genome and that genome's ground truth.

    A hit whose accession is not in the manifest means the read name did not survive the
    pipeline as expected, which would invalidate every rate computed downstream, so it is an
    error rather than a dropped row.
    """
    attributed = hits.with_columns(
        pl.col("seq_id")
        .map_elements(source_accession, return_dtype=pl.String)
        .alias("genome_id"),
    )
    truth = manifest.select(
        "genome_id",
        "panel",
        pl.col("taxid").cast(pl.Int64).alias("true_taxid"),
        pl.col("species_taxid").cast(pl.Int64).alias("true_species_taxid"),
        "name",
    )
    joined = attributed.join(truth, on="genome_id", how="left")
    unknown = joined.filter(pl.col("panel").is_null())
    if unknown.height:
        examples = unknown["seq_id"].head(3).to_list()
        raise ValueError(
            f"{unknown.height} hits name a genome absent from the manifest, e.g. {examples}",
        )
    return joined


#=======================================================================
# Assignment scoring
#=======================================================================

_TRIPLE = ("true_taxid", "true_species_taxid", "assigned_taxid")


#=======================================================================
# Empirical divergence
#=======================================================================


#=======================================================================
# What was actually run
#=======================================================================


def restrict_to_run(
    allocation: pl.DataFrame,
    read_counts: pl.DataFrame,
    excluded: Mapping[str, int] | None = None,
) -> pl.DataFrame:
    """Drop allocated genomes whose panel was not part of this pipeline run.

    The allocation says what a complete study would simulate; ``read_counts.tsv.gz`` says what
    the pipeline actually received. A panel present in the first and absent from the second was
    never tested, and scoring it anyway turns missing data into a result -- a negative panel
    that was never generated would otherwise be reported as zero false positives over a hundred
    million pairs, which is the most misleading number this code could produce.

    Panels that did run are also checked against their allocation, since a partially simulated
    panel would silently deflate every rate computed from it.

    ``excluded`` names pairs that were generated and fed to the pipeline but deliberately left
    out of the allocation -- reads from genomes since dropped from the panel, say, when an
    older read set is reused. Passing it *strengthens* the check rather than relaxing it: the
    assertion becomes ``received == allocated + explicitly excluded``, so an unexplained
    discrepancy still raises. Omitting it leaves the check exactly as it was.
    """
    ran = set(read_counts["sample"])
    missing = sorted(set(allocation["panel"]) - ran)
    if missing:
        print(f"  not in this run, so not scored: {', '.join(missing)}")

    observed = dict(
        zip(read_counts["sample"], read_counts["n_read_pairs"].cast(pl.Int64), strict=True),
    )
    for panel in sorted(ran & set(allocation["panel"])):
        expected = int(
            allocation.filter(pl.col("panel") == panel)["n_pairs"].cast(pl.Int64).sum(),
        )
        dropped = (excluded or {}).get(panel, 0)
        if observed[panel] != expected + dropped:
            accounted = (
                f"{expected:,} pairs (plus {dropped:,} excluded)" if dropped else f"{expected:,} pairs"
            )
            raise ValueError(
                f"panel {panel!r} was allocated {accounted} but the pipeline received "
                f"{observed[panel]:,}; rates computed from it would be wrong",
            )
    return allocation.filter(pl.col("panel").is_in(sorted(ran)))


#=======================================================================
# Positive panels
#=======================================================================


def _rate(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """A rate that is null rather than an error when the denominator is zero."""
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)


#=======================================================================
# Rank-projected outcomes
#=======================================================================

# The ranks a call is tested against, finest first. Species is the unit the manuscript makes
# claims about; genus and family say how far short a call fell when it could not reach species.
RESOLUTION_RANKS = ("species", "genus", "family")


def _project_categories(
    attributed: pl.DataFrame, taxonomy: Taxonomy, ranks: Sequence[str],
) -> pl.DataFrame:
    """Label each hit with the finest rank its call gets right."""
    projections: dict[str, tuple[dict[int, int | None], dict[int, int | None]]] = {}
    assigned_ids = attributed["assigned_taxid"].drop_nulls().unique().to_list()
    true_ids = attributed["true_taxid"].drop_nulls().unique().to_list()
    for rank in ranks:
        projections[rank] = (
            {t: ancestor_at_rank(taxonomy, int(t), rank) for t in assigned_ids},
            {t: ancestor_at_rank(taxonomy, int(t), rank) for t in true_ids},
        )

    projected = attributed
    for rank in ranks:
        call_at, truth_at = projections[rank]
        projected = projected.with_columns(
            pl.col("assigned_taxid").replace_strict(call_at, default=None).alias(f"_call_{rank}"),
            pl.col("true_taxid").replace_strict(truth_at, default=None).alias(f"_true_{rank}"),
        )

    # Built coarsest-first so the finest test ends up outermost and wins. A rank the source
    # taxon does not have is skipped rather than failed, so an organism NCBI files without a
    # genus falls through to family instead of being counted wrong for NCBI's filing.
    category = pl.lit("incorrect")
    for rank in reversed(ranks):
        matches = (
            pl.col(f"_true_{rank}").is_not_null()
            & pl.col(f"_call_{rank}").is_not_null()
            & (pl.col(f"_call_{rank}") == pl.col(f"_true_{rank}"))
        )
        category = pl.when(matches).then(pl.lit(f"correct_{rank}")).otherwise(category)
    return projected.with_columns(category.alias("category"))


def _category_names(ranks: Sequence[str]) -> list[str]:
    return [f"correct_{rank}" for rank in ranks] + ["incorrect"]


def summarise_resolution(
    attributed: pl.DataFrame,
    allocation: pl.DataFrame,
    taxonomy: Taxonomy,
    ranks: Sequence[str] = RESOLUTION_RANKS,
) -> pl.DataFrame:
    """Per-arm counts of how precisely each read was identified.

    Four disjoint outcomes, summing to the reads simulated for that arm: ``correct_species``,
    ``correct_genus``, ``correct_family``, and ``incorrect``. Each read is filed at the finest
    rank the call gets right, so naming Coxsackievirus A for a Coxsackievirus B read is
    ``correct_genus``.

    ``incorrect`` covers every other outcome, including reads absent from the workflow's
    output entirely.
    """
    projected = _project_categories(attributed, taxonomy, ranks)
    names = _category_names(ranks)
    rows: list[dict[str, object]] = []
    for panel in sorted(set(allocation.filter(pl.col("panel") != "negative")["panel"])):
        allocated = int(
            allocation.filter(pl.col("panel") == panel)["n_pairs"].cast(pl.Int64).sum(),
        )
        arm = projected.filter(pl.col("panel") == panel)
        counts = dict(arm.group_by("category").len().iter_rows())
        # Reads the pipeline never emitted cannot appear in `arm`; they are the shortfall
        # against what was generated, and count as incorrect.
        counts["incorrect"] = counts.get("incorrect", 0) + (allocated - arm.height)
        rows += [
            {"panel": panel, "category": name, "n_reads": int(counts.get(name, 0)),
             "n_allocated": allocated}
            for name in names
        ]
    return pl.DataFrame(rows)


#=======================================================================
# Negative panel
#=======================================================================

# A Wilson interval is unnecessary for the headline; the rule of three is the standard
# 95% bound when no event is observed, and reduces to 3/n.
_RULE_OF_THREE = 3.0

# Reads in a billion, the scale the manuscript reports vertebrate-virus abundance at.
READS_PER_BILLION = 1_000_000_000


def summarise_resolution_by_family(
    attributed: pl.DataFrame,
    allocation: pl.DataFrame,
    manifest: pl.DataFrame,
    families: Mapping[int, str],
    taxonomy: Taxonomy,
    ranks: Sequence[str] = RESOLUTION_RANKS,
) -> pl.DataFrame:
    """The same four outcomes as ``summarise_resolution``, per virus family and arm.

    ``families`` maps a species taxid to its family, since the two arms label a genome with
    different taxids but the same species. A family absent from one arm simply has no row
    for it.
    """
    names = _category_names(ranks)
    species_to_family = dict(families)
    genome_family = {
        row["genome_id"]: species_to_family.get(int(row["species_taxid"]))
        for row in manifest.iter_rows(named=True)
    }
    keyed = pl.col("genome_id").replace_strict(genome_family, default=None).alias("family")
    key = ["family", "panel"]

    counted = manifest.filter(pl.col("panel") != "negative").with_columns(keyed).group_by(
        key,
    ).agg(
        pl.col("species_taxid").n_unique().alias("n_taxa"),
        pl.col("genome").n_unique().alias("n_genomes"),
    )
    allocated = allocation.filter(pl.col("panel") != "negative").with_columns(
        keyed,
    ).group_by(key).agg(pl.col("n_pairs").cast(pl.Int64).sum().alias("n_allocated"))

    projected = _project_categories(attributed, taxonomy, ranks).filter(
        pl.col("panel") != "negative",
    ).with_columns(keyed)
    observed = projected.group_by([*key, "category"]).len().pivot(
        on="category", index=key, values="len",
    )

    table = counted.join(allocated, on=key, how="left").join(observed, on=key, how="left")
    for name in names:
        if name not in table.columns:
            table = table.with_columns(pl.lit(0, dtype=pl.UInt32).alias(name))
    table = table.with_columns(pl.col(names).fill_null(0).cast(pl.Int64))
    # Reads the pipeline never emitted are the shortfall against what was generated.
    seen = sum(pl.col(name) for name in names)
    return table.with_columns(
        (pl.col("incorrect") + pl.col("n_allocated") - seen).alias("incorrect"),
    ).select("family", "panel", "n_taxa", "n_genomes", "n_allocated", *names).sort(
        ["family", "panel"],
    )


def summarise_negative(
    attributed: pl.DataFrame,
    allocation: pl.DataFrame,
    manifest: pl.DataFrame,
) -> pl.DataFrame:
    """Per-taxon false-positive rates for the negative panel, with abundance weights.

    A negative read is a false positive when the pipeline assigns it a vertebrate-virus taxon.
    Reads that merely survive into the hits file without an assignment are counted separately
    (``n_hits``): they passed the k-mer and alignment screens but the LCA gave them no taxon,
    which is a different and less severe outcome than a confident wrong call.

    ``expected_per_billion`` converts each genome's rate into what it would contribute to a
    billion real wastewater read pairs, using the organism's measured abundance. Where no
    false positive was seen the rate is zero, so the bound -- not the estimate -- is what
    carries information, and it is reported alongside.
    """
    negatives = allocation.filter(pl.col("panel") == "negative").select(
        "genome_id", "name", pl.col("n_pairs").cast(pl.Int64),
    )
    genomes = manifest.filter(pl.col("panel") == "negative").select(
        "genome_id", "genome", pl.col("read_fraction").cast(pl.Float64),
    )
    hits = attributed.filter(pl.col("panel") == "negative")

    counted = hits.group_by("genome_id").agg(
        pl.len().alias("n_hits"),
        pl.col("assigned_taxid").is_not_null().sum().alias("n_false_positive"),
    )
    per_accession = (
        negatives.join(genomes, on="genome_id", how="left")
        .join(counted, on="genome_id", how="left")
        .with_columns(pl.col("n_hits", "n_false_positive").fill_null(0))
    )
    # Pool a taxon's accessions before any rate is computed. A genome stored as several
    # records -- human as 25 chromosomes -- is one organism with one measured abundance, and
    # rating each record separately both applies that abundance once per record and shrinks
    # every denominator. The rule-of-three bound then diverges: the mitochondrion receives 3
    # of human's 500,000 pairs, giving a per-record bound of 3/3 = 1.0, which weighted by
    # human's abundance alone implies millions of spurious pairs per billion.
    pooled = per_accession.group_by("genome").agg(
        pl.col("name").first(),
        pl.col("read_fraction").first(),
        pl.col("n_pairs").sum(),
        pl.col("n_hits").sum(),
        pl.col("n_false_positive").sum(),
    )
    pairs, fp = pl.col("n_pairs"), pl.col("n_false_positive")
    return pooled.with_columns(
        _rate(fp, pairs).alias("false_positive_rate"),
        # With zero observed, the 95% upper bound on the rate is 3/n.
        pl.when(fp == 0)
        .then(_RULE_OF_THREE / pairs)
        .otherwise(_rate(fp, pairs))
        .alias("rate_upper_bound"),
    ).with_columns(
        (pl.col("false_positive_rate") * pl.col("read_fraction") * READS_PER_BILLION)
        .alias("expected_per_billion"),
        (pl.col("rate_upper_bound") * pl.col("read_fraction") * READS_PER_BILLION)
        .alias("bound_per_billion"),
    ).sort("read_fraction", descending=True)


def false_positive_taxa(attributed: pl.DataFrame, taxonomy: Taxonomy) -> pl.DataFrame:
    """Which vertebrate-virus taxa the negative panel's false positives were assigned to.

    Says not just how often a non-viral
    read is miscalled, but what it is miscalled *as*.
    """
    false_positives = attributed.filter(
        (pl.col("panel") == "negative") & pl.col("assigned_taxid").is_not_null(),
    )
    if false_positives.height == 0:
        return pl.DataFrame(
            schema={
                "assigned_taxid": pl.Int64(),
                "assigned_name": pl.String(),
                "n_reads": pl.UInt32(),
                "n_source_genomes": pl.UInt32(),
            },
        )
    counted = false_positives.group_by("assigned_taxid").agg(
        pl.len().alias("n_reads"),
        pl.col("genome_id").n_unique().alias("n_source_genomes"),
    )
    names = pl.DataFrame(
        {
            "assigned_taxid": counted["assigned_taxid"].to_list(),
            "assigned_name": [
                taxonomy.nodes[taxid].name if taxid in taxonomy.nodes else "unknown"
                for taxid in counted["assigned_taxid"].to_list()
            ],
        },
    )
    return counted.join(names, on="assigned_taxid", how="left").sort(
        "n_reads", descending=True,
    )
