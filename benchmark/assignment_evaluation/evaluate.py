"""Join mgs-workflow's viral hits back to the ground-truth manifest."""

#=======================================================================
# Imports
#=======================================================================

from collections.abc import Sequence
from pathlib import Path

import polars as pl

#=======================================================================
# Column selection
#=======================================================================

# Map hits column -> evaluation column. Parsed by name, so extra columns and column-order
# differences are ignored. Required columns must be present; optional ones are filled with
# nulls if absent, so a column the pipeline does not emit (e.g. the natural-only
# `aligner_n_assignments` is absent in outputs carrying only the `_combined`/`_artificial`
# variants).
_REQUIRED_COLUMNS: dict[str, str] = {
    "seq_id": "seq_id",
    "aligner_taxid_lca": "assigned_taxid",
}
_OPTIONAL_COLUMNS: dict[str, str] = {
    "aligner_length_normalized_score_mean": "assigned_score",
    # 3.0.1 publishes only the _combined and _artificial variants; the natural-only column
    # exists in later versions. Both are read so either schema yields the same frame.
    "aligner_n_assignments": "assigned_n_assignments",
    "aligner_n_assignments_combined": "assigned_n_assignments_combined",
    "prim_align_genome_id_all": "assigned_genome_id",
    "prim_align_ref_start": "assigned_ref_start",
    # Edit distance against the best database match, with the trimmed read length it applies
    # to. Together these give how closely a read matched what the pipeline aligned it to,
    # which is the empirical divergence of its source genome from the database.
    "prim_align_edit_distance": "edit_distance",
    "prim_align_edit_distance_rev": "edit_distance_rev",
    "query_len": "query_len",
    "query_len_rev": "query_len_rev",
}
_HITS_COLUMNS: dict[str, str] = {**_REQUIRED_COLUMNS, **_OPTIONAL_COLUMNS}

# On-disk type for each evaluation column (seq_id stays string, untyped here).
_ASSIGNED_TYPES: dict[str, pl.DataType] = {
    "assigned_taxid": pl.Int64(),
    "assigned_score": pl.Float64(),
    "assigned_n_assignments": pl.Int64(),
    "assigned_n_assignments_combined": pl.Int64(),
    "assigned_genome_id": pl.String(),
    "assigned_ref_start": pl.Int64(),
    "edit_distance": pl.Int64(),
    "edit_distance_rev": pl.Int64(),
    "query_len": pl.Int64(),
    "query_len_rev": pl.Int64(),
}

# Integer columns 3.0.1 writes in floating-point form ("2427.0"), which a direct cast to
# Int64 rejects. Observed in real output for the alignment start; taxids are written as plain
# integers. Routed through Float64 rather than special-cased at the call site so a version
# that writes them either way parses identically.
_FLOAT_FORMATTED_INTEGERS = frozenset(
    {
        "assigned_ref_start",
        "assigned_n_assignments_combined",
        "edit_distance",
        "edit_distance_rev",
        "query_len",
        "query_len_rev",
    },
)

# The pipeline's missing-value sentinel (e.g. for artificial-only or unaligned reads).
_NULL_VALUES = ["NA"]

#=======================================================================
# Reading hits
#=======================================================================


def read_virus_hits(path: Path) -> pl.DataFrame:
    """Read one hits TSV(.gz), selecting and typing the assignment-relevant columns."""
    present = _present_columns(path)
    missing_required = set(_REQUIRED_COLUMNS) - present
    if missing_required:
        raise ValueError(f"{path} is missing required column(s) {sorted(missing_required)}")

    wanted = [source for source in _HITS_COLUMNS if source in present]
    # Read as strings (no inference surprises) and project to just `wanted` at read time.
    raw = pl.read_csv(
        path,
        separator="\t",
        columns=wanted,
        infer_schema_length=0,
        null_values=_NULL_VALUES,
    )
    selected = raw.select(
        [pl.col(source).alias(_HITS_COLUMNS[source]) for source in wanted]
        # Absent optional columns become explicit nulls, so every file yields one schema.
        + [
            pl.lit(None).alias(target)
            for source, target in _OPTIONAL_COLUMNS.items()
            if source not in present
        ],
    )
    return selected.select(_HITS_COLUMNS.values()).with_columns(
        [
            pl.col(name).cast(pl.Float64).cast(dtype)
            if name in _FLOAT_FORMATTED_INTEGERS
            else pl.col(name).cast(dtype)
            for name, dtype in _ASSIGNED_TYPES.items()
        ],
    )


def read_all_virus_hits(paths: Sequence[Path]) -> pl.DataFrame:
    """Read and concatenate several per-sample hits files into one frame.

    ``seq_id`` is globally unique across the generated samples, so the same id appearing in
    two files means the inputs overlap; that would double-count reads in the join, so it is
    rejected here rather than silently inflating the totals.
    """
    if not paths:
        raise ValueError("read_all_virus_hits requires at least one hits file")
    hits = pl.concat([read_virus_hits(path) for path in paths], how="vertical")
    duplicated = hits.filter(pl.col("seq_id").is_duplicated())
    if duplicated.height > 0:
        example = duplicated["seq_id"][0]
        raise ValueError(
            f"duplicate seq_id across hits files ({duplicated.height} rows, e.g. {example!r})",
        )
    return hits


def _present_columns(path: Path) -> set[str]:
    """The column names of a hits file, read from its header alone."""
    header = pl.read_csv(path, separator="\t", n_rows=0, infer_schema_length=0)
    return set(header.columns)
