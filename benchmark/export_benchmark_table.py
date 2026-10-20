"""Supplementary table: the read counts behind the benchmark figure."""

#=======================================================================
# Imports
#=======================================================================

import json
from pathlib import Path

import pandas as pd

#=======================================================================
# Constants
#=======================================================================

BENCHMARK_DIR = Path(__file__).resolve().parent
RUN_DIR = BENCHMARK_DIR / "results" / "20260920-2ccd8ff"
OUTPUT_PATH = BENCHMARK_DIR.parent / "tables" / "pipeline_benchmark.tsv"
FAMILY_OUTPUT_PATH = BENCHMARK_DIR.parent / "tables" / "pipeline_benchmark_by_family.tsv"

PANEL_LABEL = {
    "in_index": "In-index (positive)",
    "out_of_index": "Out-of-index (positive)",
    "negative": "Negative",
}

# The table's outcome columns are the supplementary figure's three bands, taken from the
# figure module itself so the two cannot drift: a category added to the scorer shows up in
# both or neither. `COLLAPSE` folds the scorer's six categories into those three, and
# `POSITIVE_LABEL` supplies the same wording the figure legend uses.

#=======================================================================
# Table
#=======================================================================


_SUPERSCRIPT = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")


def _count(numerator: int, denominator: int) -> str:
    """A read-pair count, with its share of the panel in parentheses.

    The share is written in scientific notation when it is far below 1%, which is the regime
    the negative panel sits in: a plain decimal there is a run of zeros nobody can read.
    """
    if denominator == 0:
        return "—"
    value = 100 * numerator / denominator
    if value >= 0.01 or value == 0:
        return f"{numerator:,} ({value:.2f}%)"
    mantissa, _, exponent = f"{value:.1e}".partition("e")
    power = str(int(exponent)).translate(_SUPERSCRIPT)
    return f"{numerator:,} ({mantissa} \u00d7 10{power}%)"


def build_benchmark_table(run_dir: Path | str | None = None) -> pd.DataFrame:
    """The per-panel summary, on the same schema as the per-family table.

    One row per panel: the taxa it contains, then accuracy as percentages of the read pairs
    simulated for it. The genus column is cumulative and includes the species-correct reads,
    so it and ``incorrect`` sum to 100% while ``species`` is a subset of it.

    The negative panel has no taxonomic columns -- any viral call on a non-viral read is
    wrong whatever it names -- so it carries a false-positive count instead.

    Args:
        run_dir: scored run directory. Defaults to the run the manuscript reports.
    """
    run_dir = Path(run_dir or RUN_DIR)
    resolution = pd.read_csv(run_dir / "resolution.csv")
    negative = pd.read_csv(run_dir / "negative_per_genome.csv")
    manifest = pd.read_csv(BENCHMARK_DIR / "panels" / "panel_manifest.tsv", sep="\t")

    rows = []
    for panel in ("in_index", "out_of_index", "negative"):
        taxa = manifest[manifest["panel"] == panel]["species_taxid"].nunique()
        row = {"Panel": PANEL_LABEL[panel], "Taxa": taxa}
        if panel == "negative":
            pairs = int(negative["n_pairs"].sum())
            false_positive = int(negative["n_false_positive"].sum())
            row |= {
                "Correct at species (%)": "—",
                "Correct at genus (%)": "—",
                "Incorrect (%)": "—",
                "False positives": _count(false_positive, pairs),
            }
        else:
            arm = resolution[resolution["panel"] == panel]
            counts = dict(zip(arm["category"], arm["n_reads"], strict=True))
            pairs = int(arm["n_allocated"].iloc[0])
            species = counts["correct_species"]
            genus = species + counts["correct_genus"]
            row |= {
                "Correct at species (%)": round(100 * species / pairs, 2),
                "Correct at genus (%)": round(100 * genus / pairs, 2),
                "Incorrect (%)": round(100 * (pairs - genus) / pairs, 2),
                "False positives": "—",
            }
        rows.append(row)
    return pd.DataFrame(rows)


def build_family_table(run_dir: Path | str | None = None) -> pd.DataFrame:
    """Per-family accuracy, with the two positive panels side by side.

    One row per virus family, and one column group per panel: the taxa that family
    contributes to the panel, then accuracy as percentages of the read pairs simulated for
    it. The genus column is cumulative and includes the species-correct reads, so it and
    ``incorrect`` sum to 100% while ``species`` is a subset of it. A family absent from a
    panel has blank cells there.
    """
    run_dir = Path(run_dir or RUN_DIR)
    table = pd.read_csv(run_dir / "resolution_by_family.csv")
    pairs = table["n_allocated"]
    table = table.assign(
        taxa=table["n_taxa"],
        species=(100 * table["correct_species"] / pairs).round(2),
        # Cumulative: a read correct at species is also correct at genus. This column and
        # `incorrect` therefore sum to 100%, while `species` is a subset of this one.
        genus=(100 * (table["correct_species"] + table["correct_genus"]) / pairs).round(2),
        incorrect=(100 * (table["correct_family"] + table["incorrect"]) / pairs).round(2),
    )
    wide = table.pivot(
        index="family", columns="panel", values=["taxa", "species", "genus", "incorrect"],
    )
    # Flat column names: a MultiIndex header renders as tuples through to_markdown, which is
    # what the manuscript displays.
    label = {"taxa": "taxa", "species": "correct at species (%)",
             "genus": "correct at genus (%)", "incorrect": "incorrect (%)"}
    out = pd.DataFrame({"Family": wide.index})
    for panel, prefix in (("in_index", "In-index"), ("out_of_index", "Out-of-index")):
        for field, name in label.items():
            column = wide[(field, panel)] if (field, panel) in wide.columns else None
            values = column.to_numpy() if column is not None else pd.NA
            out[f"{prefix} {name}"] = values
        # The pivot widens through float to allow gaps; taxa are counts.
        out[f"{prefix} taxa"] = out[f"{prefix} taxa"].astype("Int64")
    return out


def print_benchmark_table_statistics(run_dir: Path | str | None = None) -> None:
    """Print the negative panel's aggregate rates, which the table cannot carry per-panel."""
    run_dir = Path(run_dir or RUN_DIR)
    meta = json.loads((run_dir / "metadata.json").read_text())
    negative = pd.read_csv(run_dir / "negative_per_genome.csv")
    with_fp = negative[negative["n_false_positive"] > 0]
    print("=" * 78)
    print("Negative panel")
    print("=" * 78)
    print(f"  taxa:                                {len(negative):,}")
    print(f"  taxa with any false positive:        {len(with_fp):,}")
    print(f"  spurious pairs per 10^9 (observed):  {meta['spurious_per_billion_observed']:.1f}")
    print(f"  spurious pairs per 10^9 (95% bound): {meta['spurious_per_billion_bound']:,.0f}")


#=======================================================================
# CLI
#=======================================================================


def main(
    save_path: Path | str | None = OUTPUT_PATH,
    family_save_path: Path | str | None = FAMILY_OUTPUT_PATH,
) -> pd.DataFrame:
    table = build_benchmark_table()
    print_benchmark_table_statistics()
    print()
    print(table)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_path, sep="\t", index=False)
        print(f"\nSaved {save_path}")
    if family_save_path:
        family = build_family_table()
        Path(family_save_path).parent.mkdir(parents=True, exist_ok=True)
        family.to_csv(family_save_path, sep="\t", index=False)
        print(f"Saved {family_save_path}")
    return table


if __name__ == "__main__":
    main()
