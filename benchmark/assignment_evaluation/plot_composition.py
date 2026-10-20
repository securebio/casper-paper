"""Read-level outcome for the two positive panels, as stacked bars.
Each read is filed at the finest rank its call gets right.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Mirrors ../code/plot_config.py: FONT_SANS_SERIF, FONT_SIZE_BASE, FONT_SIZE_SMALL, DPI.
FONT_SANS_SERIF = ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"]
FONT_SIZE_BASE = 14
FONT_SIZE_SMALL = 12
DPI = 300

# Bottom to top: the exact call, the coarser call that is still right, then everything else.
# Teal deepens with precision so the stack reads as one quantity losing resolution; red is the
# outcome that is an error.
POSITIVE_ORDER = ("correct_species", "correct_genus", "incorrect")
POSITIVE_COLOUR = {
    "correct_species": "#0F766E",
    "correct_genus": "#2A9D8F",
    "incorrect": "#D1495B",
}
POSITIVE_LABEL = {
    "correct_species": "Correct at species-level",
    "correct_genus": "Correct at genus-level, incorrect at species-level",
    "incorrect": "Incorrect",
}

# The scorer separates a call correct only at family from other wrong calls; both are one
# band here, and the four-way counts remain in ``resolution.csv``.
COLLAPSE = {
    "correct_species": "correct_species",
    "correct_genus": "correct_genus",
    "correct_family": "incorrect",
    "incorrect": "incorrect",
}

POSITIVE_PANELS = ("in_index", "out_of_index")
PANEL_LABEL = {
    "in_index": "In-index set",
    "out_of_index": "Out-of-index set",
}

# Segments thinner than this are labelled beside the bar rather than inside it.
INLINE_LABEL_MIN_PCT = 4.0

# Paths are anchored to this file, not to the working directory, because the figure is drawn
# both from here (`python -m assignment_evaluation.plot_composition`) and from
# `code/index.qmd`, which runs
# with `code/` as its working directory.
BENCHMARK_DIR = Path(__file__).resolve().parents[1]

# The scored run this figure reports, and where the rendered figure belongs: alongside the
# other manuscript figures, not inside benchmark/.
RUN_DIR = BENCHMARK_DIR / "results" / "20260920-2ccd8ff"
OUTPUT_PATH = BENCHMARK_DIR.parent / "figures" / "pipeline_benchmark.png"


def init_style() -> None:
    """Apply the manuscript's plotting style."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_SANS_SERIF,
        "font.size": FONT_SIZE_BASE,
        "axes.labelsize": FONT_SIZE_BASE,
        "xtick.labelsize": FONT_SIZE_SMALL,
        "ytick.labelsize": FONT_SIZE_SMALL,
        "legend.fontsize": FONT_SIZE_SMALL - 2,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "0.15",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def percentages(resolution: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Percentage of simulated pairs in each drawn category, per arm.

    The scorer's categories are folded through ``COLLAPSE`` on the way in. An unrecognised
    one is an error rather than a silent omission: a category the scorer gains but this
    figure does not know about would otherwise vanish from a bar that is meant to be whole.
    """
    summary: dict[str, dict[str, float]] = {}
    for panel in POSITIVE_PANELS:
        totals = dict.fromkeys(POSITIVE_ORDER, 0.0)
        arm = resolution[resolution["panel"] == panel]
        for row in arm.to_dict("records"):
            category = COLLAPSE.get(row["category"])
            if category is None:
                raise KeyError(f"unhandled scoring category {row['category']!r}")
            totals[category] += 100 * row["n_reads"] / row["n_allocated"]
        summary[panel] = totals
    return summary


def annotate_thin_segments(
    ax: plt.Axes, index: int, segments: list[tuple[float, float, str]],
) -> None:
    """Label segments too thin to write inside, stacked clear of each other.

    Thin segments sit adjacently at the top of each bar, so labels placed at each
    segment's midpoint collide and become unreadable. They are pushed apart to a minimum
    spacing and joined back to their segment by a leader line.
    """
    if not segments:
        return
    spacing = 5.0
    placed: list[float] = []
    for mid, _, _ in segments:
        position = mid
        if placed and position - placed[-1] < spacing:
            position = placed[-1] + spacing
        placed.append(position)
    overshoot = max(0.0, placed[-1] - 99.0)
    placed = [position - overshoot for position in placed]
    for (mid, value, colour), position in zip(segments, placed, strict=True):
        ax.annotate(
            f"{value:.2f}%", xy=(index + 0.30, mid), xytext=(index + 0.44, position),
            ha="left", va="center", fontsize=FONT_SIZE_SMALL - 2, color=colour,
            arrowprops={"arrowstyle": "-", "color": colour, "linewidth": 0.6,
                        "shrinkA": 1, "shrinkB": 1},
            annotation_clip=False,
        )


def draw_positive(
    ax: plt.Axes, pct: dict[str, dict[str, float]], categories: Sequence[str],
) -> list:
    """Stacked bars for the two positive arms, on a linear 0-100 axis."""
    x = np.arange(len(POSITIVE_PANELS))
    bottoms = dict.fromkeys(POSITIVE_PANELS, 0.0)
    thin: dict[str, list[tuple[float, float, str]]] = {p: [] for p in POSITIVE_PANELS}
    handles = []
    for category in categories:
        values = np.array([pct[p].get(category, 0.0) for p in POSITIVE_PANELS])
        bases = np.array([bottoms[p] for p in POSITIVE_PANELS])
        bars = ax.bar(
            x, values, bottom=bases, width=0.55, color=POSITIVE_COLOUR[category],
            edgecolor="white", linewidth=0.6, zorder=3,
        )
        handles.append(bars)
        for index, panel in enumerate(POSITIVE_PANELS):
            value = values[index]
            if value <= 0:
                continue
            if value >= INLINE_LABEL_MIN_PCT:
                ax.text(
                    index, bases[index] + value / 2, f"{value:.1f}%", ha="center",
                    va="center", fontsize=FONT_SIZE_SMALL, color="white", fontweight="bold",
                )
            else:
                thin[panel].append(
                    (bases[index] + value / 2, value, POSITIVE_COLOUR[category]),
                )
            bottoms[panel] += value
    for index, panel in enumerate(POSITIVE_PANELS):
        annotate_thin_segments(ax, index, thin[panel])

    ax.set_xticks(x)
    ax.set_xticklabels([PANEL_LABEL[p] for p in POSITIVE_PANELS])
    ax.set_xlim(-0.62, len(POSITIVE_PANELS) - 0.18)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Simulated read pairs (%)")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    return handles


def build_figure(resolution: pd.DataFrame) -> plt.Figure:
    """Two stacked bars, one per positive arm, with a single legend beneath."""
    init_style()
    pct = percentages(resolution)
    figure, ax = plt.subplots(figsize=(7.2, 6.0))
    present = [
        c for c in POSITIVE_ORDER if any(pct[p].get(c, 0.0) > 0 for p in POSITIVE_PANELS)
    ]
    handles = draw_positive(ax, pct, present)
    legend = figure.legend(
        handles, [POSITIVE_LABEL[c] for c in present],
        title="Read outcome", loc="upper left", bbox_to_anchor=(0.11, 0.04),
        handlelength=1.1, alignment="left", borderaxespad=0.0,
    )
    legend.get_title().set_fontsize(FONT_SIZE_SMALL)
    legend.get_title().set_fontweight("bold")
    return figure



def load_resolution(run_dir: Path | str | None = None) -> pd.DataFrame:
    """The scored outcome counts for a run; the manuscript's run by default."""
    return pd.read_csv(Path(run_dir or RUN_DIR) / "resolution.csv")


def print_benchmark_statistics(resolution: pd.DataFrame | None = None) -> None:
    """Print the numbers quoted in the Results and Methods text."""
    resolution = load_resolution() if resolution is None else resolution
    pct = percentages(resolution)
    print("=" * 78)
    print("VV pipeline benchmark: read outcomes by panel")
    print("=" * 78)
    for panel in POSITIVE_PANELS:
        arm = resolution[resolution["panel"] == panel]
        print(f"\n{PANEL_LABEL[panel]} (n = {arm['n_allocated'].iloc[0]:,} read pairs)")
        for category in POSITIVE_ORDER:
            value = pct[panel].get(category, 0.0)
            print(f"  {POSITIVE_LABEL[category]:<52s} {value:6.2f}%")
        correct = pct[panel]["correct_species"] + pct[panel]["correct_genus"]
        print(f"  {'Correct at genus or finer':<52s} {correct:6.2f}%")


def plot_pipeline_benchmark(
    resolution: pd.DataFrame | None = None, save_path: Path | str | None = None,
) -> plt.Figure:
    """Draw the figure, writing it to ``save_path`` when one is given."""
    resolution = load_resolution() if resolution is None else resolution
    figure = build_figure(resolution)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path)
        print(f"Saved figure to {save_path}")
    return figure


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="plot_composition")
    # The run the manuscript reports. Its evaluation CSVs are committed under results/, so
    # the figure regenerates from this repository alone.
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    resolution = load_resolution(args.run_dir)
    print_benchmark_statistics(resolution)
    plot_pipeline_benchmark(resolution, save_path=args.out)


if __name__ == "__main__":
    main()
