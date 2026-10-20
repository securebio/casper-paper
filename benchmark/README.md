# VV pipeline benchmark

Sensitivity and specificity of the vertebrate-infecting virus (VV) identification
pipeline, measured on simulated reads. Reads are generated from known genomes, run
through mgs-workflow, and scored against the genome each read came from.

Pinned to the version the manuscript reports: **mgs-workflow 3.0.1 (tag 3.0.1.9),
reference index 20250825, `bt2_score_threshold = 20`**.

## Layout

| | |
|---|---|
| `panels/` | panel selection scripts, and the selected genomes as committed tables |
| `assignment_evaluation/` | scoring and plotting |
| `run/` | the mgs-workflow config and launcher |
| `results/20260920-2ccd8ff/` | the scores the manuscript reports |

## Panels

| Panel | Taxa | Genomes | Accessions | What it measures |
|---|---|---|---|---|
| in-index | 93 | 93 | 165 | genomes the reference database contains |
| out-of-index | 77 | 76 | 134 | genomes absent from it, median 97.9% identity to their nearest indexed relative |
| negative | 189 | 189 | 213 | the non-viral organisms that dominate wastewater, plus the human genome |

Positive panels span 92 species in 27 families.

## Redrawing the figure

The only step reproducible from this directory alone; the evaluation CSVs are committed.

```bash
pip install -e .
python -m assignment_evaluation.plot_composition
```

Writes `../figures/pipeline_benchmark.png`.

## Everything else

Rescoring a pipeline run (`assignment_evaluation/cli.py`) needs its output and the NCBI
taxonomy dumps, neither committed here. Rerunning the pipeline (`run/run_benchmark.sh`)
needs Nextflow, Docker and AWS; it reads `MGS_WORKFLOW_DIR` and `BENCHMARK_BASE`, and the
reference index must already be staged under `$BENCHMARK_BASE/ref/20250825/output`.
Rebuilding the panels needs live NCBI access.

`index_species.tsv`, `kraken_abundance.tsv` and `out_of_index_divergence.tsv` are committed
because they derive from the reference index or the CASPER libraries and cannot be rebuilt
from this repository.

**`fetch_out_of_index.py` queries GenBank live, so it is not reproducible across dates** —
re-running it returns a partly different panel. `panels/out_of_index_genomes.tsv` is the
definition of the arm the manuscript reports.
