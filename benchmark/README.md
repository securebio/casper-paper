# VV pipeline benchmark

Sensitivity and specificity of the vertebrate-infecting virus (VV) identification
pipeline, measured on simulated reads. Reads are generated from known genomes, run
through mgs-workflow (v3.0.1, index 20250825), and scored against the genome each read came from.

## Layout

| | |
|---|---|
| `panels/` | panel selection scripts and genome accessions |
| `assignment_evaluation/` | scoring and plotting |
| `run/` | the mgs-workflow config and launcher |
| `results/20260920-2ccd8ff/` | the scores the manuscript reports |

## Panels

| Panel | Taxa | Genomes | Accessions | Description |
|---|---|---|---|---|
| in-index | 93 | 93 | 165 | VV genomes contained in the reference database |
| out-of-index | 77 | 76 | 134 | VV genomes absent from the reference database |
| negative | 189 | 189 | 213 | most abundant non-VV taxa |

## Reproducing this analysis

The only step reproducible from this directory alone is plotting the evaluation results.

```bash
pip install -e .
python -m assignment_evaluation.plot_composition
```

Writes `../figures/pipeline_benchmark.png`.

Rescoring a pipeline run (`assignment_evaluation/cli.py`) needs its output and the NCBI
taxonomy dumps, which are not committed here. Rerunning the pipeline (`run/run_benchmark.sh`)
needs Nextflow, Docker and AWS; it reads `MGS_WORKFLOW_DIR` and `BENCHMARK_BASE`, and the
reference index must already be staged under `$BENCHMARK_BASE/ref/20250825/output`.
Rebuilding the panels needs live NCBI access.

`index_species.tsv`, `kraken_abundance.tsv` and `out_of_index_divergence.tsv` are committed
because they derive from the reference index or the CASPER libraries and cannot be rebuilt
from this repository.

**`fetch_out_of_index.py` queries GenBank live, so it is not reproducible across dates** —
re-running it returns a partly different panel. `panels/out_of_index_genomes.tsv` is the
definition of the arm the manuscript reports.
