# VV pipeline benchmark

Sensitivity and specificity of the vertebrate-infecting virus (VV) identification pipeline
used in this paper, measured on simulated reads from panels of reference genomes. Reads are
generated from known genomes, run through mgs-workflow, and scored against the genome each
read came from.

The pipeline under test is fixed at the version the manuscript reports: **mgs-workflow 3.0.1
(tag 3.0.1.9), reference index 20250825, `bt2_score_threshold = 20`**.

## Layout

| | |
|---|---|
| `panels/` | how the three panels were selected, and the selected genomes as committed tables |
| `assignment_evaluation/` | scoring and plotting: joins pipeline output to ground truth, places each call on the taxonomy, and draws the figure |
| `run/` | the mgs-workflow config and launcher used for the benchmark run |
| `results/20260920-2ccd8ff/` | the scores the manuscript reports |

## Running the pipeline

`run/run_benchmark.sh <reads-dir>` launches mgs-workflow over the simulated panels. It reads:

| | |
|---|---|
| `MGS_WORKFLOW_DIR` | checkout of mgs-workflow at tag 3.0.1.9 (default `~/mgs-workflow`) |
| `BENCHMARK_BASE` | working and output root, on instance storage (default `/scratch/vv-benchmark`) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | must be exported; the profile passes them into the containers |

`BENCHMARK_BASE` must hold the reference index under `ref/20250825/output` before the run
starts. `run/run.config` reads the same variable.

## Panels

| Panel | Taxa | Genomes | Accessions | What it measures |
|---|---|---|---|---|
| in-index | 93 | 93 | 165 | best case: genomes the reference database contains |
| out-of-index | 77 | 76 | 134 | genomes absent from the database, median 97.9% identity to their nearest indexed relative |
| negative | 189 | 189 | 213 | the non-viral organisms that dominate wastewater, plus the human genome |

Positive panels span 92 species in 27 families. Reads are simulated without introduced
divergence; the out-of-index arm supplies real divergence from the database, which is the
quantity of interest.

## Redrawing the figure

Needs only this directory — the evaluation CSVs are committed.

```bash
pip install -e .
python -m assignment_evaluation.plot_composition
```

Writes `../figures/pipeline_benchmark.png`. `results/20260920-2ccd8ff/resolution.csv` holds
all six scoring categories per panel, so the figure can be recut under a different grouping by
editing `COLLAPSE` in `assignment_evaluation/plot_composition.py`.

## Rescoring

Needs the pipeline's output and the NCBI taxonomy the LCA read, neither of which is committed
here (181 MB and ~500 MB respectively).

```bash
python -m assignment_evaluation.cli \
    --hits <virus_hits_final.tsv.gz> \
    --read-counts <read_counts.tsv.gz> \
    --taxonomy-nodes <taxonomy-nodes.dmp> --taxonomy-names <taxonomy-names.dmp> \
    --out-dir results/<run-id> --bt2-score-threshold 20
```

Denominators come from `panels/read_allocation.tsv`: reads the pipeline discarded never appear
in its output and can only be counted as the shortfall against what was generated.

## Rebuilding the panels

The scripts in `panels/` write the committed TSVs, and run in this order:

```
positive_taxa.py          the taxa list, resolved against the NCBI taxonomy
select_in_index_genomes.py  one RefSeq assembly per taxon, resolved into the index
fetch_out_of_index.py     post-cutoff GenBank genomes absent from the index
fetch_panel_sequences.py  the in-index arm's sequences, from GenBank
fetch_negative_genomes.py one RefSeq assembly per negative taxon
build_panel_manifest.py   the three panels as one ground-truth table
allocate_reads.py         read pairs per genome
```

`simulate_panels.py` then generates the reads, and `measure_divergence.py` measures the
out-of-index arm against the reference index.

Two inputs and one output cannot be reproduced outside the project, and are committed for
that reason: `index_species.tsv`, the species catalogue derived from the reference index;
`kraken_abundance.tsv`, which ranks negative taxa across the manuscript's libraries; and
`out_of_index_divergence.tsv`, which needs the index's masked genome FASTA.

**`fetch_out_of_index.py` queries GenBank live and so is not reproducible across dates** —
GenBank grows and records are retired, so re-running it returns a partly different panel.
`panels/out_of_index_genomes.tsv` is the definition of the arm the manuscript reports.
