#!/usr/bin/env bash
#
# run_benchmark.sh -- run mgs-workflow over the simulated panels.
#
# Usage:  ./run_benchmark.sh <reads-dir>
#
#   <reads-dir> holds the FASTQs and the samplesheet.csv written by
#   panels/simulate_panels.py. The samplesheet names absolute paths, so it does not
#   matter where it is launched from -- but Nextflow is run from a clean directory
#   regardless, because it writes .nextflow/ and reports into the launch directory.
#
# Environment:
#   MGS_WORKFLOW_DIR  checkout of mgs-workflow at tag 3.0.1.9 (default ~/mgs-workflow)
#   BENCHMARK_BASE    working and output root, on instance storage rather than EBS;
#                     must match the value run.config uses (default /scratch/vv-benchmark)
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
#                     must be exported: the profile interpolates them into
#                     docker.runOptions, and .command.run runs under `set -u`, so
#                     leaving them unset aborts every task. With AWS CLI v2,
#                     eval "$(aws configure export-credentials --format env)"
#
# NXF_VER is pinned because 3.0.1.9 declares nextflowVersion '!>=25.10.2'; newer
# majors reject the config.
#
# Runs under the vv_local profile defined in run.config: ec2_local plus a mount of
# BENCHMARK_BASE, without which Kraken2 cannot reach its database from inside the
# container.
#
set -euo pipefail

READS_DIR="${1:?usage: run_benchmark.sh <reads-dir>}"
WORKFLOW="${MGS_WORKFLOW_DIR:-$HOME/mgs-workflow}"
BASE="${BENCHMARK_BASE:-/scratch/vv-benchmark}"
LAUNCH="$BASE/launch"
CONFIG="$(cd "$(dirname "$0")" && pwd)/run.config"

export NXF_VER=25.10.2

: "${AWS_ACCESS_KEY_ID:?export AWS credentials before running; see Environment above}"
: "${AWS_SECRET_ACCESS_KEY:?export AWS credentials before running; see Environment above}"

[ -d "$WORKFLOW" ] || { echo "ERROR: no mgs-workflow checkout at $WORKFLOW; set MGS_WORKFLOW_DIR" >&2; exit 1; }
[ -f "$READS_DIR/samplesheet.csv" ] || { echo "ERROR: no samplesheet.csv in $READS_DIR" >&2; exit 1; }

mkdir -p "$LAUNCH"
cp "$CONFIG" "$LAUNCH/nextflow.config"
cp "$READS_DIR/samplesheet.csv" "$LAUNCH/samplesheet.csv"
cd "$LAUNCH"

echo "=== samples ==="
cat samplesheet.csv
echo "=== running mgs-workflow $(git -C "$WORKFLOW" describe --tags) ==="

nextflow run -resume -profile vv_local "$WORKFLOW"

echo
echo "=== output ==="
find "$BASE/output/results" -name "virus_hits_final.tsv.gz" -exec ls -la {} \;

# The reads that passed the k-mer screen, needed to tell a screening miss from an alignment
# miss when scoring. Published under intermediates/ rather than results/, so it is listed here
# explicitly -- it lives under BENCHMARK_BASE, which is wiped when the instance stops.
#
# Note the filename: the workflow publishes `bbduk_ch.fail`, and in the non-interleaved BBDUK
# process `fail` is bbduk's `outm` -- the reads that MATCHED viral k-mers, i.e. the survivors.
# "fail" here means "failed to be filtered out". The `*_bbduk_match*` files from the
# interleaved process stay in the Nextflow work directory and are not published.
echo
echo "=== k-mer screen survivors (copy these with the results) ==="
find "$BASE/output/intermediates/reads/raw_viral" -name "*bbduk_fail.fastq.gz" \
    -exec ls -la {} \; 2>/dev/null || echo "  none found -- intermediates may not have been published"
echo
echo "Next: copy the results somewhere durable, then"
echo "  python -m assignment_evaluation.cli --hits <virus_hits_final.tsv.gz> \\"
echo "      --read-counts <read_counts.tsv.gz> \\"
echo "      --taxonomy-nodes $BASE/ref/20250825/output/results/taxonomy-nodes.dmp \\"
echo "      --taxonomy-names $BASE/ref/20250825/output/results/taxonomy-names.dmp \\"
echo "      --out-dir results/ --bt2-score-threshold 20"
