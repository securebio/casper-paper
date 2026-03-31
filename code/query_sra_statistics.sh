#!/bin/bash
# Query SRA statistics for wastewater metagenomic sequencing data
# Compares CASPER and related projects against total SRA and wastewater categories
#
# Requires: gcloud CLI with BigQuery access configured
# Usage: ./query_sra_statistics.sh [output_dir]
#
# Outputs (no timestamp suffix — overwrites existing files):
#   sra_summary_statistics.csv
#   sra_monthly_timeline.csv
#   sra_monthly_timeline_by_collection.csv

set -e

# Configuration
PROJECT_ID="broad-sabeti-lab"
OUTPUT_DIR="${1:-$(dirname "$0")/../data/sra_statistics}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Querying SRA statistics..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# Query 1: Summary statistics by category
echo "Running Query 1: Summary statistics by category..."
bq query --project_id="$PROJECT_ID" --use_legacy_sql=false --format=csv '
SELECT
  category,
  runs,
  mbases,
  mbytes,
  ROUND(mbases/1e3, 3) as gigabases,
  ROUND(mbases/1e6, 4) as terabases,
  ROUND(mbytes/1e3, 2) as gigabytes,
  ROUND(mbytes/1e6, 3) as terabytes
FROM (
  SELECT "1_all_sra" as category, COUNT(*) as runs, SUM(mbases) as mbases, SUM(mbytes) as mbytes
  FROM `nih-sra-datastore.sra.metadata`

  UNION ALL

  SELECT "2_wastewater_metagenome" as category, COUNT(*) as runs, SUM(mbases) as mbases, SUM(mbytes) as mbytes
  FROM `nih-sra-datastore.sra.metadata`
  WHERE organism IN ("wastewater metagenome", "sludge metagenome")

  UNION ALL

  SELECT "3_wastewater_shotgun_metagenomic" as category, COUNT(*) as runs, SUM(mbases) as mbases, SUM(mbytes) as mbytes
  FROM `nih-sra-datastore.sra.metadata`
  WHERE organism IN ("wastewater metagenome", "sludge metagenome")
    AND librarysource IN ("METAGENOMIC", "METATRANSCRIPTOMIC")
    AND assay_type IN ("RNA-Seq", "WGS")

  UNION ALL

  SELECT "4_wastewater_shotgun_metagenomic_excl_casper" as category, COUNT(*) as runs, SUM(mbases) as mbases, SUM(mbytes) as mbytes
  FROM `nih-sra-datastore.sra.metadata`
  WHERE organism IN ("wastewater metagenome", "sludge metagenome")
    AND librarysource IN ("METAGENOMIC", "METATRANSCRIPTOMIC")
    AND assay_type IN ("RNA-Seq", "WGS")
    AND bioproject != "PRJNA1247874"

  UNION ALL

  SELECT "5_bioproject_PRJNA1247874_casper" as category, COUNT(*) as runs, SUM(mbases) as mbases, SUM(mbytes) as mbytes
  FROM `nih-sra-datastore.sra.metadata`
  WHERE bioproject = "PRJNA1247874"

  UNION ALL

  SELECT "6_bioproject_PRJNA1198001_casper_rothman" as category, COUNT(*) as runs, SUM(mbases) as mbases, SUM(mbytes) as mbytes
  FROM `nih-sra-datastore.sra.metadata`
  WHERE bioproject = "PRJNA1198001"
)
ORDER BY category
' > "$OUTPUT_DIR/sra_summary_statistics.csv"

echo "  Saved: sra_summary_statistics.csv"

# Query 2: Monthly release timeline for key categories
echo "Running Query 2: Monthly release timeline..."
bq query --project_id="$PROJECT_ID" --use_legacy_sql=false --format=csv --max_rows=100000 '
SELECT
  EXTRACT(YEAR from releasedate) as year,
  EXTRACT(MONTH from releasedate) as month,
  category,
  COUNT(*) as runs,
  SUM(mbases) as mbases,
  SUM(mbytes) as mbytes
FROM (
  SELECT releasedate, "all_sra" as category, mbases, mbytes
  FROM `nih-sra-datastore.sra.metadata`

  UNION ALL

  SELECT releasedate, "wastewater_metagenome" as category, mbases, mbytes
  FROM `nih-sra-datastore.sra.metadata`
  WHERE organism IN ("wastewater metagenome", "sludge metagenome")

  UNION ALL

  SELECT releasedate, "wastewater_shotgun_metagenomic" as category, mbases, mbytes
  FROM `nih-sra-datastore.sra.metadata`
  WHERE organism IN ("wastewater metagenome", "sludge metagenome")
    AND librarysource IN ("METAGENOMIC", "METATRANSCRIPTOMIC")
    AND assay_type IN ("RNA-Seq", "WGS")

  UNION ALL

  SELECT releasedate, "casper_PRJNA1247874" as category, mbases, mbytes
  FROM `nih-sra-datastore.sra.metadata`
  WHERE bioproject = "PRJNA1247874"

  UNION ALL

  SELECT releasedate, "casper_rothman_PRJNA1198001" as category, mbases, mbytes
  FROM `nih-sra-datastore.sra.metadata`
  WHERE bioproject = "PRJNA1198001"
)
WHERE releasedate IS NOT NULL
GROUP BY year, month, category
ORDER BY year, month, category
' > "$OUTPUT_DIR/sra_monthly_timeline.csv"

echo "  Saved: sra_monthly_timeline.csv"

# Query 3: Monthly timeline by collection date (sample collection, not upload)
echo "Running Query 3: Monthly timeline by collection date..."
bq query --project_id="$PROJECT_ID" --use_legacy_sql=false --format=csv --max_rows=100000 '
SELECT
  EXTRACT(YEAR from collection_date_sam) as year,
  EXTRACT(MONTH from collection_date_sam) as month,
  category,
  COUNT(*) as runs,
  SUM(mbases) as mbases,
  SUM(mbytes) as mbytes
FROM (
  SELECT collection_date_sam, "wastewater_metagenome" as category, mbases, mbytes
  FROM `nih-sra-datastore.sra.metadata`
  WHERE organism IN ("wastewater metagenome", "sludge metagenome")

  UNION ALL

  SELECT collection_date_sam, "wastewater_shotgun_metagenomic" as category, mbases, mbytes
  FROM `nih-sra-datastore.sra.metadata`
  WHERE organism IN ("wastewater metagenome", "sludge metagenome")
    AND librarysource IN ("METAGENOMIC", "METATRANSCRIPTOMIC")
    AND assay_type IN ("RNA-Seq", "WGS")

  UNION ALL

  SELECT collection_date_sam, "casper_PRJNA1247874" as category, mbases, mbytes
  FROM `nih-sra-datastore.sra.metadata`
  WHERE bioproject = "PRJNA1247874"

  UNION ALL

  SELECT collection_date_sam, "casper_rothman_PRJNA1198001" as category, mbases, mbytes
  FROM `nih-sra-datastore.sra.metadata`
  WHERE bioproject = "PRJNA1198001"
)
WHERE collection_date_sam IS NOT NULL
GROUP BY year, month, category
ORDER BY year, month, category
' > "$OUTPUT_DIR/sra_monthly_timeline_by_collection.csv"

echo "  Saved: sra_monthly_timeline_by_collection.csv"

echo ""
echo "All queries complete. Output files:"
ls -la "$OUTPUT_DIR"/sra_*.csv

echo ""
echo "=== Summary Statistics ==="
cat "$OUTPUT_DIR/sra_summary_statistics.csv" | column -t -s,

echo ""
echo "=== Key Findings ==="
CASPER_TB=$(awk -F, '/PRJNA1247874_casper/ {print $6}' "$OUTPUT_DIR/sra_summary_statistics.csv")
SHOTGUN_META_TB=$(awk -F, '/3_wastewater_shotgun_metagenomic,/ {print $6}' "$OUTPUT_DIR/sra_summary_statistics.csv")
SHOTGUN_META_EXCL_TB=$(awk -F, '/excl_casper/ {print $6}' "$OUTPUT_DIR/sra_summary_statistics.csv")
WW_ALL_TB=$(awk -F, '/2_wastewater_metagenome,/ {print $6}' "$OUTPUT_DIR/sra_summary_statistics.csv")
PRIOR_TB=$(awk -F, '/PRJNA1198001/ {print $6}' "$OUTPUT_DIR/sra_summary_statistics.csv")

echo "CASPER (PRJNA1247874): ${CASPER_TB} terabases"
echo "Prior project (PRJNA1198001): ${PRIOR_TB} terabases"
echo ""
echo "CASPER as % of shotgun metagenomic wastewater: $(echo "scale=1; 100 * $CASPER_TB / $SHOTGUN_META_TB" | bc)%"
echo "CASPER vs all other shotgun metagenomic wastewater: $(echo "scale=1; $CASPER_TB / $SHOTGUN_META_EXCL_TB" | bc)x larger"
echo "CASPER as % of all wastewater metagenome: $(echo "scale=1; 100 * $CASPER_TB / $WW_ALL_TB" | bc)%"

echo ""
echo "Querying median bases per CASPER biosample..."
bq query --project_id="$PROJECT_ID" --use_legacy_sql=false --format=csv '
SELECT
  ROUND(median_gbases_per_biosample, 1) as median_gbases_per_biosample,
  ROUND(mean_gbases_per_biosample, 1) as mean_gbases_per_biosample,
  n_biosamples
FROM (
  SELECT
    APPROX_QUANTILES(total_mbases / 1e3, 100)[OFFSET(50)] as median_gbases_per_biosample,
    AVG(total_mbases / 1e3) as mean_gbases_per_biosample,
    COUNT(*) as n_biosamples
  FROM (
    SELECT biosample, SUM(mbases) as total_mbases
    FROM `nih-sra-datastore.sra.metadata`
    WHERE bioproject = "PRJNA1247874"
    GROUP BY biosample
  )
)
'
