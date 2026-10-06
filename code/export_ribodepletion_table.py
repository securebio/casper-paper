#!/usr/bin/env python3
"""
Supplementary table: the SB libraries prepared without rRNA-depletion probes.

Lists every sample affected by the reagent-preparation error described in the
Methods, so a reader reusing this release can identify and exclude or flag them.
The window and the status derivation are imported from
``plot_ribodepletion_effect`` rather than restated here, so the figure, this
table and the `ribodepletion_status` metadata column cannot drift apart.

The table is deliberately an identification list and carries no measurements:
the effect on rRNA and vertebrate-infecting virus recovery is what the
accompanying figure shows, and repeating per-sample values here would invite the
two to disagree.

Accessions are BioSample (``SAMN...``) accessions, which is what the
``sra_accession`` column of the release metadata actually holds; they resolve to
SRA runs under the release BioProject.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from plot_ribodepletion_effect import (
    SITES,
    STATUS_OMITTED,
    prepare_ribodepletion_data,
)

OUTPUT_PATH = '../tables/ribodepletion_omitted_samples.tsv'


def build_ribodepletion_table(df=None):
    """One row per library prepared without rRNA-depletion probes.

    Args:
        df: Pre-loaded frame from ``prepare_ribodepletion_data``, else loaded
            here.

    Returns:
        pd.DataFrame, 22 rows ordered by site then collection date.
    """
    if df is None:
        df = prepare_ribodepletion_data()

    omitted = df[df['ribodepletion_status'] == STATUS_OMITTED]

    records = []
    for site in SITES:
        site_rows = omitted[omitted['site_name'] == site].sort_values('date')
        for _, row in site_rows.iterrows():
            records.append({
                'Site': site,
                'Collection date': row['date'].strftime('%Y-%m-%d'),
                'BioSample accession': row['sra_accession'],
            })

    return pd.DataFrame(records)


def main(save_path=OUTPUT_PATH):
    table = build_ribodepletion_table()
    print("SB libraries prepared without rRNA-depletion probes\n")
    print(table.to_string(index=False))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_path, sep='\t', index=False)
        print(f"\nSaved {save_path}")
    return table


if __name__ == '__main__':
    main()
