#!/usr/bin/env python3
"""
Add the repo-only `ribodepletion_status` column to data/sample_metadata.csv.

The release metadata records concentration, extraction and platform but nothing
about rRNA depletion, so a reader cannot tell from the shipped files which
libraries were affected by the reagent-preparation error described in the
Methods. This script materialises that fact as a column.

It is a script rather than a hand edit because `data/sample_metadata.csv` is a
generated artifact: it is exported upstream from the internal library metadata,
so a manual edit would be unauditable and lost on the next export. Keeping the
rule in committed code makes the derivation inspectable and the result
reproducible. The derivation itself lives in `plot_ribodepletion_effect`, which
is also what the figure and the table use, so the three cannot disagree.

The status cannot be read from the internal metadata either: the `ribodepleted`
field there is a per-laboratory constant (true for every SB library, false for
every MU one) and does not record the omission, which is why it is derived from
laboratory and collection date here.

Re-running is a no-op: the column is recomputed and overwritten with identical
content, and no other column is touched.

Note this column is not in the SRA or BioSample metadata; it exists only in this
repository.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from plot_ribodepletion_effect import (
    STATUS_DEPLETED,
    STATUS_NONE,
    STATUS_OMITTED,
    ribodepletion_status,
)

DEFAULT_PATH = '../data/sample_metadata.csv'

COLUMN = 'ribodepletion_status'

# Placed in protocol order: concentration -> extraction -> ribodepletion ->
# platform, so the column sits where a reader scanning the prep steps expects.
INSERT_AFTER = 'extraction'

# A reagent-error window is the kind of constant that rots silently. These are
# asserted before writing so that a change upstream fails loudly rather than
# quietly reclassifying samples.
EXPECTED_COUNTS = {
    STATUS_NONE: 1078,
    STATUS_DEPLETED: 106,
    STATUS_OMITTED: 22,
}
EXPECTED_OMITTED_BY_SITE = {
    'Boston DITP North, MA': 7,
    'Boston DITP South, MA': 7,
    'Miami-Dade CDWWTP, FL': 8,
}


def annotate(df):
    """Return a copy of the metadata frame with the status column inserted."""
    df = df.copy()
    status = [ribodepletion_status(lab, date)
              for lab, date in zip(df['sequencing_lab'], df['date'])]
    if COLUMN in df.columns:
        df[COLUMN] = status
    else:
        df.insert(df.columns.get_loc(INSERT_AFTER) + 1, COLUMN, status)
    return df


def _check(df):
    """Fail loudly if the derived counts have moved."""
    counts = df[COLUMN].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise AssertionError(
            f'ribodepletion_status counts changed: expected {EXPECTED_COUNTS}, '
            f'got {counts}. If the window in plot_ribodepletion_effect was '
            f'corrected, update EXPECTED_COUNTS to match.')

    omitted = df[df[COLUMN] == STATUS_OMITTED]['site_name'].value_counts()
    if omitted.to_dict() != EXPECTED_OMITTED_BY_SITE:
        raise AssertionError(
            f'omitted samples per site changed: expected '
            f'{EXPECTED_OMITTED_BY_SITE}, got {omitted.to_dict()}.')


def main(path=DEFAULT_PATH, check_only=False):
    # Read without date parsing so every other column round-trips byte for byte;
    # only the new column should appear in the diff.
    df = pd.read_csv(path, dtype=str)
    annotated = annotate(df)
    _check(annotated)

    print(f'{COLUMN} over {len(annotated):,} samples:')
    for status, n in annotated[COLUMN].value_counts().items():
        print(f'  {status:<18} {n:5,}')
    print('\nRiboPOOLs omitted, by site:')
    for site, n in (annotated[annotated[COLUMN] == STATUS_OMITTED]
                    ['site_name'].value_counts().items()):
        print(f'  {site:<24} {n:3}')

    if check_only:
        print('\nCheck only, nothing written.')
        return annotated

    annotated.to_csv(path, index=False)
    print(f'\nSaved {path}')
    return annotated


if __name__ == '__main__':
    main(check_only='--check' in sys.argv)
