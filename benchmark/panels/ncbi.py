"""A small, rate-limited, stdlib-only NCBI E-utilities client shared by the panel
builders.
"""

#=======================================================================
# Imports
#=======================================================================

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

#=======================================================================
# Constants
#=======================================================================

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "casper-pipeline-benchmark"

# NCBI permits 3 requests/second without an API key and 10 with one; stay just inside both.
RATE_DELAY_NO_KEY = 0.40
RATE_DELAY_WITH_KEY = 0.12

# E-utilities returns transient 429/50x under load often enough to need retries.
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 2.0

# Default cap on search hits considered per query.
SEARCH_RETMAX = 200

#=======================================================================
# Client
#=======================================================================


class Eutils:
    """A minimal, rate-limited E-utilities client (stdlib only, so the repo has no new dep)."""

    def __init__(self, email: str | None = None, api_key: str | None = None) -> None:
        self.email = email
        self.api_key = api_key
        self.delay = RATE_DELAY_WITH_KEY if api_key else RATE_DELAY_NO_KEY
        self._last_request = 0.0

    def _params(self, extra: dict[str, str]) -> dict[str, str]:
        params = {"tool": TOOL_NAME, **extra}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _get(self, endpoint: str, params: dict[str, str]) -> str:
        url = f"{EUTILS_BASE}/{endpoint}?" + urllib.parse.urlencode(self._params(params))
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self._last_request = time.monotonic()
            try:
                with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                    body: str = response.read().decode()
                    return body
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
                time.sleep(BACKOFF_SECONDS * (attempt + 1))
        raise RuntimeError(f"{endpoint} failed after {MAX_ATTEMPTS} attempts: {last_error}")

    def esearch(self, term: str, retmax: int = SEARCH_RETMAX) -> list[str]:
        """UIDs matching a nuccore query, newest first."""
        payload = json.loads(
            self._get(
                "esearch.fcgi",
                {
                    "db": "nuccore",
                    "term": term,
                    "retmax": str(retmax),
                    "retmode": "json",
                    "sort": "date_modified",
                },
            ),
        )
        return list(payload.get("esearchresult", {}).get("idlist", []))

    def esummary(self, ids: list[str]) -> list[dict[str, Any]]:
        """Document summaries for UIDs or accession.versions, in batches."""
        summaries: list[dict[str, Any]] = []
        for start in range(0, len(ids), 200):
            batch = ids[start : start + 200]
            payload = json.loads(
                self._get(
                    "esummary.fcgi",
                    {"db": "nuccore", "id": ",".join(batch), "retmode": "json"},
                ),
            )
            result = payload.get("result", {})
            summaries.extend(result[uid] for uid in result.get("uids", []) if uid in result)
        return summaries

    def efetch_fasta(self, accessions: list[str]) -> str:
        """FASTA text for the given accessions, in batches."""
        chunks: list[str] = []
        for start in range(0, len(accessions), 100):
            batch = accessions[start : start + 100]
            chunks.append(
                self._get(
                    "efetch.fcgi",
                    {
                        "db": "nuccore",
                        "id": ",".join(batch),
                        "rettype": "fasta",
                        "retmode": "text",
                    },
                ),
            )
        return "".join(chunks)


#=======================================================================
# Sequence helpers
#=======================================================================


def parse_fasta(text: str) -> dict[str, str]:
    """Parse FASTA text into ``{first header token: sequence}``."""
    records: dict[str, str] = {}
    record_id: str | None = None
    chunks: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if record_id is not None:
                records[record_id] = "".join(chunks)
            record_id = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if record_id is not None:
        records[record_id] = "".join(chunks)
    return records


def usable_window_fraction(sequence: str, window: int = 150) -> float:
    """Fraction of ``window``-length positions that contain only unambiguous ACGT bases.

    This is the quantity that actually matters for read generation: the generator discards
    any fragment overlapping a non-ACGT base, so a genome's usefulness is the share of
    read-length windows it can still produce -- not its overall ambiguity fraction. The two
    come apart badly on large genomes. 0.28% ambiguous bases concentrated in two herpesvirus
    repeat regions costs almost no coverage, while the same fraction scattered singly would
    ruin nearly half the windows, and a flat per-base threshold cannot tell those apart (it
    also penalises long genomes, which carry more absolute ambiguity at the same rate).
    """
    if not sequence or len(sequence) < window:
        return 0.0
    upper = sequence.upper()
    # Prefix count of ambiguous bases, so each window is an O(1) lookup.
    prefix = [0] * (len(upper) + 1)
    for i, base in enumerate(upper):
        prefix[i + 1] = prefix[i] + (base not in "ACGT")
    n_windows = len(upper) - window + 1
    usable = sum(
        1 for start in range(n_windows) if prefix[start + window] == prefix[start]
    )
    return usable / n_windows
