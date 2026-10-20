"""Record the panels, allocation, seed and InSilicoSeq version a run was made from."""

#=======================================================================
# Imports
#=======================================================================

import json
from pathlib import Path
from typing import Any

#=======================================================================
# Metadata
#=======================================================================


def write_metadata(metadata: dict[str, Any], path: Path) -> None:
    """Write run metadata, merging into an existing file rather than overwriting it.

    A run is filled in stages -- staged, then simulated, then run through the pipeline, then
    scored -- and each stage adds to the record instead of replacing what came before.
    """
    merged: dict[str, Any] = {}
    if path.exists():
        merged = json.loads(path.read_text())
    merged.update(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
