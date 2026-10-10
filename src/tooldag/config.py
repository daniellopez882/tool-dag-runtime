"""Where fetched data, the derived catalogue and fitted weights live."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "TOOLDAG_DATA_DIR"


def data_dir() -> Path:
    """Return the data directory: ``$TOOLDAG_DATA_DIR`` or ``~/.cache/tool-dag-runtime``."""
    raw = os.environ.get(DATA_DIR_ENV)
    base = Path(raw) if raw else Path.home() / ".cache" / "tool-dag-runtime"
    return base.expanduser()


def catalog_dir(root: Path | None = None) -> Path:
    return (root or data_dir()) / "catalog"


def weights_path(root: Path | None = None) -> Path:
    return (root or data_dir()) / "scorer_weights.json"
