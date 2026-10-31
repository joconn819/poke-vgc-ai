"""Small JSONL I/O helpers shared by corpus readers."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import BinaryIO


def open_jsonl_binary(path: str | Path) -> BinaryIO:
    """Open JSONL, transparently resolving a losslessly compressed archive."""
    source = Path(path)
    if not source.exists() and source.suffix != ".gz":
        compressed = source.with_suffix(source.suffix + ".gz")
        if compressed.exists():
            source = compressed
    if source.suffix == ".gz":
        return gzip.open(source, "rb")
    return source.open("rb")
