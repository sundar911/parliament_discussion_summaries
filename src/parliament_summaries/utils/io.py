from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunked(iterable: Iterable, size: int) -> Iterator[list]:
    """Yield lists of length `size` from `iterable`."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
