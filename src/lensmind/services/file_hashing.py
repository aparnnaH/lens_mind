from __future__ import annotations

import hashlib
from pathlib import Path


def calculate_sha256(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()
