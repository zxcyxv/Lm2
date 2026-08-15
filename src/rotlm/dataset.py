"""Materialize and memory-map the repository's tokenized WikiText-103 data."""
from __future__ import annotations

import csv
import gzip
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_DATA_ROOT = Path("data/wikitext103")


def _manifest_rows(root: Path) -> dict[str, dict[str, str]]:
    manifest = root / "manifest.tsv"
    if not manifest.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {manifest}")
    with manifest.open(newline="") as handle:
        return {
            row["split"]: row
            for row in csv.DictReader(handle, delimiter="\t")
        }


def materialize_token_split(
    split: str,
    *,
    root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    """Return ``<split>.bin``, inflating its checked gzip archive if needed."""
    output = root / f"{split}.bin"
    if output.is_file():
        return output

    archive = root / f"{split}.bin.gz"
    if not archive.is_file():
        raise FileNotFoundError(
            f"token split not found: expected {output} or {archive}"
        )
    row = _manifest_rows(root).get(split)
    if row is None:
        raise ValueError(f"split {split!r} is absent from dataset manifest")
    expected_bytes = int(row["bytes"])
    expected_sha256 = row["sha256"]

    root.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{split}.",
            suffix=".bin.tmp",
            dir=root,
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            digest = hashlib.sha256()
            written = 0
            with gzip.open(archive, "rb") as source:
                while chunk := source.read(1024 * 1024):
                    written += len(chunk)
                    if written > expected_bytes:
                        raise ValueError(
                            f"{archive} expands beyond {expected_bytes} bytes"
                        )
                    digest.update(chunk)
                    destination.write(chunk)
        if written != expected_bytes:
            raise ValueError(
                f"{archive} expanded to {written} bytes; "
                f"expected {expected_bytes}"
            )
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{archive} SHA-256 {actual_sha256} != "
                f"{expected_sha256}"
            )
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output


def token_memmap(
    split: str,
    *,
    root: Path = DEFAULT_DATA_ROOT,
) -> np.memmap:
    path = materialize_token_split(split, root=root)
    row = _manifest_rows(root).get(split)
    if row is None:
        raise ValueError(f"split {split!r} is absent from dataset manifest")
    dtype_name = row.get("dtype") or "uint16"
    dtypes = {
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
    }
    if dtype_name not in dtypes:
        raise ValueError(
            f"unsupported token dtype {dtype_name!r} for split {split!r}"
        )
    values = np.memmap(path, dtype=dtypes[dtype_name], mode="r")
    expected_tokens = int(row["tokens"])
    if values.size != expected_tokens:
        raise ValueError(
            f"{path} contains {values.size} {dtype_name} values; "
            f"manifest declares {expected_tokens}"
        )
    return values
