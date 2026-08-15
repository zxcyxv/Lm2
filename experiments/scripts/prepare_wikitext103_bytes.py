"""Prepare deterministic UTF-8 byte streams for the closure experiment."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from datasets import __version__ as datasets_version
from datasets import load_dataset


ROOT = Path("data/wikitext103_bytes")
CACHE = Path("outputs/data-cache/wikitext103_bytes")
SOURCE = "Salesforce/wikitext"
CONFIG = "wikitext-103-raw-v1"
TRAIN_LIMIT = 55_000_000


def encode_split(split, path: Path, limit: int | None) -> tuple[int, str]:
    digest = hashlib.sha256()
    written = 0
    with path.open("wb") as handle:
        for text in split["text"]:
            if not text:
                continue
            values = (text + "\n").encode("utf-8")
            if limit is not None:
                values = values[: max(0, limit - written)]
            if values:
                handle.write(values)
                digest.update(values)
                written += len(values)
            if limit is not None and written >= limit:
                break
    return written, digest.hexdigest()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(SOURCE, CONFIG, cache_dir=str(CACHE))
    records = []
    for split_name, limit in (("train", TRAIN_LIMIT), ("validation", None)):
        path = ROOT / f"{split_name}.bin"
        count, checksum = encode_split(dataset[split_name], path, limit)
        records.append(
            (
                split_name,
                str(path),
                count,
                path.stat().st_size,
                checksum,
                "uint8",
            )
        )
        print(records[-1], flush=True)

    with (ROOT / "manifest.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ("split", "path", "tokens", "bytes", "sha256", "dtype")
        )
        writer.writerows(records)
    with (ROOT / "provenance.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerow(("source", SOURCE))
        writer.writerow(("configuration", CONFIG))
        writer.writerow(("datasets_version", datasets_version))
        writer.writerow(("encoding", "literal_utf8_bytes_0_through_255"))
        writer.writerow(("text_join", "skip_empty_then_append_newline"))
        writer.writerow(("train_byte_limit", TRAIN_LIMIT))
        writer.writerow(("test_split", "not_materialized"))
        for split_name in ("train", "validation"):
            writer.writerow(
                (
                    f"{split_name}_dataset_fingerprint",
                    dataset[split_name]._fingerprint,
                )
            )


if __name__ == "__main__":
    main()
