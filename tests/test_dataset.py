import csv
import gzip
import hashlib

import numpy as np

from rotlm.dataset import materialize_token_split, token_memmap


def test_gzip_token_split_is_verified_and_materialized(tmp_path):
    values = np.asarray([0, 1, 8191, 42, 7], dtype=np.uint16)
    raw = values.tobytes()
    with gzip.open(tmp_path / "train.bin.gz", "wb") as handle:
        handle.write(raw)
    with (tmp_path / "manifest.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("split", "path", "tokens", "bytes", "sha256"))
        writer.writerow(
            (
                "train",
                "data/wikitext103/train.bin",
                values.size,
                len(raw),
                hashlib.sha256(raw).hexdigest(),
            )
        )

    output = materialize_token_split("train", root=tmp_path)
    assert output == tmp_path / "train.bin"
    assert output.read_bytes() == raw
    np.testing.assert_array_equal(
        token_memmap("train", root=tmp_path),
        values,
    )
