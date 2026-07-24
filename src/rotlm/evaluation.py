"""Small shared helpers used by the MSE+CE evaluation scripts."""
from __future__ import annotations

import numpy as np
import torch

from rotlm.dataset import DEFAULT_DATA_ROOT, token_memmap


SEED = 1337
DATA_ROOT = DEFAULT_DATA_ROOT


def memmap(split: str) -> np.memmap:
    return token_memmap(split, root=DATA_ROOT)


def windows(
    data: np.ndarray,
    starts: torch.Tensor,
    length: int,
    *,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    values = np.stack(
        [
            np.asarray(
                data[int(start) : int(start) + length],
                dtype=np.int64,
            )
            for start in starts
        ]
    )
    return torch.from_numpy(values).to(device, non_blocking=True)


def token_stats(tokens: torch.Tensor) -> dict[str, float]:
    values = tokens.tolist()
    bigrams = list(zip(values, values[1:]))
    return {
        "distinct_1": len(set(values)) / max(1, len(values)),
        "distinct_2": len(set(bigrams)) / max(1, len(bigrams)),
        "immediate_repeat": (
            sum(left == right for left, right in bigrams)
            / max(1, len(bigrams))
        ),
    }
