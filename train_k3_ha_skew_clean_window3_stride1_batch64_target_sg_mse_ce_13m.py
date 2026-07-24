"""Run stride-one K^1..K^3 with stop-gradient gold MSE targets."""
from __future__ import annotations

from pathlib import Path

import train_k3_ha_skew_clean_window3_mse_ce_13m as base


EXPERIMENT_ID = (
    "EXP-20260724-k3-ha-skew-clean-window3-stride1-batch64-"
    "target-sg-mse-ce-13m"
)


def main() -> None:
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.ANCHOR_STRIDE = 1
    base.TRUNCATE_HORIZON_GRADIENT = False
    base.DETACH_GOLD_TARGETS = True
    base.ANCHOR_COUNT = len(
        range(0, base.CONTEXT, base.ANCHOR_STRIDE)
    )
    base.DECISIONS_PER_EXAMPLE = (
        base.ANCHOR_COUNT * base.TRAIN_HORIZONS
    )
    base.DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
    base.DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
    base.main()


if __name__ == "__main__":
    main()
