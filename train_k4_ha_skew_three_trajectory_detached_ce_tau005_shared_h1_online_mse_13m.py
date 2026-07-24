"""Run tau=0.05 trajectory CE with online attached h1 state MSE."""
from __future__ import annotations

from pathlib import Path

import train_k4_ha_skew_three_trajectory_detached_ce_13m as base


EXPERIMENT_ID = (
    "EXP-20260724-k4-ha-skew-stride1-batch64-three-trajectory-"
    "detached-ce-tau005-shared-h1-online-mse-13m"
)


def main() -> None:
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.TEMPERATURE = 0.05
    base.MSE_WEIGHT = 1.0
    base.MSE_HORIZONS = (1,)
    base.DETACH_MSE_TARGETS = False
    base.DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
    base.DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
    base.main()


if __name__ == "__main__":
    main()
