"""Scale the prefix-conditioned orthogonal affine scan (parallel-scan
architecture) to ~121M parameters, matching the byte256 state-dependent
121M comparison in scale.

Only two things change relative to the registered 13M producer
(``train_k4_ha_skew_prefix_conditioned_orthogonal_affine_scan_ce_closure_13m.py``,
``EXP-20260726-...``): ``WIDTH`` goes from 896 to 3600 (121,403,425
parameters, matching the byte256 state-dependent 121M run's 121,336,064
within 0.06%), and peak LR is cut to 1/3 (the scale-up heuristic requested
for this run) -- everything else (architecture, horizons, anchor stride,
batch schedule, seed, data) is identical.
"""
from __future__ import annotations

from pathlib import Path

import train_k1_decoder_inverse_ablation_13m as lr_origin
import train_k4_ha_skew_prefix_conditioned_orthogonal_affine_scan_ce_closure_13m as base


EXPERIMENT_ID = (
    "EXP-20260815-k4-ha-skew-prefix-conditioned-orthogonal-affine-scan-"
    "ce-closure-121m"
)

base.WIDTH = 3600
# lr_at() (imported by base, not redefined there) closes over its
# *defining* module's globals, so PEAK_LR must be overridden on
# lr_origin too, or the schedule silently keeps using 3e-4.
lr_origin.PEAK_LR = lr_origin.PEAK_LR / 3.0
base.PEAK_LR = lr_origin.PEAK_LR
base.EXPERIMENT_ID = EXPERIMENT_ID
base.DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
base.DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID


if __name__ == "__main__":
    base.main()
