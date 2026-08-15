"""Scale the *correct* parallel-scan central recurrence
(``ComplexSelfPredictedKVTransition.rollout_time_varying_scan`` -- the
forcing-tape + complex-memory-prefix-scan + latent-prefix-scan architecture
specified in parallel_scan_central_recurrence.md) to 121M parameters,
matching the byte256 state-dependent 121M comparison exactly.

Architecture, encoder, loss, seed, split, and central-recurrence math are
identical to the registered 13M time-varying-scan producer
(``train_byte256_unitary_time_varying_scan_h16_1000.py`` ->
``EXP-20260803-byte256-unitary-time-varying-scan-h16-1000step-rtx5090``) --
only ``WIDTH`` (1344 -> 4352, giving 121,336,064 parameters, exactly the
registered byte256 121M state-dependent count) and peak LR (cut to 1/3)
change.
"""
from __future__ import annotations

from pathlib import Path

import train_byte256_unitary_time_varying_scan_h16_1000 as run
import train_k1_decoder_inverse_ablation_13m as lr_origin


base = run.base
EXPERIMENT_ID = (
    "EXP-20260815-byte256-unitary-time-varying-scan-h16-121m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID

base.WIDTH = 4352
base.EXPECTED_PARAMETERS = 121_336_064
# lr_at() is imported by base but *defined* in lr_origin, so it closes over
# lr_origin's globals -- overriding base.PEAK_LR alone would silently do
# nothing (this bit exactly what the earlier trash-lineage wrapper did).
lr_origin.PEAK_LR = lr_origin.PEAK_LR / 3.0
base.PEAK_LR = lr_origin.PEAK_LR
run.EXPERIMENT_ID = EXPERIMENT_ID
run.DEFAULT_RECORD = DEFAULT_RECORD
run.DEFAULT_OUTPUT = DEFAULT_OUTPUT
base.EXPERIMENT_ID = EXPERIMENT_ID
base.DEFAULT_RECORD = DEFAULT_RECORD
base.DEFAULT_OUTPUT = DEFAULT_OUTPUT


if __name__ == "__main__":
    run.main()
