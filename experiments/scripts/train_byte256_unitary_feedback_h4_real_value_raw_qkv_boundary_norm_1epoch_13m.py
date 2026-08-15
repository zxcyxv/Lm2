"""Ablation 3: real write value and boundary-normalized hidden carrier.

This is the conjunction of the two single-factor ablations:

* the rank-one write value ``v`` is real rather than complex
  (`train_byte256_unitary_feedback_h4_real_value_1epoch_13m`), and
* the fixed RMS moves off the Q/K/V branch onto the recurrent hidden boundary
  (`train_byte256_unitary_feedback_h4_raw_qkv_boundary_norm_1epoch_13m`).

Running it alongside both single-factor runs separates additive effects from
interaction.  The LR schedule remains the parent's 33,570-step cosine so that
the first epoch is step-for-step comparable with the preserved parent metrics.
"""
from __future__ import annotations

from pathlib import Path
import sys

import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch as run


base = run.base
h4 = run.h4

EXPERIMENT_ID = (
    "EXP-20260814-byte256-unitary-feedback-h4-real-value-raw-qkv-boundary-"
    "normalized-hidden-1epoch-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID

EPOCH_STEPS = run.EPOCH_STEPS
SCHEDULE_STEPS = run.TOTAL_STEPS
RECURRENT_HIDDEN_MODE = "unitary-raw-qkv-boundary-normalized-hidden-residual"
# Only the real-value factor changes the parameter total: 8 * 31 * 1344.
EXPECTED_PARAMETERS = 13_215_008 - 333_312

MODE_AGNOSTIC_PREFLIGHT = sys.modules[
    "train_byte256_complex_self_predicted_kv_post_update_full_read_h4_ce_only_13m"
].preflight

for module in (run, run.run, run.run.parent, run.run.h16, h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

run.ORIGINAL_PREFLIGHT = MODE_AGNOSTIC_PREFLIGHT
base.REAL_VALUE = True
base.RECURRENT_HIDDEN_MODE = RECURRENT_HIDDEN_MODE
base.PRE_NORMALIZE_QKV_INPUTS = False
base.NORMALIZE_INITIAL_RECURRENT_ROOT = True
base.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
base.STEPS = EPOCH_STEPS
base.SCHEDULE_STEPS = SCHEDULE_STEPS
base.OBJECTIVE_NAME_OVERRIDE = (
    "four_step_unitary_feedback_real_value_raw_qkv_boundary_normalized_"
    "hidden_frobenius_residual_ce_only_attached_stride4_h16_monitor"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "Joint ablation: the rank-one write value is real, and the fixed "
    "non-affine RMS is removed from the Q/K/V branch and applied instead to "
    "the initial recurrent root and every successor hidden state re-entering "
    "the next horizon. Memory stays raw with a Frobenius-normalized read."
)


if __name__ == "__main__":
    base.main()
