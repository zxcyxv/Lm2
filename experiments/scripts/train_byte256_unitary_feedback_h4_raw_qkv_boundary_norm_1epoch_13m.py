"""Ablation 2: move the fixed RMS from the QKV branch to the hidden boundary.

The registered recurrence normalizes the hidden row only where it feeds Q/K/V
and leaves the recurrent carrier raw.  This ablation inverts that placement:
Q/K/V read the raw rotated hidden state, while the state that actually crosses
the recurrent boundary is normalized -- both the initial encoder root and the
successor that is re-input at the next horizon.  Memory stays raw and the
measurement keeps its Frobenius denominator, so only the hidden-carrier
normalization placement changes.

The LR schedule remains the parent's 33,570-step cosine so that the first epoch
is step-for-step comparable with the preserved parent metrics.
"""
from __future__ import annotations

from pathlib import Path
import sys

import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch as run


base = run.base
h4 = run.h4

EXPERIMENT_ID = (
    "EXP-20260814-byte256-unitary-feedback-h4-raw-qkv-boundary-normalized-"
    "hidden-1epoch-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID

EPOCH_STEPS = run.EPOCH_STEPS
SCHEDULE_STEPS = run.TOTAL_STEPS
RECURRENT_HIDDEN_MODE = "unitary-raw-qkv-boundary-normalized-hidden-residual"
# No projection changes width, so the registered parent total is unchanged.
EXPECTED_PARAMETERS = 13_215_008

# The parent's preflight is specific to the branch-normalized formula, which
# this ablation deliberately changes.  Fall back to the mode-agnostic CE-only
# preflight, which validates the recurrence against the module constants set
# below instead of against one hard-coded recurrence.
MODE_AGNOSTIC_PREFLIGHT = sys.modules[
    "train_byte256_complex_self_predicted_kv_post_update_full_read_h4_ce_only_13m"
].preflight

for module in (run, run.run, run.run.parent, run.run.h16, h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

run.ORIGINAL_PREFLIGHT = MODE_AGNOSTIC_PREFLIGHT
base.RECURRENT_HIDDEN_MODE = RECURRENT_HIDDEN_MODE
base.PRE_NORMALIZE_QKV_INPUTS = False
base.NORMALIZE_INITIAL_RECURRENT_ROOT = True
base.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
base.STEPS = EPOCH_STEPS
base.SCHEDULE_STEPS = SCHEDULE_STEPS
base.OBJECTIVE_NAME_OVERRIDE = (
    "four_step_unitary_feedback_raw_qkv_boundary_normalized_hidden_"
    "frobenius_residual_ce_only_attached_stride4_h16_monitor"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "Ablation of the registered branch-normalized feedback recurrence in "
    "which the fixed non-affine RMS is removed from the Q/K/V branch and "
    "applied instead to the recurrent hidden carrier: the initial encoder "
    "root and every successor re-entering the next horizon are normalized. "
    "Memory stays raw and the read keeps its Frobenius denominator."
)


if __name__ == "__main__":
    base.main()
