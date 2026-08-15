"""Mamba-3 style state-dependent memory decay on top of the post-norm carrier.

This is the conjunction of two axis-separated interventions:

* hidden axis -- the fixed RMS moves off the Q/K/V branch onto the recurrent
  boundary, so the carrier is renormalized every horizon
  (`train_byte256_unitary_feedback_h4_raw_qkv_boundary_norm_1epoch_13m`), and
* memory axis -- the unit-modulus memory multiplier becomes a Mamba-3 style
  state-dependent contraction
  (`train_byte256_unitary_feedback_h4_m3_state_decay_1epoch_13m`).

The registered architecture leaves both axes lossless: the hidden carrier grows
because ``R`` is orthogonal, and the memory grows because ``exp(-i*theta)`` has
unit modulus.  Running the memory intervention with and without the hidden one
separates whether bounding the two carriers is additive or whether one axis
already accounts for the effect.

The decay parameterization and the two departures from
`20260813_new/LA/mamba3_ref.py` (the ``heavy_tail`` clamp precedence, and the
``dt`` range rescaled to this recurrence's depth) are documented in the
memory-axis script and are identical here.
"""
from __future__ import annotations

from pathlib import Path
import sys

import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch as run


base = run.base
h4 = run.h4

EXPERIMENT_ID = (
    "EXP-20260814-byte256-unitary-feedback-h4-postnorm-m3-state-decay-"
    "1epoch-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID

EPOCH_STEPS = run.EPOCH_STEPS
SCHEDULE_STEPS = run.TOTAL_STEPS
RECURRENT_HIDDEN_MODE = "unitary-raw-qkv-boundary-normalized-hidden-residual"
EXPECTED_PARAMETERS = 13_215_008 + 193_544

MODE_AGNOSTIC_PREFLIGHT = sys.modules[
    "train_byte256_complex_self_predicted_kv_post_update_full_read_h4_ce_only_13m"
].preflight

for module in (run, run.run, run.run.parent, run.run.h16, h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

run.ORIGINAL_PREFLIGHT = MODE_AGNOSTIC_PREFLIGHT
base.STATE_DEPENDENT_MEMORY_DECAY = True
base.RECURRENT_HIDDEN_MODE = RECURRENT_HIDDEN_MODE
base.PRE_NORMALIZE_QKV_INPUTS = False
base.NORMALIZE_INITIAL_RECURRENT_ROOT = True
base.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
base.STEPS = EPOCH_STEPS
base.SCHEDULE_STEPS = SCHEDULE_STEPS
base.OBJECTIVE_NAME_OVERRIDE = (
    "four_step_unitary_feedback_boundary_normalized_hidden_m3_state_decay_"
    "frobenius_residual_ce_only_attached_stride4_h16_monitor"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "Joint ablation bounding both carriers: the hidden state is renormalized "
    "at every recurrent boundary, and the memory multiplier is a Mamba-3 "
    "style state-dependent contraction whose decay rate and signed rotation "
    "increment share one predicted step size."
)


if __name__ == "__main__":
    base.main()
