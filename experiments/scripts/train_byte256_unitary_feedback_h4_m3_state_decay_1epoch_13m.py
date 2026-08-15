"""Mamba-3 style state-dependent decay/rotation on the memory transport.

The registered memory transport is ``S_j = exp(-i*theta) S_(j-1) + k v^dagger``
with a unit-modulus multiplier: a lossless integrator whose every past write is
retained at full amplitude forever.  This ablation replaces that multiplier
with the Mamba-3 SISO parameterization, evaluated on the recurrence axis:

    A     = -heavy_tail(W_A z)  floored at -A_floor          (< 0)
    DT    = softplus(W_dt z + dt_bias)                       (> 0)
    decay = exp(A * DT)                                      (in (0,1))
    dphi  = tanh(W_ang z) * DT * pi                          (signed)
    mult  = decay * exp(-i * (theta + dphi))

Both the decay rate and the rotation increment are predicted from the same
normalized row that feeds Q/K/V and share one step size ``DT``, so a single
timestep governs decay and phase advance together (exponential-Euler).  On this
axis the "input" is the model's own hidden carrier, so input-dependence becomes
state-dependence -- consistent with the surrounding self-predicted-KV design.

Two deliberate departures from `20260813_new/LA/mamba3_ref.py` are recorded:

1. That reference computes ``-heavy_tail(dd_A).clamp(max=-A_floor)``.  Method
   call binds tighter than unary minus, so it evaluates as
   ``-(heavy_tail(...).clamp(max=-A_floor))``; since ``heavy_tail`` is strictly
   positive the clamp collapses every input to ``-A_floor`` and the negation
   turns it into the constant ``+A_floor``.  That destroys state dependence and
   inverts the sign into growth.  This implementation negates first, then
   floors.
2. The reference's ``dt`` range ``[1e-3, 1e-1]`` is calibrated for token
   sequences thousands of steps long; on it the mean effective memory horizon
   is ~56 steps, far beyond this recurrence's H4 training / H16 monitoring
   depth, which would make the intervention nearly inert here.  The range is
   therefore rescaled to ``[5e-2, 5e-1]``, giving ~5.6 steps -- matched to the
   rollout depth actually being trained and measured.

Keeping the registered constant ``theta`` as the phase baseline makes the
change strictly additive: a zero predicted increment recovers the parent
rotation exactly, so the only genuinely new behaviour is the decay.
"""
from __future__ import annotations

from pathlib import Path

import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch as run


base = run.base
h4 = run.h4
branch_normalized = run.run

EXPERIMENT_ID = (
    "EXP-20260814-byte256-unitary-feedback-h4-m3-state-decay-1epoch-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID

EPOCH_STEPS = run.EPOCH_STEPS
SCHEDULE_STEPS = run.TOTAL_STEPS
# memory_gate [1344, 2*8 + 8*16] = 193,536 plus the 8 dt_bias entries.
EXPECTED_PARAMETERS = 13_215_008 + 193_544
# See the replication script: the two-GEMM formula audit's fixed 2e-6 absolute
# envelope is below float32 GEMM reassociation noise on this GPU.
FORMULA_TOLERANCE_BASE = 1e-5

for module in (run, run.run, run.run.parent, run.run.h16, h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

branch_normalized.FORMULA_TOLERANCE_BASE = FORMULA_TOLERANCE_BASE
base.STATE_DEPENDENT_MEMORY_DECAY = True
base.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
base.STEPS = EPOCH_STEPS
base.SCHEDULE_STEPS = SCHEDULE_STEPS
base.OBJECTIVE_NAME_OVERRIDE = (
    "four_step_unitary_feedback_branch_rms_m3_state_decay_frobenius_"
    "residual_ce_only_attached_stride4_h16_monitor"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "Ablation of the registered branch-normalized feedback recurrence in "
    "which the unit-modulus memory multiplier is replaced by a Mamba-3 style "
    "state-dependent contraction: decay rate and signed rotation increment "
    "are predicted from the same normalized row that feeds Q/K/V and share "
    "one step size. The hidden carrier, the read's Frobenius denominator, and "
    "the H1--H4 attached CE objective are unchanged."
)


if __name__ == "__main__":
    base.main()
