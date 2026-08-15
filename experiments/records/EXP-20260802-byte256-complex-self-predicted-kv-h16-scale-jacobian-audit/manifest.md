# EXP-20260802 H16 scale and Jacobian audit

## Status

- State: completed on the first two registered validation starts
- Test split remains unmaterialized and unread.
- This is a read-only checkpoint audit. It does not update model parameters.

## Question

Which quantity distinguishes the stable intermediate-normalization H16 run
from the divergent unnormalized and final-carrier-only runs: recurrent-carrier
RMS, an unnormalized internal consumer, decoder-observation scale, aligned
innovation accumulation, or the tangent gain of the complete tied transition?

The audit is specifically intended to decide whether the next intervention
must control:

1. the complex memory before the post-update full read;
2. the decoder-facing observation without changing the latent recurrence;
3. the innovation-to-transport ratio across sixteen tied visits;
4. projection initialization near the RMSNorm epsilon regime; or
5. some combination of the above.

## Fixed comparisons

The following existing producers and checkpoints are compared without
relabeling or replacing their records:

- unnormalized H16: step 100 and step 300;
- intermediate P/memory/Z post-normalization H16: steps 100, 500, and 1000;
- final-carrier-only post-normalization H16: steps 100 and 300;
- seed-matched initialization for each distinct architecture when needed to
  establish initialization scale.

Checkpoint files remain under ignored `outputs/experiments/` directories.

## Fixed data and seed

- WikiText-103 raw UTF-8 validation bytes only
- validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`
- model seed 1337
- fixed validation starts are the first registered rows from the existing H16
  records, generated with seed `1337 + 999`
- context 256, horizon 16, anchor stride 16
- strict float32; TF32 disabled
- no train or test examples are read

## Registered measurements

For identical validation roots and every horizon, record separately:

- encoded current-state and literal next-state RMS;
- rotated-memory RMS;
- raw preliminary observation `P` RMS;
- decoded token-space hidden RMS and logit RMS;
- innovation RMS and its ratio/cosine with the rotated memory;
- raw updated-memory RMS and the stored updated-memory RMS;
- raw post-update full-read hidden RMS and stored recurrent-hidden RMS;
- Q/K/V/readout matrix Frobenius and spectral norms;
- empirical tangent gains for perturbations initially restricted to hidden and
  memory state fields;
- a power-iteration estimate of the largest singular value of one complete
  recurrent transition and of its sixteen-step composition, subject to the
  available memory budget.

RMSNorm denominators and their implied tangent multiplier `1 / rms` are
recorded before each normalized consumer. Complex memory uses its real
Hilbert-space norm; real and imaginary perturbations are both included.

## Discriminating predictions

- A carrier-norm-only explanation requires divergent checkpoints to have a
  larger stored `S` or `Z` RMS than the stable control.
- A decoder-scale-gauge explanation predicts that final-carrier-only step 300
  has sharply larger raw `P`, decoded-hidden, or logit RMS even while stored
  `S` and `Z` stay normalized.
- An internal-consumer explanation predicts excessive raw updated-memory or
  full-read scale before the boundary norm, absent when memory is normalized
  before the full read.
- A tied-write accumulation explanation predicts positive cross-horizon
  alignment and a cumulative write magnitude closer to linear than
  square-root growth.
- A Jacobian explanation requires the divergent path to show materially larger
  one-step or composed tangent gain on the same validation roots.

No single prediction is treated as exclusive. Conflicting directions are
retained with checkpoint and horizon scope.

## Completion criteria

The audit is complete only if all registered checkpoints load with their
producer configuration, all reported values are finite or explicitly marked
as overflow, identical validation token prefixes and anchor indices are used
across comparisons, and enough
quantities are measured to select a minimal next ablation without inferring
causality from global gradient norm alone.

Machine-readable measurements go to TSV. Interpretation and literature-based
inferences go to Markdown.

## Producer

```bash
python eval_byte256_complex_self_predicted_kv_h16_scale_jacobian.py \
  --examples 2 --power-iterations 6
```
