# EXP-20260801 post-update residual/Shapley audit

## Status

- State: completed; all 16,384 registered rows audited
- Parent:
  `../EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-h1-ema-sg-mse-ce-13m/`
- Read-only post-hoc audit of the parent's step-1000 full-model EMA.
- No optimizer update, checkpoint mutation, or test-split access.

### Result

- All additive, head-sum, Shapley-sum, replay, and decision-margin structural
  checks passed.
- H1 innovation reduced canonical-AR hidden error on `99.73%` of rows.
- H2 innovation was nearly neutral on average but slightly harmful; H3 and H4
  innovation were increasingly harmful and had negative global optimal scales.
- Every head had negative mean H2--H4 Shapley error reduction.
- At the same weights, the registered full-read recurrence achieved `0.242269`
  H2--H4 agreement versus `0.193604` for the old recurrent formula. The new
  formula itself is therefore helpful but insufficient.

## Primary question

Why did the coherent post-update full reread finish at H2--H4 agreement
`0.242269`, below the previous split-query control's `0.272135`?

The target for this audit is never the held-out continuation.  At every
horizon it is the canonical hidden obtained by re-encoding the same model's
own greedy-AR history.  This matches the research goal of central recurrence
equivalence to greedy AR.

## Exact additive decomposition

For each central step, use the newly computed query on both memory terms:

```text
A       = O(Read(Q(P), Mrot))
B[h]    = head-h contribution of O(Read(Q(P), W))
B       = sum_h B[h]
Znext   = A + B
target  = canonical hidden from the model's greedy-AR history
resid   = target - A
```

The audit verifies `Znext == A + sum_h B[h]` numerically. It then records:

- prior and full cosine/relative MSE to the canonical AR hidden;
- exact error reduction from adding the innovation;
- cosine and norm ratio between total innovation `B` and prior residual;
- the least-squares scalar on `B` that would minimize hidden error, plus its
  oracle MSE and the regret at the fixed coefficient one;
- the exact quadratic-error Shapley allocation per head:
  `2 dot(resid,B[h]) - dot(B[h],B)`, normalized by target energy.

The head allocations sum exactly to the total innovation error reduction, so
they distinguish useful from harmful heads without depending on insertion
order.

### Reporting addendum

After the first structural pass and before final interpretation, the report
was extended with one aggregate least-squares scalar per horizon/group in
addition to the already registered per-row oracle scalar. This is derived
only from the same fixed residual rows and performs no parameter fitting or
checkpoint update. It tests whether one implementable global coefficient
could repair a failure that a separate per-row oracle may overstate.

## Decision-boundary stratification

For every one of the 16,384 fixed rows (64 examples, 64 anchors, 4 horizons),
split measurements by whether central and greedy-AR top-1 tokens agree. Also
record the AR decision margin and exact central-versus-AR logit perturbation.

Interpretation:

- positive innovation error reduction with an optimum scale near one supports
  the write direction and magnitude;
- positive reduction but a far-from-one optimum scale and large oracle gain
  indicates magnitude calibration;
- low or negative residual cosine with little oracle gain indicates that the
  innovation points in the wrong hidden direction;
- similar hidden errors in match/mismatch rows but much smaller AR margins in
  mismatches indicates a decoder-boundary bottleneck;
- much larger hidden errors in mismatches indicates a recurrence bottleneck.

## Same-weight counterfactual

Without changing any parameter, replace only the recurrent-hidden formula by
the previous split-query residual formula and recompute greedy central/AR
agreement. H1 CE and the AR reference remain unchanged; only the target-free
central composition changes after H1. This isolates whether the coherent full
reread itself is useful at the learned step-1000 weights.

## Fixed evidence

- parent EMA checkpoint: step 1000
- WikiText-103 raw-byte validation split only
- parent's stored 64 starts, validation seed `1337 + 999`
- context 256; horizons 1--4; anchor stride 4; microbatch 2
- strict float32; TF32 disabled
- parent train SHA-256:
  `062a4d5419f3865f66add500c323f03f1223b86f809d47a787bdbc60b54759dc`
- parent validation SHA-256:
  `7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0`

## Completion checks

- EMA checkpoint configuration is coefficient-free post-update full-read.
- all 16,384 rows are finite;
- additive reconstruction max error is below `1e-5`;
- head Shapley sum max error is below `1e-5`;
- decision-margin flip identity has zero non-tied classification errors;
- metrics are TSV and interpretation is Markdown.

## Producer

```bash
python eval_byte256_complex_self_predicted_kv_post_update_residual_shapley.py \
  --record-dir \
    experiments/records/EXP-20260801-byte256-complex-self-predicted-kv-post-update-full-read-residual-shapley-audit \
  --examples 64 --microbatch 2
```
