# Actual-Adam-delta same-batch line search

## Status

- State: completed; near-zero finite-difference convergence criterion not
  satisfied
- Parent evidence: `update_direction_manifest.md`
- Test split remains unmaterialized and unread.

## Question

Along the already reproduced actual update direction
`delta = theta_1001 - theta_1000`, the current directional derivative is
negative although the full alpha-1 update raises same-batch CE. At what alpha
does CE reach its local minimum, reverse from decreasing to increasing, and
cross back above its alpha-0 value?

## Fixed state and evaluation

- Load the ignored exact pre/post update-1001 model states produced by the
  registered group-swap audit and the ignored exact saved update-1001 batch.
- Evaluate only interpolated weights `theta(alpha) = theta_1000 + alpha*delta`.
- Do not call backward, an optimizer, or any training update.
- Reuse the existing same-batch evaluation helper and common source objective.
- Split: checkpointed train batch, batch 64, microbatch 16, context 256, H16,
  stride 16. Seed and sampled rows are fixed by the saved batch.

## Registered alpha search

1. Coarse values include alpha 0 and 1, symmetric near-zero values
   `+/-{1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3}`, positive log/dense values
   through 1, and `-1e-2` as a negative-direction sanity check.
2. Bracket the lowest positive coarse CE by its two adjacent coarse alphas and
   evaluate 65 equally spaced values inside that bracket.
3. Around the lowest value after step 2, evaluate another 33 equally spaced
   values between its adjacent evaluated alphas.
4. After the minimum, locate the first bracket where CE minus alpha-0 CE
   changes from negative to nonnegative and perform 20 bisection evaluations.
5. Estimate the derivative at zero using forward and central differences at
   `1e-6`, `3e-6`, and `1e-5`; compare it with the independently registered
   `g dot delta` rather than replacing that autograd derivative.

For every unique alpha, record total and H1--H16 CE plus H1/H16 P, innovation-W,
and logit RMS/max scales. The minimum and baseline crossing are empirical
same-line estimates, not global guarantees.

## Outputs and success criteria

- Metrics: `update_line_search.tsv`
- Derivative/refinement summary: `update_line_search_summary.tsv`
- Interpretation: `update_line_search_interpretation.md`

The audit succeeds only if alpha 0 and 1 reproduce the registered pre/post CE,
the near-zero finite difference has the same negative sign as `g dot delta`,
and all requested horizon/scale fields are finite. If no baseline crossing is
found by alpha 1, that absence is retained instead of extrapolated.

## Evidence boundary

This is a deterministic same-batch slice through one actual Adam delta. It
does not establish validation behavior, a safe learning rate, a Hessian model,
or stability over subsequent updates.

The completed curve is exactly repeatable both with fixed loaded weights and
with weights reapplied before every forward, but it is locally jagged because
very small scaled deltas are heavily quantized in float32. Consequently the
record reports a lowest sampled point and a baseline sign-change region, not a
smooth stationary point or a high-precision root.

The alpha-0/alpha-1 reproduction and finite metric criteria passed. The
registered near-zero finite-difference criterion did not: central differences
at epsilon `1e-6`, `3e-6`, and `1e-5` were respectively `+72.9981`,
`-15.4432`, and `-10.6709`, versus autograd `g dot delta = -7.15459`.
The smallest epsilon has the wrong sign and the sequence does not converge
cleanly. This failure is retained as part of the completed audit.

## Producer

```bash
python eval_byte256_complex_self_predicted_kv_no_qkv_prenorm_step1000_update_line_search.py
```
