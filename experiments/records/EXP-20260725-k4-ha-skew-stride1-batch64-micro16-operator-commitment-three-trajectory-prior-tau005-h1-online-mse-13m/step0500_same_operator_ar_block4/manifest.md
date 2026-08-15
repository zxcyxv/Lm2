# Step-500 same-operator AR versus block-4 audit

## Status

- State: preregistered before generation
- Authorization: user-requested intermediate generation comparison
- Checkpoint step: 500
- Checkpoint SHA-256:
  `ff09df1d107ecd34b48503e4c6f531d85185da2abaf7e1c5af71b1fc36cc8d45`
- Direct control:
  [`../step0100_same_operator_ar_block4/`](../step0100_same_operator_ar_block4/)
- Test split remains unread

## Question

Between step 100 and step 500, does additional training close the discrepancy
between open-loop `T_i^j hA` block generation and re-anchored AR generation
under the same retained `T_i=Q_iK`?

## Fixed policies and data

The protocol is unchanged from the step-100 control.

- WikiText-103 validation split only
- the same 64 starts drawn with seed `1337+2024`
- 64-token prompts and 64-token greedy continuations
- one prior-argmax `Q_i` compiled and retained within each four-token block
- `same_operator_block4`: `T_i hA ... T_i^4 hA`, then re-encode
- `same_operator_ar`: re-encode each emitted token but retain and reapply the
  same `T_i` until the next block boundary
- no target-aware selection, sampling, search, rejection, or correction
- strict float32, TF32 disabled

## Success criteria

The same near-equivalence thresholds are retained without looking at the
step-500 generation:

- same-prefix one-block token agreement at least `0.80`;
- same-prefix agreement at every horizon at least `0.70`;
- full-rollout token agreement at least `0.50`;
- block-4 distinct-2 within 10% relative of AR;
- immediate-repeat and period-4-repeat differences at most `0.05`
  absolute; and
- collapsed-sample fraction difference at most `0.10` absolute.

In addition, evidence of learning progress requires same-prefix overall
agreement and each of h2--h4 to exceed the recorded step-100 values
`0.7109`, `0.5469`, `0.6406`, and `0.6562`, respectively. High free-running
agreement caused by common token collapse is not counted as local
AR-equivalence evidence.
