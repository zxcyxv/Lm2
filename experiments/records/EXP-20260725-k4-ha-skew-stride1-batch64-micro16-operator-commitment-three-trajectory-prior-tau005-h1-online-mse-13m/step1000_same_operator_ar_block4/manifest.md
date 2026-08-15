# Step-1000 same-operator AR versus block-4 audit

## Status

- State: protocol and thresholds preregistered in the parent manifest before
  training; checkpoint identity added before generation
- Authorization: user-requested training and AR/block comparison
- Checkpoint step: 1000
- Checkpoint SHA-256:
  `1a3d763607083ddac734085f28bda0d56887fc7fff1b781142734c37e6283385`
- Intermediate controls:
  [`../step0100_same_operator_ar_block4/`](../step0100_same_operator_ar_block4/)
  and
  [`../step0500_same_operator_ar_block4/`](../step0500_same_operator_ar_block4/)
- Test split remains unread

## Question

At the completed checkpoint, how closely does open-loop
`T_i hA,...,T_i^4 hA` generation match token-reanchored AR generation when
both retain the same compiled `T_i=Q_iK` within each four-token block?

## Fixed protocol

- WikiText-103 validation split only
- the same 64 starts drawn with seed `1337+2024`
- 64-token prompts and 64-token greedy continuations
- one prior-argmax branch and `Q_i` compiled per four-token block
- `same_operator_block4`: apply `T_i^1,...,T_i^4`, then re-encode
- `same_operator_ar`: re-encode after every emitted token while retaining the
  same branch, plane, angle, and `T_i` until the block boundary
- no target-aware selection, sampling, search, rejection, or correction
- strict float32, TF32 disabled

## Parent-registered architecture thresholds

- full same-operator AR/block token agreement at least `0.15`;
- block-4 period-4 repetition at most `0.4451`; and
- block-4 collapsed-sample fraction at most `0.2719`.

## Retained near-equivalence thresholds

- same-prefix one-block token agreement at least `0.80`;
- same-prefix agreement at every horizon at least `0.70`;
- full-rollout token agreement at least `0.50`;
- block-4 distinct-2 within 10% relative of AR;
- immediate-repeat and period-4-repeat differences at most `0.05`
  absolute; and
- collapsed-sample fraction difference at most `0.10` absolute.
