# Post-hoc H2 feedback-mechanism stratification

## Status and controls

- State: registered before this diagnostic was executed
- Scope: exploratory mechanism diagnostic nested under the completed
  continuous-vs-discrete feedback ablation
- Checkpoint: the same latent epoch-5 `last.pt`
- Seed: 1337
- Split: all 965 WikiText-2 validation chunks
- Prefix and target: the same literal 240-token prefix and held-out suffix
- Precision: strict float32 with TF32 disabled

## Question

Does the H2 disadvantage of hard argmax-token re-encoding already remain when
the first selected token is correct, or is it concentrated among trajectories
whose first hard selection is wrong?

## Comparison

Partition examples by whether the shared H1 argmax equals the first held-out
gold token. Within each partition report:

- group size and fraction
- H2 teacher, continuous-state, and discrete-re-encode NLL and accuracy
- `discrete H2 NLL - continuous H2 NLL`
- relative state displacement between the predicted first continuous state
  and the encoder state obtained after appending its argmax token

The partition key uses the target only after both paths have produced their
shared H1 logits. No target enters either open feedback path.

## Interpretation criteria

- If discrete re-encoding is no worse than continuous feedback in the
  H1-correct group but substantially worse in the H1-wrong group, the observed
  aggregate gap is dominated by hard error commitment/off-gold-prefix
  exposure rather than an unconditional penalty for token re-entry.
- If discrete re-encoding remains worse in the H1-correct group, a feedback
  state-domain mismatch remains even when argmax selected the gold token.
- This diagnostic cannot by itself separate information destroyed by hard
  selection from lack of training on generated wrong-token prefixes; that
  causal separation would require a matched training intervention.
- Report both groups and retain conflicting evidence.

