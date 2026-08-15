# Epoch-2 H1/H4 natural greedy-reinput H16 block comparison

## Status

- State: completed
- Fixed validation only; test unread

The H1/H4/scan H1--H16 mean rollout CEs were respectively `7.063616096`,
`3.494413549`, and `2.912845770`, over 16,384 labels per mode.

## Question and protocol

At the same step 6,714, seed 1337, validation starts, architecture, optimizer
trajectory, and 16,384 training labels/update, compare each model at its
natural output chunk: H1-loss uses greedy one-token re-entry, H4-loss uses
greedy four-token re-entry, and the historical parallel scan uses one open
sixteen-token chunk.

At anchors 0,16,...,240, start from the gold prefix. Within the following
sixteen targets, append and canonically re-encode only greedy model outputs;
discard generated history and restore gold only at the next anchor. Score all
logits against the original gold H1--H16 bytes. No hidden or memory state is
carried across a re-encoding boundary.

This is greedy-self-conditioned gold-aligned rollout cross entropy, commonly
called block NLL here, not a teacher-forced likelihood.

## Evidence gates

- H1/H4 checkpoint step, experiment ID, objective, schedule, batch, and
  13,215,008-parameter architecture must match.
- H1/H4/scan validation starts must be byte-identical.
- Each mode has 1,024 labels/horizon and 16,384 total finite labels.
- H1 first-token and H4 first-four-token scores must match their open rows
  within `2e-3`; retain the result whether it helps or hurts.
- Metrics are TSV, interpretation Markdown, checkpoints remain ignored.

## Producer

```bash
python eval_byte256_unitary_feedback_epoch2_h1_h4_greedy_reinput_h16_rollout_ce.py
```
