# Epoch-2 H1/H4 feedback gold-token-one H16 NLL

## Status

- State: superseded before execution by user clarification; no result metrics
  were produced
- Fixed validation records only; test unread
- No checkpoint is created or mutated by this audit

The intended protocol re-enters greedy model predictions and restores gold
context only at sixteen-token boundaries. This preregistration instead used
gold-token teacher forcing every step, so it is preserved but will not be
executed for the requested comparison. The corrected producer is
`eval_byte256_unitary_feedback_epoch2_h1_h4_greedy_reinput_h16_rollout_ce.py`.

## Question and comparisons

Does supervising only H1, rather than H1--H4, improve the 13M feedback
model's sixteen-token behavior after the same 6,714 optimizer updates and the
same 16,384 labels per update?

The primary requested natural chunk-size comparison is:

1. H1-loss epoch-2 checkpoint with canonical gold re-entry every token;
2. H4-loss epoch-2 checkpoint with canonical gold re-entry every four tokens.

This compares the intended chunking behavior of the two objectives but
changes conditioning frequency. The secondary conditioning-matched comparison
re-evaluates both H1-loss and H4-loss checkpoints with re-entry every token.

The audit also ingests, without recomputation, the epoch-2 open-H16 horizon
metrics for both feedback checkpoints and the completed parallel-scan run.
Open-H16 is retained only as an out-of-training-contract structural diagnostic
for H1 loss. It must not be used to rank the token-reinput model.

The already preserved H4-loss natural block-four re-entry NLL is `2.137559340`
in
`EXP-20260805-byte256-unitary-feedback-h4loss-epoch2-block4-reinput-h16-nll`.
It remains a separate conditioning protocol, not the H4 row used for the
primary token-one comparison. The audit will nevertheless ingest its
per-horizon record so the requested natural chunk-size comparison (H1-loss
with token-one re-entry versus H4-loss with block-four re-entry) is explicit
and labeled as confounded by conditioning frequency.

## Fixed protocol

- checkpoints:
  - `outputs/experiments/EXP-20260805-byte256-unitary-feedback-h1-stride1-h16-monitor-2epoch-13m/step6714.pt`
  - `outputs/experiments/EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-10epoch-13m/step6714.pt`
- both checkpoints must report step 6,714, their registered H1/H4 experiment
  IDs and objectives, the matched batch/schedule config, and exactly
  13,215,008 parameters
- seed 1337; the byte-identical fixed 64 validation starts shared by H1,
  H4, and parallel scan
- all sixteen stride-16 anchors in each 256-byte context
- 1,024 labels per absolute horizon and 16,384 labels per reported mode
- for every absolute H1--H16 target, append the preceding gold bytes,
  canonically re-encode the complete prefix, initialize central memory from
  that prefix root, and run exactly one state-dependent transition
- strict float32 with TF32 disabled; online checkpoint weights only
- ingest the H1-loss and H4-loss open-H16 rows and the parallel-scan
  step-6,714 row from their authoritative `metrics.tsv` records
- ingest the preserved H4-loss gold-block-four horizon record, without
  recomputing or relabeling it as token-one conditioning
- scalar and per-horizon metrics in TSV; interpretation in Markdown

Gold token-one re-entry is a proper teacher-forced likelihood. It neither
feeds generated tokens nor preserves an earlier open-loop central memory
across the canonical re-encoding boundary. The parallel-scan reference is
open-H16, so differences against token-one rows are requested numerical
context rather than conditioning-matched architecture effects.

## Success and evidence gates

- both checkpoints load at the registered step with the registered parameter
  count
- H1, H4, and scan validation-start files are byte-identical
- all 16,384 labels per mode and all metrics are finite
- each token-one model's absolute H1 NLL matches its recorded open-H1 NLL
  within `2e-3`; variable-length canonical re-encoding can differ slightly
  from the preserved full causal tape because attention association shapes
  differ in strict float32
- retain and report the result whether H1 loss helps or hurts open H16 or
  token-one H16
- never read the test split

## Producer

Run only after the H1-loss step-6,714 checkpoint and final metrics row exist:

```bash
python eval_byte256_unitary_feedback_epoch2_h1_h4_token1_reinput_h16_nll.py \
  --microbatch 8
```
