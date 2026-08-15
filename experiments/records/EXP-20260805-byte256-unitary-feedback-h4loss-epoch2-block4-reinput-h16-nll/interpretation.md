# Epoch-2 block-4 gold-reinput H16 NLL

This is a separate gold-teacher-forcing audit, not the requested greedy
self-reinput block metric. The corrected result is in
`EXP-20260805-byte256-unitary-feedback-h1-h4loss-epoch2-greedy-reinput-h16-rollout-ce`.

Every four gold validation bytes were appended and canonically re-encoded before predicting the next four-byte block. This is a proper blockwise teacher-forced likelihood; it is not free-running greedy re-input.

| Mode | H1--H16 block NLL |
|---|---:|
| Feedback open H16 | 3.616870 |
| Feedback gold block-4 reinput | 2.137559 |
| Parallel scan open H16 | 2.912846 |

Gold block-4 reinput changed feedback NLL by -1.479310 versus open H16 and was -0.775286 relative to the epoch-matched parallel scan.

The parallel-scan H16 single-horizon NLL was 3.119667; its H1--H16 mean block NLL was 2.912846. These are different quantities.
