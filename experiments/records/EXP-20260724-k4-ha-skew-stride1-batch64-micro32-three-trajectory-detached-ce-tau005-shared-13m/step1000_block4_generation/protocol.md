# Step-1000 AR versus block-4 generation protocol

- Checkpoint: `step1000.pt` from the parent experiment
- Split: WikiText-103 validation
- Prompt selection: 5 fixed starts from seed `1337+2024`
- Prompt length: 64 BPE tokens
- Continuation length: 64 BPE tokens
- Decoding: greedy for both methods
- AR: predict one token from `K hA`, commit it, then re-encode
- block-4: jointly decode clean `K hA` through `K^4 hA`, commit four
  tokens, then re-encode
- Comparison: decoded samples plus distinct-1, distinct-2, immediate repeat,
  within-block repeat, and block-boundary repeat

The learned noise tape and gold-dependent trajectory responsibility cannot be
used at inference, so block-4 uses the shared clean orbit. This is a
generation-quality audit, not a likelihood estimate.
