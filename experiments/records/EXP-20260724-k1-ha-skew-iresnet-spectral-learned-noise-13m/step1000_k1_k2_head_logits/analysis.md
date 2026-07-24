# Step-1000 causal K / K^2 head-logit audit

- Checkpoint step: 1000
- Fixed validation prefixes: 256
- h1 accuracy (short tape): 0.226562
- h1 accuracy (joint K,K^2 tape): 0.226562
- h2 accuracy (joint tape): 0.042969
- h1 short-vs-joint hidden max-abs (global max): 0.000000e+00
- h1 short-vs-joint logits max-abs (global max): 0.000000e+00
- h1-vs-h2 head-input cosine (mean): 0.828321
- h1-vs-h2 logits cosine (mean): 0.947289
- h1-vs-h2 centered-logits cosine (mean): 0.912880
- h1-vs-h2 probability cosine / JS (mean): 0.771455 / 0.134303
- h1-vs-h2 top-5 overlap (mean fraction): 0.566406
- h1/h2 same top-1 token rate: 0.500000
- h1 target-rank median: 9.0
- h2 target-rank median: 134.0
- exact-gold hB/hC head accuracy: 1.000000 / 1.000000

Full distributions are in `summary.tsv`; token examples are in `examples.md`.