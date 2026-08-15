# Interpretation

## Successful shallow predictions

Among 32 H1--H4 predictions, 17 were exact: nine spaces and eight non-space
bytes. The non-space successes were `d`, `g`, `n`, `h`, `e`, `t`, `r`, and
`a`. Their mean top-1 probability was `0.519`, so the subset includes both
high-confidence local continuations and low-confidence correct choices.

The unstratified successful-case summary initially looks favorable:

| Metric | Correct non-space (8) | All incorrect (15) |
|---|---:|---:|
| negative/positive projection | 0.0065 | 0.0145 |
| effective channel-write terms | 42.8 | 63.9 |
| dominant single-term share | 5.90% | 4.20% |
| current-write share | 65.8% | 43.2% |
| coefficient phase coherence | 0.354 | 0.342 |

However, this aggregate contrast is largely explained by the fact that correct
non-space predictions are concentrated at H1 and H2. Within each horizon the
sample is small and the pattern is mixed:

- H1 correct predictions have less cancellation and higher phase coherence,
  but H1 contains only one write and therefore cannot demonstrate selective
  retrieval across reasoning steps.
- H2 correct predictions have lower cancellation (`0.0032` versus `0.0061`)
  and fewer effective terms (`45.6` versus `48.5`), but essentially the same
  50/50 current/previous-write balance and no coherence advantage.
- H3 correct predictions use fewer effective terms and a slightly larger
  dominant term, but have lower phase coherence than errors. All three correct
  H3 bytes are spaces.
- H4 has only three correct predictions and only one non-space success (`a`);
  it shows no consistent selectivity advantage.

## Mechanistic implication

The best evidence of the desired mechanism is modest: early correct predictions
tend to avoid unnecessary destructive cancellation. It is not evidence of a
stationary-phase selector. In particular, the correct H2 bytes `g`, `n`, `e`,
and `r` still combine the two writes almost equally rather than sharply
reinforcing one relevant age.

Thus the shallow competence currently reached by the model is better described
as **locally coherent additive continuation**:

1. the encoder root already identifies a plausible next-byte basin;
2. one or two recurrent writes reinforce that local continuation with little
   cancellation;
3. after H2, phase dispersion and broad channel participation grow faster than
   useful age selection develops.

The desirable endpoint is therefore not simply “more destructive interference.”
It is **conditional selectivity**: correct deep predictions should preserve the
low-cancellation coherent subspace seen in successful H1/H2 cases while using
negative interference specifically to remove competing writes. The current
H8/H16 dynamics acquire cancellation without acquiring that selectivity.

This audit is a mechanism case study over eight contexts, not a statistically
powered estimate. Every token-level value is retained in `tokens.tsv`.

