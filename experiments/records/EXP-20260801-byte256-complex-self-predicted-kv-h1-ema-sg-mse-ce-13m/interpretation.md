# EMA-SG H1 interpretation

## Outcome

The run completed all 1000 updates. The preregistered all-required criterion
did not pass: mean central/AR state cosine was `0.495614`, and the step-100
trajectory did not reproduce the separately launched online-SG run within
`1e-5`.

The agreement-specific result is narrower and positive. EMA inference reached
H2--H4 central/greedy-AR token agreement `0.272135`, exceeding the registered
attached-target reference `0.261149`. It does not establish AR equivalence:
only `0.024658` of four-token blocks agreed exactly, and H4 agreement remained
`0.120605`.

## Final matched measurements

| Metric | Attached target, step 1000 | This run online, step 1000 | This run EMA, step 1000 |
|---|---:|---:|---:|
| H1 validation NLL | 1.565493 | 1.456422 | **1.393542** |
| H1 validation accuracy | 0.538330 | 0.573242 | **0.585938** |
| H2 agreement | 0.444092 | 0.475342 | **0.507812** |
| H3 agreement | 0.164795 | 0.167236 | **0.187988** |
| H4 agreement | **0.174561** | 0.106445 | 0.120605 |
| H2--H4 agreement | 0.261149 | 0.249674 | **0.272135** |
| Exact four-token block agreement | 0.012695 | 0.018555 | **0.024658** |
| H2--H4 no-write agreement | 0.073649 | 0.148763 | 0.171631 |
| Write minus no-write agreement | **0.187500** | 0.100911 | 0.100505 |
| Mean state cosine | **0.945871** | 0.488749 | 0.495614 |

EMA improved the same run's H2--H4 agreement by `0.022461` and exact-block
agreement by `0.006104`. But it raised no-write agreement by `0.022868` too.
The write-specific gain was therefore essentially unchanged (`0.100911` to
`0.100505`). The final improvement is best attributed to parameter averaging
and sharper token behavior, not to evidence that EMA specifically repaired the
self-predicted innovation recurrence.

The average gain over the attached parent is also horizon-local: H2 and H3
improved, while H4 fell by `0.053955`. The central path still accumulates a
large semantic boundary error with composition depth.

## What happened to cosine

The sub-0.9 cosine regime began with target-side stop-gradient, before EMA
decay became active:

| Run | Step 50 mean cosine | Step 100 mean cosine |
|---|---:|---:|
| Attached online target | 0.922607 | 0.933858 |
| Online target with SG | 0.650909 | 0.595454 |
| EMA-SG, still exact-copy through step 100 | 0.636459 | 0.543779 |

Thus EMA did not cause the initial drop. Removing the gradient into the future
encoder state removed the mechanism by which prediction and target coordinates
could move toward one another. The attached run's high cosine therefore mixed
transition learning with target-coordinate co-adaptation.

Cosine is not the primary semantic criterion here. A high-dimensional cosine
can remain high while small decoder-sensitive components cross a simplex token
boundary. The token agreement, exact-block agreement, and horizon profile are
the decisive evidence.

## EMA and causality checks

- The EMA target had no gradient and never entered the central recurrence.
- Primary inference used one complete EMA snapshot, so encoder and analytic
  inverse decoder were exact counterparts; the transition came from the same
  snapshot.
- At step 100, all online and EMA validation metrics inside this run were
  exactly equal because the post-update EMA decay was zero.
- The earlier online-SG run and this run diverged numerically after step 1, so
  the preregistered cross-run `1e-5` replay checks failed. That conflict is
  retained rather than treated as an EMA effect.
- H1 central/AR logit error remained `1.56e-4`, below the `5e-4` structural
  tolerance, and no beta/temperature parameter was present.

## Conclusion

Warm-started full-model EMA is useful for H1 token quality and gives a modest
average agreement improvement. It does not solve the intended AR-equivalence
problem: exact four-token agreement is only `2.47%`, H4 agreement is `12.06%`,
and the innovation's incremental agreement gain does not improve over the
online snapshot.
