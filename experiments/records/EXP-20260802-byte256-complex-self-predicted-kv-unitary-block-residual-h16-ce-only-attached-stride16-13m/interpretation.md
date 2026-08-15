# Interrupted result

The run was deliberately stopped after the registered step-50 report. Raw
global gradient norm was `322.899536` at step 1 and `60.648613` at step 50;
block validation NLL was `19.582928` and `4.245232`, respectively. The
step-50 gradient exceeded the registered ceiling of 10 and was far outside
the desired approximately 2-scale stable regime, so continuing to step 300
was not informative for the stated stability question.

This rejects the tested combination of single-write/single-measurement block
closure, ordinary output initialization, fixed residual scale 0.1, and
sqrt(count) memory coordinates. It does not isolate which member of that
combination caused the failure.
