# Actual-Adam-delta line search

This is the fixed native update-1001 batch and the fixed actual Adam delta. No backward pass, optimizer step, or training was performed.

- Alpha 0 CE: `3.181236438`.
- Lowest registered point: alpha about `0.006827`, CE `3.180537501` (change `-0.000698936945`).
- Alpha 1 CE: `3.195414762` (change `+0.0141783244`).
- The deterministic bisection first found a sign change around alpha `0.08711`; because the float32 path is jagged, the robust coarse statement is only that CE is below baseline at alpha `0.075` and above it at `0.1`.
- Registered autograd `g dot delta`: `-7.15458989`. Central finite differences at epsilon `1e-6`, `3e-6`, and `1e-5` were `+72.9981`, `-15.4432`, and `-10.6709`. They do not converge cleanly to the autograd value. The preregistered near-zero finite-difference criterion therefore `failed`. The closest was `-10.6709254` at epsilon `1e-05`, where the realized float32 path had alpha projection `8.78012e-06` and relative displacement error `59.634%`.

At the sampled minimum, H6 had the largest CE reduction (`-0.00237715244`). H1 P/W/logit RMS changed from `1.94283/398.075/15.8346` to `1.94173/397.478/15.8274`. H16 changed from `0.575298/20.919/6.57516` to `0.575503/20.9337/6.56897`.

Five same-loaded-weight forwards and five independent weight reapplications each had CE range at most `0`. The dense local variation is therefore deterministic float32 path quantization/sensitivity rather than forward or reapplication noise. The sampled minimum is only the lowest registered production-precision point, not a smooth stationary-point estimate, and the narrow bisection interval is a sign-change bracket, not a smooth root-confidence interval. The later baseline crossing is also distinct from the sampled minimum. This slice does not select a learning rate or establish validation/long-run behavior.
