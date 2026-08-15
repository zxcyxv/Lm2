# Warm-start EMA closure interpretation

## Outcome

The warm start fixed the cold-start implementation failure, but EMA produced
only a modest behavioral regularization effect. It did not reverse the
long-horizon Open/canonical divergence.

Through update 100, post-update EMA decay was zero. The validation metrics
matched the non-EMA run to numerical precision, confirming that the new run
changed only the teacher schedule after the registered boundary.

## Matched evidence

| metric | original control step 1000 | warm-start EMA step 1000 | change |
|---|---:|---:|---:|
| teacher NLL | 4.461288 | 4.523159 | +0.061871 |
| overall Open NLL | 5.959662 | 5.966937 | +0.007275 |
| H2--H4 Open NLL | 6.456272 | 6.445196 | -0.011077 |
| H2--H4 Open accuracy | 0.089844 | 0.088389 | -0.001455 |
| H2--H4 current-online top-1 agreement | 0.236715 | 0.232920 | -0.003794 |
| current-online closure KL | 1.882352 | 1.797062 | -0.085290 |
| innovation/rotation RMS ratio | 0.273504 | 0.468736 | +0.195231 |
| H4 Open-state relative MSE | 2.216210 | 5.518185 | +3.301975 |

The strict overall Open-NLL criterion missed by `0.007275`. The H2--H4 NLL
criterion and teacher-NLL cap passed. The preregistered H2--H4 agreement target
did not pass.

At step 500, the only available matched direct-Open control showed the same
trade-off:

- current-online closure: `1.480903` to `1.416095`
- H2--H4 top-1 agreement: `0.244171` to `0.253876`
- H2--H4 Open NLL: `6.612508` to `6.636425`

Thus EMA initially bought better behavioral agreement for a small token-NLL
cost. By step 1000, the closure benefit remained, but the top-1 agreement
advantage did not.

## What this rules out

The final EMA-target closure was `1.780787`, while closure to the current
online canonical teacher was `1.797062`. Their small difference (`0.016274`)
means the apparent closure improvement is not merely agreement with a badly
stale EMA teacher, unlike the stopped cold-start run.

The larger latent relative MSE is not hidden by the output-level result. The
objective permits behaviorally similar token distributions at different
latent coordinates, and the transition used that freedom aggressively. EMA
slightly reduced this drift relative to the direct-Open run at step 500, but
did not restore the original control's latent geometry.

## Conclusion

Warm-started full-model EMA is a valid closure teacher and modestly lowers
distributional Open/canonical divergence. It is not a complete solution:
closure still rises as training proceeds, final top-1 agreement does not
improve, and latent drift remains large. The WikiText-2 20-epoch run therefore
tests the same objective as a scale-and-budget transfer, not as an already
confirmed superior loss.
