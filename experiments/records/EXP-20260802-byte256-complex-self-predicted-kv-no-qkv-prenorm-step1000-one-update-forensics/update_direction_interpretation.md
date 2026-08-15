# Update-direction alignment

This audit exactly reproduced the native step-1000 current gradient and its already registered clipped-AdamW update 1001. `delta` means the actual parameter change `theta_1001 - theta_1000`; therefore `g dot delta < 0` is first-order descent. Global clipping is only a positive scalar rescaling and cannot rotate the gradient.

- Same-batch CE: `3.181236461` -> `3.195414748` (actual change `+0.0141782872`).
- Raw gradient L2: `69.363283`.
- Actual Adam parameter-delta L2: `0.203918129`.
- Global `g dot delta`: `-7.15458989` (descent).
- Cosine(`g`, `delta`): `-0.505823808`; cosine(`g`, `-delta`): `+0.505823808`.
- Actual minus first-order CE change: `+7.16876818`.

The most negative group contribution was `revblock0.ffn` (`-1.16670472`); the most positive was `revblock0.attn.qkv` (`+9.4569593e-07`). Group contributions are disjoint and sum to the global dot product.

The actual delta is first-order descent even though the finite same-batch update increased CE. Thus current-gradient/Adam direction misalignment does not explain the sign reversal; higher-order curvature or other finite-step nonlinearity must dominate along this update. This audit does not identify which.

Reproduction checks: maximum per-parameter delta-L2 relative error `6.24e-08`; group-dot sum relative error `1.24e-16`. This is a one-update local result, not a historical step-900 or long-run claim.
