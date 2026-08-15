# EXP-20260804 13M conditional ray flow matching

## Status

- State: epoch-1 execution in progress; original ten-epoch run curtailed by
  user request because the available GPU is an RTX A4500 rather than RTX 5090
- Parent: completed 13M fused Triton time-varying scan at epoch 10
- Training and fixed validation splits only; test unread

## Question and matched comparison

Does a mathematically explicit conditional flow-matching objective on the
decoder-visible latent ray improve the central model's H16 transport and
epoch-level generation quality, rather than merely making a discrete noisy
recurrence insensitive to its noise?

The matched reference is
`EXP-20260803-byte256-unitary-fused-triton-scan-h16-10epoch-13m-rtx5090`:
the same byte data, seed, random-window protocol, width 1344 reversible
encoder/exact-inverse decoder, H16 central scan, optimizer schedule, and ten
epoch-equivalent 33,570-step budget.  The new model adds only a two-vector
flow-time conditioner (2,688 parameters), so both remain in the 13.2M class.

This run is from a fresh seed-1337 initialization, not a continuation from
the reference checkpoint.  Preserved reference records are not overwritten.

## Registered conditional flow

The flow state is the complete H16 decoder-visible ray tape in the rotating
frame.  For prefix-root ray `r`, each horizon has an independent Gaussian
tangent draw with the same RMS angular scale `sigma = 0.8` radians:

```math
e_h=(I-rr^T)\epsilon_h/\sqrt d,\qquad
x_{0,h}=\operatorname{Exp}_r(0.8 e_h).
```

Independent equal-scale draws retain full product-space source dimension;
replicating one draw over all horizons would make the deterministic flow
dimension-deficient.  The target is the detached future-encoder ray
co-rotated back by `R^{-h}`.  For one shared `t ~ Uniform(0,1)` per H16 tape,
`x_t` is the shortest-sphere SLERP bridge from `x_0` to the target and
`u_t = d x_t / dt` is its analytic tangent velocity.

The learned vector field receives `x_t`, `t`, and causal horizon order.  It
uses the same packed Q/K/V projections, unitary complex-memory scan, and
readout as the token scan; a learned sine/cosine width-1344 shift supplies
explicit time conditioning.  Its output is co-rotated back and projected
onto the tangent space at `x_t`.

The registered training objective is

```math
L = L_{CE}^{scan} + 0.1\,E_{t,x_0,x_1}\|v_\theta(x_t,t)-u_t\|_2^2.
```

Endpoint pairs and bridge inputs are detached for the CFM branch so the
encoder cannot reduce this loss by moving its coordinate system.  CE remains
attached and trains the reversible representation and the same central
transition.  No score loss is used: this is deterministic conditional flow
matching, not an SDE or diffusion likelihood.

At evaluation the registered flow generator starts from the same conditional
tangent source and integrates the learned tangent field from 0 to 1 with
eight exponential-map Euler steps, then rotates the H16 tape forward and
decodes it through the exact inverse.  This numerical endpoint path is
reported separately from the ordinary deterministic scan path.

## Fixed protocol

- seed 1337; data sampler seed 1337; flow-noise seed 25617
- `data/wikitext103_bytes/train.bin`: 55,000,000 byte tokens
- 33,570 optimizer steps; batch/microbatch 64; 16,384 CE targets per step
- width 1344, two reversible encoder blocks, exact inverse decoder
- H16, CE anchor stride 16; CFM anchor stride 64 to control the second-scan
  cost without changing its bridge or horizon distribution
- eight heads, complex key dimension 16, value dimension 31
- fused Triton rotating-frame memory and latent scans where applicable
- strict float32, TF32 disabled, fused AdamW, beta `(0.9, 0.95)`, zero weight
  decay, raw global clip norm 1.0
- 100-step linear warmup followed by cosine decay over 33,570 steps
- fixed 64-example scan validation; fixed first 16 examples for CFM bridge
  and eight-step flow-endpoint audits
- reports and ignored checkpoints every 3,357 steps; test split unread
- after training, fixed prompts and sampling seeds evaluate both scan H1/H16
  generation and eight-step flow H16 generation at every epoch checkpoint

## Success and interpretation criteria

- all losses and gradients remain finite through epoch 10
- report CE, absolute CFM velocity MSE, target energy, relative velocity MSE,
  velocity cosine, ordinary scan H1/H16/block NLL, and fixed-noise flow
  endpoint NLL/cosine in TSV
- epoch quality is considered improved only when held-out metrics improve;
  decreasing training CFM loss alone is insufficient
- compare each epoch with its predecessor and compare the final ordinary-scan
  metrics with the preserved CE-only reference at the same step budget
- inspect fixed-prompt free generation for repetition, byte/UTF-8 validity,
  entropy, and qualitative collapse; endpoint NLL is not by itself a free-run
  quality claim
- a single fixed-noise flow endpoint NLL is a trajectory diagnostic, not an
  exact marginal token likelihood; do not relabel it as MLE
- keep negative or conflicting evidence and restrict conclusions to this
  source scale, bridge, vector-field parameterization, and compute budget
- metrics go to TSV and interpretation to Markdown; checkpoints remain under
  ignored `outputs/` and are not added to Git

## Preflight (not outcome evidence)

- Shared CFM geometry, causal vector field, and endpoint integrator passed 36
  focused tests including all preserved complex-scan tests.
- A two-step batch-64 CUDA smoke used 9,200,798,208 peak allocated bytes.
- At the first smoke update, CE was `44.9962` and unweighted CFM MSE was
  `1.4872`; the preregistered weight therefore contributed about `0.1487`.
- Smoke records and checkpoints live only under ignored `outputs/smoke/`.

## Execution-scope amendment

After the run began, measured throughput was about `0.53 s/step`, roughly a
five-hour ten-epoch ETA on the available RTX A4500.  The user requested an
approximate first-epoch result instead of waiting for all ten epochs.  The
process therefore retains the originally registered 33,570-step LR schedule
but will stop after the step-3,357 epoch checkpoint is fully written.  This
preserves an exact epoch-1 comparison with the parent schedule; it is not a
ten-epoch outcome and must not be reported as one.
