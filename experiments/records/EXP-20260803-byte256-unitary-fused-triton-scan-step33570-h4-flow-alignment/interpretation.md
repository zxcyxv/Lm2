# Epoch-10 H1-H4 flow-alignment interpretation

## Verdict

There is a statistically clear learned direction signal on the normalized
carrier ray, especially at H3 and H4.  Its magnitude is nevertheless far too
small to make the present CE-only recurrence an encoder-conjugate flow, and
the signal is not stronger on correctly decoded bytes.

The result supports a small controlled ray-velocity auxiliary experiment.  It
does not support reinterpreting the current checkpoint as already performing
flow matching or replacing CE with a flow objective.

## Matched H1-H4 results

Only H1-H4 were evaluated.  Step 10071 and step 33570 use identical fixed
validation starts and the identical fused Triton recurrence.

| Horizon | Ray cosine, ep. 3 | Ray cosine, ep. 10 | Paired change (95% CI) | Predicted/gold ray-step norm, ep. 10 |
| ---: | ---: | ---: | ---: | ---: |
| H1 | 0.0362 | 0.0358 | -0.0004 [-0.0117, 0.0109] | 3.53% |
| H2 | 0.0113 | 0.0306 | +0.0193 [0.0093, 0.0293] | 2.33% |
| H3 | 0.0521 | 0.1124 | +0.0603 [0.0516, 0.0690] | 1.46% |
| H4 | 0.1078 | 0.1652 | +0.0574 [0.0503, 0.0645] | 0.87% |

The sample-clustered H1-H4 mean ray-cosine change is +0.0342 with a 95%
paired interval of [0.0309, 0.0374].  Therefore the H3/H4 directional increase
is not ordinary finite-sample noise.

The absolute epoch-10 ray-cosine 95% intervals are also positive:

- H1: [0.0166, 0.0549]
- H2: [0.0138, 0.0474]
- H3: [0.0946, 0.1303]
- H4: [0.1509, 0.1795]

## Why this is not yet a flow

Direction cosine omits vector magnitude.  The learned ray-step becomes
smaller with horizon while the gold ray-step remains about 0.13:

| Horizon | Predicted ray-step norm | Gold ray-step norm | Relative error |
| ---: | ---: | ---: | ---: |
| H1 | 0.00471 | 0.13903 | 0.99944 |
| H2 | 0.00314 | 0.13841 | 0.99943 |
| H3 | 0.00193 | 0.13546 | 0.99840 |
| H4 | 0.00113 | 0.13175 | 0.99855 |

Combining direction and norm, the component of the prediction along the gold
velocity is only 0.12%, 0.09%, 0.17%, and 0.15% of the target magnitude at
H1-H4.  The positive cosine therefore describes a tiny directional bias, not
a trajectory-following update.

Raw-carrier velocity is even less compatible.  Its epoch-10 cosine is negative
at every horizon: -0.0682, -0.0360, -0.0250, and -0.0925.  Because encoder
carrier scale changes during training, the ray result is the fairer geometry,
but neither geometry supplies a magnitude-matched velocity.

## Endpoint and correctness controls

Endpoint state-direction cosine remains high because the unitary base orbit
starts close to the gold encoder orbit, but it worsens with both horizon and
training:

| Horizon | Endpoint cosine, ep. 3 | Endpoint cosine, ep. 10 |
| ---: | ---: | ---: |
| H1 | 0.9904 | 0.9899 |
| H2 | 0.9843 | 0.9791 |
| H3 | 0.9733 | 0.9614 |
| H4 | 0.9580 | 0.9357 |

Token accuracy improves over the same checkpoints, from 41.0% to 47.7%
across H1-H4.  Thus better CE predictions do not arise by moving the full
latent state closer to the encoder trajectory.

Within every epoch-10 horizon, ray-velocity cosine is lower for correct bytes
than incorrect bytes:

| Horizon | Correct | Incorrect |
| ---: | ---: | ---: |
| H1 | 0.0085 | 0.0916 |
| H2 | 0.0237 | 0.0375 |
| H3 | 0.0989 | 0.1223 |
| H4 | 0.1568 | 0.1690 |

Consequently the direction signal cannot currently be interpreted as the
mechanism that makes token predictions correct.  It may be a generic
horizon-dependent geometric component rather than instance-specific semantic
transport; a shuffled-target vector control would be needed to separate those
possibilities.

## FM potential

The evidence separates three claims:

1. **Can the architecture represent a gold-oriented tangent direction?**
   Probably yes.  The robust H3/H4 cosine increase shows a learnable signal.
2. **Does CE alone turn that signal into encoder-conjugate dynamics?** No.  Its
   magnitude shrinks, endpoint agreement worsens, and correctness does not
   track alignment.
3. **Is an FM-style auxiliary worth one controlled run?** Yes, but only as a
   small ray-space auxiliary with CE retained.

The appropriate first intervention is a detached-target relative ray-velocity
loss over H1-H4, or the simpler endpoint ray-direction loss.  Raw innovation
MSE is inappropriate because radial scale is not decoder-identifiable and has
already drifted substantially.  This intervention remains discrete latent
velocity matching; full distributional flow matching would additionally need
an explicit interpolation/noise path and would be a larger mechanism change.

## Artifacts

- `velocity.tsv`: paired per-sample H1-H4 observations for both checkpoints
- `velocity_summary.tsv`: checkpoint, correctness, and horizon aggregates
