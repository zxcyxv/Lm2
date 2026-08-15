# Full-logit and inverse-hidden canonical-basin audit

- exact inverse hidden/token-embedding relative error mean / max:
  `0.000083278 /
  0.000220206`

## Centered full logits

- useful local lift by registered gates: **False**
- initial alignment mean / median / positive fraction:
  `0.008171 /
  0.002335 /
  0.507812`
- proposal distance step 0 / 40:
  `0.027459 /
  0.042390`
- residual removal:
  `-0.543738`
- distance monotone: **False**

## Exact-inverse hidden reconstruction

- useful local lift by registered gates: **False**
- initial alignment mean / median / positive fraction:
  `0.121222 /
  0.118762 /
  1.000000`
- proposal distance step 0 / 40:
  `0.027459 /
  0.024244`
- residual removal:
  `0.117104`
- distance monotone: **True**

The comparison locates whether canonical information is retained in the full
relative-logit fingerprint or only in the pre-head exact-inverse hidden.

## Interpretation

The analytic inverse itself is functioning: decoding canonical `hB` recovers
the selected token embedding with mean relative error `8.33e-5` and maximum
`2.20e-4`.

Full centered logits do not identify canonical `hB` in the current RMS-tied
head. The initial descent direction is effectively unrelated to the
canonical residual (median cosine `0.0023`, only 50.78% positive). Although
median logit energy falls from `3.4077` to `0.3547` and B confidence rises
from `0.1439` to `0.9939`, canonical distance increases from `0.02746` to
`0.04239`. The full relative-logit fingerprint therefore has the same
fundamental problem as single-token confidence, not merely a weaker version
of the desired inverse lift.

Inverse-hidden reconstruction behaves differently. Every initial gradient
has a positive canonical component and median distance decreases
monotonically. Active energy falls from `0.04012` to `0.00113`, but latent
distance decreases only from `0.02746` to `0.02424`—11.71% residual removal
after 40 iterations. Moreover the fixed token temporarily ceases to be
top-1: proposal-start top-1 falls to 6.25% at step 8 and recovers to only
96.09% at step 40. Thus this energy has the correct unique zero and the
correct local tendency, but ordinary Euclidean gradient descent is poorly
conditioned and is not a mode-preserving fast solver.

The mismatch between 97% inverse-hidden energy reduction and only 12% latent
residual removal exposes weak singular directions of the conditional inverse
decoder. Exact bijectivity guarantees the zero-energy endpoint but does not
guarantee a useful Euclidean condition number. A Gauss--Newton/natural
gradient preconditioner could correct those directions, but computing it
must be cheaper than literal forward encoding to preserve the proposed
advantage.

Random-sphere starts do not reach a tested low-energy distant
counterexample in 40 steps for either energy. They also barely move under
inverse-hidden reconstruction, reinforcing the severe conditioning rather
than establishing a global basin.

For this checkpoint the layer boundary is now clear:

- single-token likelihood: mode sharpening, not canonical lifting
- full centered logits: still not a canonical lift
- exact-inverse pre-head reconstruction: canonical endpoint is identifiable,
  but the naive iterative solver is slow and temporarily changes the mode

The remaining question is therefore computational, not one of endpoint
existence: whether a cheap approximation to the inverse-decoder
Gauss--Newton step can remove the 2.7% projection residual in fewer operations
than directly evaluating the encoder on the selected token.
