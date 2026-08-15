# State-dependent feedback: complete gradient-path and hidden-RC audit

## Status and scope

This is a symbolic audit of the registered
`unitary-branch-normalized-residual` graph.  It does not use transient metric
values as evidence, does not change the running training process, and does not
claim that an unexecuted counterfactual has empirical support.

The questions are:

1. Does the hidden recurrent connection
   `z_(r+1) = R z_r + delta_r` cause hidden-state norm accumulation?
2. What exact forward and backward paths disappear if `R z_r` is removed from
   the successor?
3. Which paths remain capable of amplification after that removal?
4. What condition would actually make every recurrent path stable?

## Conclusion first

Yes: the hidden recurrent connection is precisely the operation that turns a
bounded per-step measurement `delta_r` into an accumulating hidden carrier.
For a supplied innovation tape,

```text
z_N = R^N z_0 + sum_(n=0)^(N-1) R^(N-1-n) delta_n.
```

Since `R` is orthogonal, every innovation is retained without decay.  The raw
hidden norm can therefore grow as `sqrt(N)` for incoherent innovations and as
`N` for aligned innovations.  The decoder-side RMS normalization hides this
scale from the logits but does not remove it from the next recurrent state or
from the recurrent Jacobian.

Literal deletion of the skip,

```text
z_(r+1) = delta_r
```

does remove that accumulation, but it is not a viable full-width replacement
in the registered model.  `delta_r = O m_r`, where `O` maps only 496 real read
features into width 4352.  Hence every successor after the first would lie in
the fixed subspace `image(O)`, of dimension at most 496.  RMS normalization
cannot restore the missing dimensions.

The direct, full-width correction is instead to make the already-computed
decoder ray the actual successor:

```text
z_(r+1) = RMS(R z_r + delta_r).
```

This keeps the full-width input direction, fixes recurrent RMS at every step,
and forces the `R` path through the normalization Jacobian instead of allowing
an unnormalized bypass.  It needs no external input reinjection: if
`delta_r = 0`, the recurrence is exactly the unitary transport
`z_(r+1) = R z_r` on the fixed-RMS sphere.

This boundary post-normalization is a necessary structural correction, but it
is not by itself a proof
that every gradient path is non-expansive.  The bidirectional controller loop

```text
z --B(write)--> S --Q(read)--> z
```

remains.  Removing the hidden RC changes its dangerous local factor from
`R + P + Q B` to `P + Q B`; it removes the unit-modulus bypass and makes
contraction possible, but `P + Q B` must still be controlled.  No placement
of RMS/Frobenius normalization alone proves that its tangent singular values
are at most one.

## 1. Exact registered forward graph

All complex maps below are understood through their equivalent real maps;
`*` denotes the real adjoint (or conjugate adjoint in complex notation).

### 1.1 Encoder

For token embedding rows `e_t`, each reversible encoder block splits the
width into `(x_1, x_2)` and computes

```text
a       = N_1(x_2)
p       = Attn(a)
y_1     = x_1 + alpha p
b       = N_2(y_1)
f       = FFN(b)
y_2     = x_2 + beta f.
```

The registered model has a fixed finite number of these blocks.  The selected
causal anchor row is the recurrent root

```text
z_0 = Encoder([prefix, x_t])_anchor,
S_0 = 0.
```

The encoder residuals are finite-depth feature transforms.  They are not
repeated once per central horizon and therefore are not the hidden RC under
audit.

### 1.2 One central step

For horizon index `r`, the current code computes

```text
u_r       = R z_r
x_r       = N_z(u_r)
q_r       = Q x_r
k_r       = K x_r
v_r       = V x_r
W_r       = k_r v_r^dagger
S_(r+1)   = U S_r + W_r
rho_r     = ||S_(r+1)||_F
a_r       = q_r^dagger S_(r+1) / sqrt(d_k)
m_r       = a_r / (rho_r + eps)
delta_r   = O m_r
z_(r+1)   = u_r + delta_r
zhat_(r+1)= N_z(z_(r+1)).
```

`z_(r+1)` and `S_(r+1)` are the raw recurrent carriers.  Only
`zhat_(r+1)` is sent to the inverse decoder.  In particular,

```text
N_z(z_(r+1))
```

is not fed back as the next `z` in the registered mode.

### 1.3 Exact inverse decoder and head

For every future branch row, each encoder block is inverted in reverse order:

```text
x_2 = y_2 - beta FFN(N_2(y_1))
x_1 = y_1 - alpha Attn(N_1(x_2)).
```

The inverse attention is causal across the literal prefix and the preceding
future branch rows.  The registered simplex-raw tied head is

```text
ell = 16 E y,
L   = mean CE(ell, target)
```

over the supervised H1--H4 labels.  `E` is the fixed simplex embedding table
in this configuration.

## 2. Normalization differentials

### 2.1 Hidden RMS normalization

For width `d`, define

```text
r(x) = sqrt(||x||^2 / d + eps)
N_z(x) = x / r(x).
```

Its exact directional derivative is

```text
D N_z(x)[h]
  = h / r(x) - x <x,h> / (d r(x)^3).
```

Ignoring `eps`, this is

```text
D N_z(x) = (I - xhat xhat^T) / r(x).
```

It removes the radial differential but does not cap tangent amplification
unless `r(x)` and every following operator are also controlled.

### 2.2 Frobenius-normalized memory read

Let `S = S_(r+1)`, `rho = ||S||_F`, and

```text
m(q,S) = q^dagger S / (sqrt(d_k) (rho + eps)).
```

Then

```text
d rho = Re <S,dS>_F / rho
```

and

```text
d m
 = ((dq)^dagger S + q^dagger dS)
     / (sqrt(d_k) (rho + eps))
   - (q^dagger S) d rho
     / (sqrt(d_k) (rho + eps)^2).
```

At `eps = 0`, the memory part is the tangent projector

```text
d_S m = q^dagger P_(S perp) dS / (sqrt(d_k) rho).
```

Thus memory normalization removes radial memory scale.  It does not remove a
tangent perturbation that changes the interference pattern.

## 3. One-step central differential, layer by layer

Define the following local linear maps at the current trajectory point:

```text
A_r = D N_z(u_r) R
```

for the normalized hidden input,

```text
dq_r = Q A_r dz_r
dk_r = K A_r dz_r
dv_r = V A_r dz_r,
```

and

```text
B_r dz_r
  = (K A_r dz_r) v_r^dagger
    + k_r (V A_r dz_r)^dagger
```

for the `z -> write` differential.  The memory update is

```text
dS_(r+1) = U dS_r + B_r dz_r.
```

Split the measurement delta differential into its direct query path and its
memory path:

```text
d delta_r = P_r dz_r + Q_r dS_(r+1),
```

where

```text
P_r dz_r
  = O [ (Q A_r dz_r)^dagger S_(r+1)
        / (sqrt(d_k) (rho_r + eps)) ]
```

and `Q_r` is `O` composed with the exact Frobenius-normalized memory
differential in Section 2.2.  Therefore the current hidden successor has

```text
dz_(r+1)
  = (R + P_r + Q_r B_r) dz_r + Q_r U dS_r.
```

For the joint recurrent state `y_r = (z_r,S_r)`, the exact block Jacobian is

```text
                 [ R + P_r + Q_r B_r    Q_r U ]
J_r^(RC)      =  [                              ].
                 [ B_r                         U     ]
```

The unitary diagonal blocks do not make this joint matrix unitary.  The term
`Q_r B_r` is the same-step closed write/read loop, while `R` is the direct
hidden bypass around every normalization and measurement operation.

## 4. Exact central backward adjoints

Let `lambda^z_(r+1)` and `lambda^S_(r+1)` be the total adjoints arriving from
later horizons, plus the decoder loss injected at horizon `r+1`.  Reverse the
step in execution order.

### 4.1 Decoder-ray injection

If `g_(r+1)` is the adjoint returned by the inverse decoder for
`zhat_(r+1)`, then the raw successor receives

```text
lambda^z_(r+1) += D N_z(z_(r+1))^* g_(r+1).
```

This normalization lies on the loss branch only.  It does not occur between
`z_(r+1)` and central step `r+1`.

### 4.2 Read and memory update

The read first adds

```text
bar S_(r+1) = lambda^S_(r+1) + Q_r^* lambda^z_(r+1).
```

The local query path adds

```text
P_r^* lambda^z_(r+1)
```

to `z_r`.  Reversing the memory write and the two unitary transports gives

```text
lambda^S_r
  = U^* (lambda^S_(r+1) + Q_r^* lambda^z_(r+1))
```

and

```text
lambda^z_r
  = R^* lambda^z_(r+1)
    + P_r^* lambda^z_(r+1)
    + B_r^* (lambda^S_(r+1) + Q_r^* lambda^z_(r+1)).
```

Equivalently,

```text
lambda^z_r
  = (R + P_r + Q_r B_r)^* lambda^z_(r+1)
    + B_r^* lambda^S_(r+1).
```

The first term `R^* lambda^z_(r+1)` is an exact unnormalized gradient copy at
every horizon.  All loss injections can accumulate on this highway.

### 4.3 Parameter gradients at one step

The shared parameter gradients are sums over all horizons.  At one horizon:

```text
bar O        += bar delta_r m_r^*
bar q_r      += derivative of normalized read wrt q
bar S_(r+1)  += derivative of normalized read wrt S
bar k_r      += bar W_r v_r
bar v_r      += bar W_r^* k_r
bar Q        += bar q_r x_r^*
bar K        += bar k_r x_r^*
bar V        += bar v_r x_r^*
bar theta_R  += <bar u_r, (partial R / partial theta_R) z_r>
bar theta_U  += <bar S_rot,r, (partial U / partial theta_U) S_r>.
```

Because Q/K/V/O and both phases are reused at every horizon, the final
parameter gradient is the sum of these contributions after later-horizon
adjoints have propagated through the joint Jacobians.

## 5. Encoder, inverse decoder, head, and CE adjoints

These paths can amplify a central adjoint, but their depth is fixed when the
central horizon `N` grows.  They must be included in the global gradient, but
they are not an `N`-deep recurrent product.

### 5.1 Cross entropy and raw simplex head

For one target `t`,

```text
p       = softmax(ell)
bar ell = (p - onehot(t)) / number_of_labels
bar y   = 16 E^T bar ell.
```

The tied simplex table is frozen in the registered model.  Thus the head sends
the adjoint to the inverse decoder but has no trainable embedding update.

### 5.2 One inverse reversible block

The inverse block is

```text
x_2 = y_2 - beta F(N_2(y_1))
x_1 = y_1 - alpha A(N_1(x_2)).
```

Given output adjoints `(bar x_1, bar x_2)`, first accumulate

```text
bar x_2,total
  = bar x_2 - alpha D N_1(x_2)^* J_A^* bar x_1,
```

then

```text
bar y_2 = bar x_2,total
bar y_1 = bar x_1
          - beta D N_2(y_1)^* J_F^* bar x_2,total.
```

Causal branch attention also distributes each later branch loss into earlier
branch rows and the literal prefix.  This creates cross-horizon sums in the
decoder, but only across the fixed number of inverse blocks.

### 5.3 One forward reversible encoder block

The encoder block is

```text
y_1 = x_1 + alpha A(N_1(x_2))
y_2 = x_2 + beta F(N_2(y_1)).
```

Given `(bar y_1, bar y_2)`, define

```text
bar y_1,total
  = bar y_1 + beta D N_2(y_1)^* J_F^* bar y_2.
```

Then

```text
bar x_1 = bar y_1,total
bar x_2 = bar y_2
          + alpha D N_1(x_2)^* J_A^* bar y_1,total.
```

The encoder receives both the recurrent-root adjoint and prefix adjoints from
the exact inverse decoder.  Again, the number of these reversible blocks is
fixed and does not grow with central recurrence horizon.

## 6. Where the horizon dependence enters

For a final-horizon loss, parameter sensitivity satisfies

```text
G_(r+1) = J_r G_r + C_r,
```

so

```text
G_N = sum_(s=0)^(N-1)
        J_(N-1) ... J_(s+1) C_s.
```

With a mean loss over every horizon,

```text
L_N = (1/N) sum_(h=1)^N ell_h,
```

the gradient contains a triangular set of paths:

```text
grad L_N
  = (1/N) sum_(h=1)^N sum_(s=0)^(h-1)
      C_s^* J_(s+1)^* ... J_(h-1)^* grad ell_h.
```

There are `O(N^2)` path terms before the `1/N` loss average.  If recurrent
products are norm-preserving and terms align, the result is `O(N)`.  If a
joint Jacobian has a persistent singular value above one, the product is
exponential in `N`.

## 7. Why the hidden RC grows the raw hidden norm

The current successor obeys the exact identity

```text
||z_(r+1)||^2
 = ||z_r||^2 + ||delta_r||^2
   + 2 <R z_r, delta_r>,
```

because `R` is orthogonal.  Therefore:

- orthogonal or zero-mean innovations accumulate energy, giving typical
  `||z_N|| = O(sqrt(N))`;
- positively aligned innovations give `||z_N|| = O(N)`;
- only systematically negative cross terms can prevent accumulation.

Branch normalization bounds the scale of each measurement but does not
subtract previously accumulated hidden energy.  Decoder RMS normalization
does not help because the raw `z_(r+1)` is the next recurrent input.

In a rotating frame, define `ztilde_r = R^(-r) z_r`.  Then

```text
ztilde_(r+1) = ztilde_r + R^(-(r+1)) delta_r.
```

Thus the supposedly unitary hidden dynamics are exactly an ordinary additive
residual accumulation in rotating coordinates.

## 8. Literal RC-removal counterfactual

Change only the hidden successor to

```text
z_(r+1) = delta_r.
```

The memory update and state-dependent Q/K/V remain attached.  The joint
Jacobian becomes

```text
                 [ P_r + Q_r B_r    Q_r U ]
J_r^(no-RC)   =  [                         ].
                 [ B_r               U     ]
```

The exact `R` bypass is gone.  All hidden-to-hidden gradient must pass through
the query/write/read controller, while `U` remains as the single lossless
memory highway.

### 8.1 Horizon-independent hidden forward bound

By Frobenius Cauchy--Schwarz,

```text
||q_r^dagger S_(r+1)||
  <= ||q_r|| ||S_(r+1)||_F.
```

Hence

```text
||m_r|| <= ||q_r|| / sqrt(d_k)
```

up to the positive `eps` correction, and

```text
||z_(r+1)||
  = ||O m_r||
  <= ||O|| ||Q|| ||N_z(R z_r)|| / sqrt(d_k).
```

Since `N_z(R z_r)` has fixed RMS, this bound depends on the learned operators
but not on the number of previous recurrent steps and not on `||S_r||`.
This proves that removing the hidden RC eliminates horizon-driven hidden norm
accumulation.

### 8.2 What input information remains, and what is lost

The registered initialization is `S_0 = 0`, but the first update is

```text
S_1 = W(z_0).
```

Therefore a projection of the encoded condition enters the persistent unitary
memory before the first hidden successor is formed.  This is not an injective
copy of the width-4352 root.  The first write is generated only from 256 real
key features and 496 real value features, while the query contributes another
256 real features to the first read.  More decisively, every literal no-RC
successor lies in the at-most-496-dimensional `image(O)`.

Thus `S_1` can retain task-relevant projected information, but it cannot be
used as a proof that the complete input condition survives literal skip
deletion.  A viable no-external-reinjection design must retain a full-width
state path inside a bounded recurrent map.

### 8.3 What RC removal does not prove

Set `R = U = I`, ignore the direct query term `P`, and restrict `B,Q` to one
paired scalar direction with `B=b`, `Q=d`, and `g=db`.

With the hidden RC,

```text
J_RC = [[1+g, d],
        [b,   1]],
```

whose eigenvalues are

```text
lambda_+- = (2+g +- sqrt(g^2+4g))/2.
```

For small positive `g`, `lambda_+ = 1 + sqrt(g) + O(g)`.

Without the hidden RC,

```text
J_noRC = [[g, d],
          [b, 1]],
```

whose eigenvalues are `0` and `1+g`.  Removing the RC eliminates the much more
sensitive `sqrt(g)` hyperbolic split and removes hidden accumulation, but a
positive write/read loop gain still gives `1+g > 1`.

Thus the correct statement is:

```text
hidden RC removal is necessary for bounded hidden scale and removes the worst
neutral-bypass instability; it is not sufficient to certify the remaining
state-dependent feedback loop.
```

## 9. Path inventory before and after RC removal

| Path | Current RC graph | After hidden-RC removal |
|---|---|---|
| `z_r -> R z_r -> z_(r+1)` | Exact unit-modulus bypass | Removed |
| bounded `delta` accumulation in `z` | Retained forever | Removed |
| decoder RMS radial projection | Loss branch only | Can also become recurrent boundary |
| `S_r -> U S_r -> S_(r+1)` | Exact unitary memory highway | Preserved |
| first-root information | In both `z` and `S_1` | Preserved in `S_1` |
| direct query path `P_r` | Added on top of `R` | Becomes part of the complete hidden transition |
| write/read loop `Q_r B_r` | Added on top of `R` | Still present, but no identity/unitary bypass |
| shared-parameter horizon sum | Present | Still present |
| raw memory write accumulation | Present | Still present |

## 10. What each possible intervention actually guarantees

### A. Literally remove only the hidden RC

```text
z_(r+1) = delta_r.
```

Guarantees:

- no hidden `R^N` forward or backward bypass;
- no horizon-driven accumulation of bounded innovations in raw hidden norm;
- one lossless carrier (`S`) remains for projected memory information.

Does not guarantee:

- `||P_r + Q_r B_r|| <= 1`;
- bounded raw memory norm;
- elimination of the shared-parameter `O(N)` sum.
- preservation of a full-width hidden representation; in fact the successor
  rank is at most 496 in the registered width-4352 model.

This is a diagnostic ablation, not the recommended replacement.

### B. Put the full residual update inside the recurrent post-norm boundary

```text
z_(r+1) = N_z(R z_r + delta_r).
```

Guarantees:

- the recurrent hidden state has fixed RMS;
- its radial derivative is removed at every boundary;
- the state consumed by the next step is exactly the same full-width ray that
  the decoder already consumes;
- when `delta_r = 0`, the complete input state follows the exact unitary orbit
  and is not forgotten;
- the hidden `R` path no longer bypasses normalization.

Still not guaranteed:

- tangent non-expansion of `P_r + Q_r B_r`.

This is the cleanest direct correction of the present architecture.  It is
the analogue of keeping residuals inside a post-normalized recurrent unit
without adding an outer unnormalized RC around that unit.

Writing `y_r = R z_r + delta_r`, its joint Jacobian is

```text
                    [ D N_z(y_r)(R + P_r + Q_r B_r)   D N_z(y_r) Q_r U ]
J_r^(hidden-post) =  [                                                    ].
                    [ B_r                              U                  ]
```

There is no longer a path from `z_r` to `z_(r+1)` that bypasses
`D N_z(y_r)`.  This, rather than literal deletion of the full-width carrier,
is the relevant removal of the outer recurrent RC.

### B.1 Tangent-only innovation

Boundary post-norm can amplify tangent gradients if `delta_r` strongly
anti-aligns with `u_r = R z_r` and makes the pre-normalization norm small.  A
structural way to remove that case is

```text
delta_perp = delta_r
             - u_r <u_r,delta_r> / ||u_r||^2
z_(r+1)    = N_z(u_r + delta_perp).
```

Then

```text
||u_r + delta_perp||^2
  = ||u_r||^2 + ||delta_perp||^2
  >= ||u_r||^2.
```

For a frozen innovation tape, the normalization factor on the direct carrier
path is therefore never larger than one.  The update changes only the carrier
direction, which is also the only quantity observed by the current decoder.
This still does not bound the state derivative of the innovation generator
itself; that derivative is the remaining `P_r + Q_r B_r` problem.

### C. Normalize the memory carrier after every write

```text
S_(r+1) = N_S(U S_r + W_r).
```

This bounds memory scale, but it changes the relative coefficients of all past
writes and inserts a rank-deficient projector into the only remaining lossless
carrier.  It is not a neutral companion to hidden-RC removal and must be a
separate mechanistic experiment.

### D. Break the feedback edge

Precompile writes from a root orbit, making

```text
partial W_(r+1) / partial z_r = 0.
```

Then the recurrent `B` edge vanishes and the result is the registered parallel
scan.  This gives the cleanest affine stability but removes the desired
state-dependent correction.

### E. Preserve state dependence with a hard all-path condition

For a positive-definite joint metric `M`, the exact condition is

```text
J_r^* M J_r <= M
```

for every reachable recurrent state.  In Euclidean coordinates this reduces
to `||J_r||_2 <= 1`.  Neither branch RMS nor Frobenius read normalization
implies this inequality.  To guarantee it while retaining state dependence,
the complete coupled update—not Q/K/V/O independently—must be parameterized
as a non-expansive joint map or must satisfy a tied negative-feedback/Lyapunov
condition.  An additive pair of independently learned write and read shears
does not provide that guarantee.

## 11. Structural recommendation

The next architecture question should not be “reinject the original encoder
root,” and literal `z_(r+1)=delta_r` is not a full-width solution.  The
controlled sequence is:

1. Normalize the private initial recurrent root once.
2. Keep `R z_r + delta_r` as an internal full-width residual update, but feed
   `N_z(R z_r + delta_r)` back as the actual next recurrent state.
3. Project `delta_r` onto the tangent space of `R z_r` before the sum so an
   anti-aligned innovation cannot shrink the normalization denominator.
4. Remove the now-redundant Q/K/V pre-normalization: the initial root is
   normalized, `R` preserves its RMS, and every later successor is already a
   boundary-postnormalized state.  Keeping another pre-normalizer changes the
   backward map even when it is an identity in forward values.
5. Use this same normalized successor for both the next Q/K/V step and the
   inverse decoder.
6. Measure the remaining controller loop `P_r + Q_r B_r` separately from the
   memory highway `U`.
7. If a hard N=512 guarantee is required, redesign or tie the complete
   write/read coupling to satisfy `J_r^* M J_r <= M`; another local
   normalization cannot establish it.

The first five items preserve the full-width initial condition without external
reinjection while eliminating raw hidden norm accumulation.  Item seven is the
remaining mathematical requirement for “all paths stable”; it must not be
conflated with forward norm control.

## 12. Full N-scaling verdict for the hidden-postnorm candidate

The candidate

```text
z_(r+1) = N_z(R z_r + delta_perp_r)
```

does **not** eliminate every `N`, `N^2`, or exponential gradient factor.  It
solves recurrent hidden-scale growth and removes an unnormalized hidden bypass;
the following paths remain.

### 12.1 Exact N factor from a learned unitary phase

Set every innovation to zero.  Because the state already has fixed RMS,

```text
z_N = R(theta)^N z_0.
```

For a commuting skew generator `A`, `R(theta) = exp(theta A)`, so

```text
partial z_N / partial theta
  = N A R(theta)^N z_0.
```

This derivative is tangent to the fixed-norm sphere.  Boundary RMS
normalization therefore does not remove it.  A learned neutral transport used
`N` times has an exact `O(N)` parameter sensitivity even when all activations
and state adjoints have constant norm.

The same fact holds for a final-horizon loss.  For a mean loss over all
horizons,

```text
(1/N) sum_(h=1)^N O(h) = O(N).
```

Thus state-norm stability and horizon-invariant shared-parameter gradients are
different requirements.

### 12.2 Raw N^2 age weighting in the learned memory phase

For a supplied write tape,

```text
S_N = sum_(n=1)^N U^(N-n) W_n,
```

and hence

```text
partial S_N / partial theta_U
  = sum_(n=1)^N (N-n)
      U^(N-n-1) (partial U / partial theta_U) W_n.
```

The scalar age coefficients sum to `N(N-1)/2`.  This is an `O(N^2)` raw
carrier sensitivity for coherently aligned writes.  Dividing the read by
`||S_N||_F` usually removes one amplitude power—coherent `||S_N||=O(N)` and
incoherent `||S_N||=O(sqrt(N))` both generically leave `O(N)` normalized phase
sensitivity—but it is not an `O(1)` path and cancellation can make the
denominator worse conditioned.

### 12.3 Positive write/read feedback survives hidden postnorm

Restrict the hidden-postnorm joint Jacobian to one tangent hidden direction
and one memory direction.  Let the normalization tangent gain be `0 < c <= 1`,
the write derivative be `b`, the read derivative be `d`, and `g = db > 0`.
Ignoring the direct query derivative only makes the counterexample smaller:

```text
J_post = [[c(1+g), c d],
          [b,        1]].
```

Its determinant is `c`, its trace is `1 + c(1+g)`, and its characteristic
polynomial evaluated at one is

```text
p(1) = -c g < 0.
```

Therefore the larger eigenvalue is strictly greater than one.  Repetition
gives `lambda_+^N`.  Tangent innovation and boundary postnorm prevent radial
state growth, but they do not change the sign of the state-dependent
write/read loop gain.

### 12.4 Shared controller parameters still sum across uses

Even if every recurrent state Jacobian were exactly non-expansive, a parameter
used once per step has

```text
partial y_N / partial theta
  = sum_(s=0)^(N-1)
      J_(N-1) ... J_(s+1) C_s.
```

With a neutral carrier and aligned terms, this is `O(N)`.  An H1--HN mean loss
contains `O(N^2)` path terms before its `1/N` average and therefore also has an
`O(N)` coherent worst case.  Reassociation, parallel scan, postnorm, and
fixed activation RMS do not alter this algebra.

### 12.5 Consequence: what is required for no horizon amplification

No autonomous architecture can simultaneously have all three properties
without an additional separation:

1. preserve arbitrary information about the initial condition indefinitely;
2. let shared trainable parameters modify that neutral carrier at every step;
3. guarantee parameter sensitivity independent of `N`.

Indefinite information preservation requires a neutral direction.  Reusing a
trainable parameter on that direction makes its derivative accumulate with the
number of uses.

For a hard N-independent design without external input reinjection, split the
state into:

```text
c_(r+1) = R_0 c_r
```

with fixed, non-trainable unitary `R_0` as the full-width conserved context
carrier, and

```text
h_(r+1), S_(r+1) = F_theta(h_r, S_r; c_r)
```

as a postnormalized or contractive working subsystem with no outer residual.
The conserved carrier keeps the input condition; because its transport has no
trainable parameter, it has no `N`-fold phase gradient.  The trainable working
subsystem must satisfy a joint Lyapunov inequality

```text
J_F^* M J_F <= rho^2 M,   rho < 1,
```

so its repeated parameter contributions form a bounded geometric series.
Any trainable unitary memory phase retained in a neutral subsystem has the same
exact `O(N)` sensitivity as `R(theta)` and must likewise be fixed, applied only
once, or explicitly normalized per use.

Therefore the hidden-postnorm/tangent candidate is a valid forward-stability
ablation, not an all-path N-independent solution.  The all-path solution
requires separating a parameter-free conserved carrier from a contractive
trainable recurrent controller.

The later distinction between hidden-only detach, full `(z,S)` carry detach,
and post-update memory normalization is recorded separately in
[urm_carry_vs_complex_memory_recurrence_ko.md](urm_carry_vs_complex_memory_recurrence_ko.md).
