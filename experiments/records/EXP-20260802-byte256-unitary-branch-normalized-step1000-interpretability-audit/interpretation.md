# Step-1000 real-sample interpretability result

## Exact algebraic preservation

The measurement correction decomposition reconstructed every `delta_r` with
maximum absolute error between `5.96e-8` and `1.79e-7`. Transporting every
correction through the learned non-identity `R` into the H16 frame reconstructed
the final latent with maximum error `3.43e-5` to `5.34e-5`. Thus both the
write-superposition decomposition and the `R`-transported residual
decomposition are numerically valid without imposing `R=I`.

The signed lower-triangular interference maps had effective rank
`8.04--8.28`, so they were not rank-one matrices. However, rank alone
overstates their semantic resolution: mean current-write share was `0.503` at
H2, `0.249` at H4, and `0.112` at H8, close to uniform `1/r` allocation. At
H16 it was `0.099`; most remaining mass was spread over prior writes. The
learned map therefore showed broad accumulation rather than sharp selective
retrieval of a particular earlier reasoning write.

## Grounding through actual byte predictions

Across eight fixed real validation contexts, exact H16 byte accuracy averaged
`0.25`, but each predicted 16-byte sequence used only `1--3` distinct bytes
(mean `2.125`). H1 accuracy was `0.625`; later predictions commonly collapsed
to the space byte. Example outputs included:

```text
context: red along the Matanikau River from 23 – 27 September and 6 –
gold:     9 October . \n\n 
pred:     1              

context: e Malayan Communist Party ( MCP ) . Australian involvement began
gold:     in June 1950 , 
pred:                    

context: partially supportive stance towards Scientology in relation to G
gold:    ermany . Richard
pred:    ere             
```

Mean correction norm decreased from `5.684` at H1 to `0.628` at H16 while
memory Frobenius norm increased from `210.98` to `3656.15`. This coincided
with diminishing latent changes and increasingly repetitive decoded bytes.

## Conclusion within the registered boundary

The machinery is interpretable in the algebraic sense: relative write
contributions, transported residual contributions, occupancy, and per-step
decodes can all be extracted exactly. On these samples it does not yet expose
a compelling semantic reasoning trace. The dominant observed behavior is an
initially grounded next-byte prediction followed by broad nearly-uniform
memory accumulation, shrinking corrections, and decode collapse toward a
high-frequency byte. This is evidence about these samples and checkpoint, not
a proof that the architecture cannot learn selective latent reasoning.
