# Interpretation

The preregistered initial mechanism criterion passed.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
The matched no-QKV-pre-norm final-carrier-boundary H16 recurrence was retained exactly, except a fixed non-affine RMS normalization is applied to the private encoder-root copy before it initializes both Z0 and the first KV write S0. Raw encoder and exact inverse paths remain unchanged. The fully attached loss is sixteen equal token CEs with no auxiliary target, count scaling, detach, EMA, beta, or token feedback.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 6.529102 | 2.407045 |
| H1 latent relative MSE | 1.003392 | 1.180439 |
| H1 posterior target accuracy | 0.009766 | 0.083008 |
| Mean continuous/AR state cosine | -0.001903 | -0.000440 |
| H2--H4 token agreement | 0.559896 | 0.340885 |
| No-write H2--H4 agreement | 0.559896 | 0.195182 |
| Posterior/readout mean logit delta | 0.350024 | 2.728015 |

## Initial recurrent-root normalization

Only the private encoder-root copy used to initialize recurrent Z0 and S0 was fixed-RMS-normalized. The raw encoder representation and exact inverse decoder path were preserved; every post-initialization transition remained identical to the matched control.

## Final-carrier-only boundary post-normalization

P, innovation, and the raw successor memory/read remained unnormalized. Fixed non-affine norms were applied only to the complete successor hidden and memory fields immediately before they were reused by the next central step.

| Metric | Boundary-only | Intermediate-norm control |
|---|---:|---:|
| Final block NLL | 3.095199 | 3.136612 |
| Step-50 raw global gradient norm | 21.100031 | 765.149048 |
| Step-100 raw global gradient norm | 68.031082 | 236.737488 |

## Attached stride-sixteen H16 CE

Sixteen stride-16 anchors and sixteen horizons retained 256 CE labels per sequence while increasing central sequential depth.

| Horizon | Initial NLL | Final NLL | Final target accuracy |
|---:|---:|---:|---:|
| H1 | 6.529102 | 2.407045 | 0.311523 |
| H2 | 6.483220 | 2.865540 | 0.229492 |
| H3 | 6.416051 | 2.995173 | 0.204102 |
| H4 | 6.460923 | 3.062979 | 0.206055 |
| H5 | 6.473709 | 3.229422 | 0.174805 |
| H6 | 6.355949 | 3.176746 | 0.186523 |
| H7 | 6.362181 | 3.200571 | 0.174805 |
| H8 | 6.301882 | 3.157268 | 0.170898 |
| H9 | 6.198056 | 3.178082 | 0.174805 |
| H10 | 6.296983 | 3.173594 | 0.184570 |
| H11 | 6.237132 | 3.129533 | 0.197266 |
| H12 | 6.267374 | 3.173616 | 0.198242 |
| H13 | 6.282806 | 3.188584 | 0.207031 |
| H14 | 6.297260 | 3.222126 | 0.169922 |
| H15 | 6.329270 | 3.216169 | 0.176758 |
| H16 | 6.479615 | 3.146739 | 0.192383 |

H16 block NLL: 6.360720 -> 3.095199.
H4/stride-4 block NLL: 2.385889.
H16 peak VRAM: 2828434944 bytes; H4/stride-4 peak: 2816060416 bytes.

## Matched raw-root control

The direct comparison changes only the private recurrent root used to
initialize Z0 and S0. Full paired rows are retained in `matched_control.tsv`.

| Step | Raw-root gnorm | Root-RMS gnorm | Raw-root block NLL | Root-RMS block NLL |
|---:|---:|---:|---:|---:|
| 50 | 42.898861 | 21.100031 | 3.453313 | 3.353405 |
| 300 | 95.226547 | 49.031799 | 3.184573 | 3.163019 |
| 500 | 17.839920 | 17.392740 | 3.130142 | 3.180916 |
| 600 | 26.386745 | 53.249088 | 3.125081 | 3.295152 |
| 800 | 3.379766 | 3.406077 | 3.080667 | 3.120118 |
| 900 | 1563.952271 | 2.388309 | 11.598250 | 3.102862 |
| 1000 | 35.983208 | 1.824580 | 3.167410 | 3.095199 |

The candidate was not uniformly better: step 600 retained a larger gradient
and worse block NLL than control. It recovered by step 700. At the registered
step-900 control event, however, the candidate remained in its ordinary scale
regime and avoided both the gradient and validation-NLL spike.

## Fixed anchor-0 byte-h monitor

On the registered 256-window sample, the step-1000 control's anchor-0 byte-h
H6 innovation W and raw successor Z were `3.79x` and `9.86x` its non-h rows.
The candidate ratios were `0.955x` and `0.976x`; its H2--H16 monitored raw
scale ratios never exceeded `1.053x`. Candidate H6 W and Zraw were
`0.00505x` and `0.000829x` the control values. This supports removal rather
than relocation of the localized late-horizon event, while the accompanying
global scale shift prevents a unique causal attribution from this monitor
alone.
