# Interpretation

The requested performance evaluation restores gold context only at each
sixteen-token boundary and greedily re-enters the model's own selected tokens
inside that block. The open-H16 trace below was produced by the inherited
monitor; it is retained as an out-of-training-contract diagnostic and is not
used to rank the H1 token-reinput model.

The preregistered initial mechanism criterion passed.

The central recurrence consumed only its own continuous predictions. The greedy-AR path was used only by the evaluator and never supplied a token, embedding, hidden state, probability, or gradient to the trained recurrence.
The state-dependent H16 feedback architecture used H1 token CE as its only loss at all 256 stride-one anchors. Causally downstream H2--H16 training nodes were dead-code-eliminated after preflight proved their removal preserved the selected loss and parameter gradients. Validation unfolded full H16.

| Metric | Step 0 | Final |
|---|---:|---:|
| H1 validation NLL | 48.574464 | 1.303748 |
| H1 latent relative MSE | 0.965087 | 0.872284 |
| H1 posterior target accuracy | 0.012695 | 0.355469 |
| Mean continuous/AR state cosine | 0.975603 | 0.432982 |
| H2--H4 token agreement | 0.997266 | 0.065820 |
| No-write H2--H4 agreement | 1.000000 | 0.085286 |
| Posterior/readout mean logit delta | 3.327733 | 3.392050 |

## H1-supervised feedback recurrence with H16 monitoring

Training backpropagated CE through one state-dependent central transition
only. Validation unfolded the same learned transition open-loop through H16;
H2--H16 are retained diagnostics, not the intended inference protocol.

| Horizon | Supervised in training | Initial NLL | Final NLL |
|---:|:---:|---:|---:|
| H1 | yes | 48.574463 | 1.303748 |
| H2 | no | 47.105810 | 7.780687 |
| H3 | no | 46.189923 | 7.973337 |
| H4 | no | 45.569324 | 8.999118 |
| H5 | no | 45.260020 | 10.245600 |
| H6 | no | 45.482996 | 10.993760 |
| H7 | no | 45.470674 | 12.320088 |
| H8 | no | 45.309254 | 13.608441 |
| H9 | no | 45.122315 | 15.144634 |
| H10 | no | 44.584617 | 16.714743 |
| H11 | no | 44.656865 | 17.962675 |
| H12 | no | 44.118214 | 18.918137 |
| H13 | no | 43.403324 | 20.991753 |
| H14 | no | 44.332850 | 20.783013 |
| H15 | no | 44.496900 | 20.922381 |
| H16 | no | 43.872139 | 21.577350 |
