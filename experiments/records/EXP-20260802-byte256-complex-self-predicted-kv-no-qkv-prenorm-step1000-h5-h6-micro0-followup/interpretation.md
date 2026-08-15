# Interpretation

## Evidence boundary

This follow-up was preregistered after microbatch 0 was identified as the second large physical component. It reuses the same native batch and therefore tests within-batch repetition, not population frequency or causality. Every differentiation preserves the original batch-16 forward shape.

## Reconstruction

Microbatch-0 H5/H6/combined norms were `6.13888528` / `20.0344504` / `26.1548927` and reproduced the primary TSV within maximum relative error `0`. Row reconstruction errors were `0.000109` / `3.76e-05`; anchor reconstruction errors were `1.54e-05` / `2.79e-05`.

## Row localization

- row 10: L2 `18.7842564`, CE `3.39453363`, cosine `0.998808994`, full projection coefficient `0.323785262`
- row 4: L2 `8.26445763`, CE `3.16828418`, cosine `0.998973442`, full projection coefficient `0.142478365`
- row 1: L2 `0.937457323`, CE `3.85464549`, cosine `-0.959446789`, full projection coefficient `-0.015522191`
- row 3: L2 `0.255942547`, CE `3.14706004`, cosine `-0.00759131091`, full projection coefficient `-3.35304892e-05`
- row 7: L2 `0.225187111`, CE `3.05056095`, cosine `0.0244382254`, full projection coefficient `9.49716139e-05`

The dominant row was `10`.

Rows 10 and 4 supplied positive full-vector projection coefficients `0.32379`
and `0.14248`; row 1 supplied an opposed `-0.01552`. These three rows are the
only microbatch-0 windows whose anchor-0 input is byte ID 104 (`h`). Because
the model is causal, their anchor-0 forward trajectories are exactly identical;
their H5/H6 targets differ (`o/s`, `a/n`, and `y/a`), so the observed alignment
and cancellation are target-adjoint differences on one shared trajectory.

## Anchor localization

- anchor 0: L2 `18.7831691`, CE `3.59116054`, cosine `0.999030842`, full projection coefficient `0.323838432`
- anchor 11: L2 `0.0828331605`, CE `5.20880175`, cosine `-0.00281065856`, full projection coefficient `-4.01784285e-06`
- anchor 6: L2 `0.0595226644`, CE `4.39832187`, cosine `-0.000127470395`, full projection coefficient `-1.30939995e-07`
- anchor 5: L2 `0.0589907914`, CE `3.94135952`, cosine `-0.00763370497`, full projection coefficient `-7.77141708e-06`
- anchor 12: L2 `0.057571622`, CE `3.57235909`, cosine `0.00568877671`, full projection coefficient `5.6520758e-06`

The dominant anchor was `0` at position `0`, with input token ID `104`.

## Direct pattern comparison

- followup_micro0: H5 Z/S denominator ratios `0.0532708` / `0.0559199`; H6 P/W/Zraw/Sraw ratios `2.04358` / `3.79184` / `9.87216` / `3.66243`
- primary_micro3: H5 Z/S denominator ratios `0.0531905` / `0.0558927`; H6 P/W/Zraw/Sraw ratios `2.0425` / `3.78875` / `9.86026` / `3.65958`

The ratios are descriptive co-location. They do not establish that the small H5 denominator is the causal derivative path; the complete raw metrics and conflicting anchors remain in the TSV files.

The selected raw values themselves, not only their ratios, are identical in
microbatches 0 and 3: H5 Z/S denominators `0.392857/1.214110`, followed by H6
P/W/Zraw/Sraw RMS `1.170614/78.741806/73.096992/79.712311`. Together with the
same byte-104 root, this shows that both large physical components repeat one
state-specific central trajectory. It still does not distinguish the boundary
normalization derivative from upstream Q/K/V, shared decoder, or tied-parameter
use-site effects.
