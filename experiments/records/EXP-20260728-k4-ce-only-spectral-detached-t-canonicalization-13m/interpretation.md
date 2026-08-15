# Running interpretation

## Step 100

The token-level AR reference had NLL 6.916388 (PPL 1008.67), while
four-token block-boundary teacher forcing had NLL 7.079090 (PPL 1186.89).
The block/AR PPL ratio was 1.177 and top-1 agreement was 0.613.

The detached T corrector reduced paired canonical-state relative MSE from
0.218922 to 0.157024, a ratio of 0.745. This did not preserve decoder
semantics: only 0.307 of corrected states retained the proposal-selected
token.

## Step 500

The token-level AR reference improved to NLL 5.420738 (PPL 226.05).
Four-token block-boundary teacher forcing reached NLL 6.307470
(PPL 548.65), a +0.886732 NLL gap and 2.427 PPL ratio. Horizon PPL was
224.68, 573.15, 790.38, and 890.27. The h1 path therefore remains aligned
with the AR reference, while the block-frozen K orbit degrades sharply after
the first token.

The detached T corrector reduced paired state MSE from 0.692888 to 0.512300,
a ratio of 0.743. Absolute errors increased as the CE-only base geometry
moved, while the relative gain stayed near 26%. Selected-token preservation
fell to 0.0395 and corrected-state gold NLL was 8.9967. Thus the present T
learns an Euclidean correction but not a decoder-semantic retraction.

## Scope

Both PPL audits decode the raw K orbit. The current T has no independent
eigenbasis: it applies input-conditioned complex-diagonal gains in the
detached K basis and is not used to generate the current block. These results
do not evaluate the proposed architecture in which a separately MSE-trained
T basis and its powers form the parallel generation orbit.
