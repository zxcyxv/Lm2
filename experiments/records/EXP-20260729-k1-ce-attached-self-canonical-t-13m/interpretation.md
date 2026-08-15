# Training interpretation

The registered 1000-step run completed in 490.1 seconds. Validation one-token
NLL fell from `9.2021` at step 1 to `4.4673` at step 1000.

Attached T regression reduced paired self-canonical relative MSE from the raw
q value `0.020063` to `0.008002`, a ratio of `0.3989`. This passes the
registered MSE ratio gate.

Decoder semantics did not follow the Euclidean improvement. Decoding the
corrected state preserved the raw q-selected token on only `0.1210` of
validation positions at step 1000, failing the registered 0.90 supporting
criterion. Corrected-state gold NLL was `7.7589`, versus raw-q gold NLL
`4.4673`.

The run therefore establishes that attached T can lower self-canonical
Euclidean error while jointly shaping K and the encoder, but it does not
establish that `T(q)q` is a usable decoder state. Raw q remains the only
trained token readout path.
