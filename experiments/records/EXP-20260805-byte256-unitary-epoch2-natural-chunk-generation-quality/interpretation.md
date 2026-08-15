# Natural-chunk 128-byte generation interpretation

All three epoch-2 checkpoints eventually degenerate under 128 bytes of greedy
self-generation. This audit measures long closed-loop stability and is not a
fair substitute for their matched sixteen-byte block scores.

H1 remained the most text-like but repeated generic templates such as
`the second season`; H4 degraded into spaces, `@`, and short `a/t` motifs;
scan became almost entirely spaces after its initial characters. The scan
longest identical-byte run averaged 105.7 of 128 bytes.

This long-generation collapse does not establish that scan's block NLL is a
margin artifact. The matched-block rank and temperature audit addresses that
question directly.
