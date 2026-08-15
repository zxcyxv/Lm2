# AR versus block-4 interpretation

## Result

Neither greedy method produces acceptable 64-token prose at step 1000.
Both collapse into frequent high-level templates such as repeated section
markers, entities, and phrases. Block-4 is qualitatively worse: its samples
contain more broken syntax, duplicated function words, malformed word pieces,
and four-token-local repetition.

## Metric scope

Block-4 has slightly lower aggregate immediate repetition than AR
(`0.210` versus `0.219`) and higher distinct-2 (`0.403` versus `0.362`).
Those metrics do not establish better sentence quality. Its distinct-1 is
lower (`0.181` versus `0.231`) and within-block repetition is higher
(`0.267` versus `0.217`). The very low block-boundary repetition (`0.027`)
shows that most block-4 repetition is concentrated inside jointly decoded
blocks rather than at re-grounding boundaries.

## Scope and conclusion

This is a five-prompt greedy audit of the CE-only seed-1337 checkpoint, not a
distributional generation evaluation. On this evidence, four-token decoding
does not preserve AR sentence quality. The trajectory objective improves
teacher-forced validation metrics but does not by itself yield a usable
target-free trajectory selector; this audit therefore uses the clean orbit
and cannot claim inference-time benefit from gold-dependent responsibilities.
