# H2 feedback-mechanism stratification

The hard re-entry penalty is concentrated in H1-wrong trajectories; correct hard tokens instead restore the literal teacher prefix.

| H1 stratum | examples | continuous H2 NLL | discrete H2 NLL | discrete - continuous | state relative L2 |
|---|---:|---:|---:|---:|---:|
| h1_correct | 210 | 6.259794 | 5.648711 | -0.611083 | 0.650791 |
| h1_wrong | 755 | 6.837961 | 8.471115 | 1.633154 | 0.723388 |

This is a post-hoc conditional diagnostic. It separates correct from wrong hard decisions, but does not replace a matched training intervention for causal attribution.
