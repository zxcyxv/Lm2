# Step-1000 causal dependency audit

- Checkpoint step: 1000
- Examples: 16

## Interventions

| intervention | max abs | mean relative MSE |
|---|---:|---:|
| short_vs_joint_h1_hidden | 0 | 0 |
| short_vs_joint_h1_logits | 0 | 0 |
| change_s2_prefix_plus_h1_hidden | 0 | 0 |
| change_s2_h1_logits | 0 | 0 |
| change_s2_h2_hidden | 25.0808239 | 0.99924612 |
| change_s2_h2_logits | 15.7399654 | 13.2898808 |
| change_s1_prefix_hidden | 0 | 0 |
| change_s1_h1_hidden | 23.9773521 | 0.999890447 |
| change_s1_h2_hidden | 0.045249857 | 0.00211119931 |
| change_s1_h2_logits | 0.660796165 | 0.000570398115 |

## Autograd dependency probes

| gradient | L2 norm |
|---|---:|
| d_h1_probe_d_s1_norm | 40.7090378 |
| d_h1_probe_d_s2_norm | 0 |
| d_h2_probe_d_s1_norm | 1.79802895 |
| d_h2_probe_d_s2_norm | 34.7537994 |