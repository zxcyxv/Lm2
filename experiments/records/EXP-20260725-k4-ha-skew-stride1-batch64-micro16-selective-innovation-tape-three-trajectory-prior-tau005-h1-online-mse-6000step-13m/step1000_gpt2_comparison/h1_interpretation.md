# Step-1000 h1 comparison

| method | NLL | top-1 accuracy | target-free policy |
|---|---:|---:|:---:|
| gpt2_ar | 4.934292 | 0.216003 | yes |
| latent_path_mean | 5.231960 | 0.194163 | no |
| latent_posterior | 4.778155 | 0.229493 | no |
| latent_oracle | 4.778026 | 0.229431 | no |
| latent_confidence | 5.233169 | 0.203827 | yes |
| latent_prior | 5.217033 | 0.194000 | yes |

Posterior and oracle rows use gold labels for branch assignment and are diagnostic upper bounds. Confidence and prior rows are deployable target-free selectors.
