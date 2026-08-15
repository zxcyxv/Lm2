# Step-4000 h1 comparison

| method | NLL | top-1 accuracy | target-free policy |
|---|---:|---:|:---:|
| gpt2_ar | 4.934292 | 0.216003 | yes |
| latent_path_mean | 4.474244 | 0.266256 | no |
| latent_posterior | 3.925800 | 0.317386 | no |
| latent_oracle | 3.925674 | 0.317383 | no |
| latent_confidence | 4.477749 | 0.279297 | yes |
| latent_prior | 4.486582 | 0.263031 | yes |

Posterior and oracle rows use gold labels for branch assignment and are diagnostic upper bounds. Confidence and prior rows are deployable target-free selectors.
