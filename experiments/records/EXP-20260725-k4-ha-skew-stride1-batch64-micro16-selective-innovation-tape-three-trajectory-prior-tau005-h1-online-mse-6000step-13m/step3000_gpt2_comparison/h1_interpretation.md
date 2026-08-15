# Step-3000 h1 comparison

| method | NLL | top-1 accuracy | target-free policy |
|---|---:|---:|:---:|
| gpt2_ar | 4.934292 | 0.216003 | yes |
| latent_path_mean | 4.575760 | 0.253286 | no |
| latent_posterior | 4.044487 | 0.301625 | no |
| latent_oracle | 4.044334 | 0.301666 | no |
| latent_confidence | 4.578397 | 0.266846 | yes |
| latent_prior | 4.591133 | 0.252777 | yes |

Posterior and oracle rows use gold labels for branch assignment and are diagnostic upper bounds. Confidence and prior rows are deployable target-free selectors.