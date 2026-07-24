# Step-1000 matched hB-noise decoder audit

- Checkpoint step: 1000
- Examples: 256
- Random draws: 8

| variant | input_relative_mse | input_cosine | decoded_relative_mse | decoded_cosine | local_l2_gain | accuracy | nll | target_rank | target_margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| actual_kha | 0.021729387 | 0.98933196 | 18.016426 | 0.22874682 | 1.180478 | 0.2265625 | 4.6697912 | 207.87891 | -2.4337561 |
| matched_norm_noise | 0.021729387 | 0.98933458 | 46.239291 | 0.13361979 | 1.8709469 | 0.44189453 | 5.6722437 | 25.01709 | -0.15199913 |
| matched_geometry_noise | 0.021729387 | 0.98933197 | 43.231358 | 0.13598201 | 1.8128054 | 0.44628906 | 5.6243771 | 22.131348 | -0.13403126 |

- Geometry-matched / actual decoded relative-MSE ratio: 2.3995524
- Gold hB accuracy: 1
- Gold embedding roundtrip max: 8.7544322e-07
- Geometry distance-match max residual: 4.7683716e-07
- Geometry state-norm-match max residual: 1.9073486e-06