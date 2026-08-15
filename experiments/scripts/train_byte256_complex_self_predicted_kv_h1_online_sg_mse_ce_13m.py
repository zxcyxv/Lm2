"""Run the online stop-gradient H1 latent-target complex KV experiment."""
from pathlib import Path

import train_byte256_complex_self_predicted_kv_h1_online_mse_ce_13m as base


base.EXPERIMENT_ID = (
    "EXP-20260801-byte256-complex-self-predicted-kv-"
    "h1-online-sg-mse-ce-13m"
)
base.DEFAULT_RECORD = Path("experiments/records") / base.EXPERIMENT_ID
base.DEFAULT_OUTPUT = Path("outputs/experiments") / base.EXPERIMENT_ID
base.TEMPERED_READOUT = False
base.RESIDUAL_PRIOR = False
base.DETACH_LATENT_TARGET = True
base.EXPECTED_PARAMETERS = 13_216_352
base.MIN_H2_H4_TOKEN_AGREEMENT = 0.261149088542
base.MIN_MEAN_AR_STATE_COSINE = 0.75
base.MIN_H1_POSTERIOR_TARGET_ACCURACY = 0.109619140625


if __name__ == "__main__":
    base.main()
