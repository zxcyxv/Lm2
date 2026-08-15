"""Run H1 complex-KV regression with an EMA stop-gradient target."""
from pathlib import Path

import train_byte256_complex_self_predicted_kv_h1_online_mse_ce_13m as base


base.EXPERIMENT_ID = (
    "EXP-20260801-byte256-complex-self-predicted-kv-"
    "h1-ema-sg-mse-ce-13m"
)
base.DEFAULT_RECORD = Path("experiments/records") / base.EXPERIMENT_ID
base.DEFAULT_OUTPUT = Path("outputs/experiments") / base.EXPERIMENT_ID
base.TEMPERED_READOUT = False
base.RESIDUAL_PRIOR = False
base.DETACH_LATENT_TARGET = True
base.EMA_TARGET_DECAY = 0.99
base.EMA_WARM_START_STEPS = 100
base.EVALUATE_WITH_EMA = True
base.REPORT_ONLINE_WITH_EMA = True
base.EXPECTED_PARAMETERS = 13_216_352
base.MIN_H2_H4_TOKEN_AGREEMENT = 0.261149088542
base.MIN_MEAN_AR_STATE_COSINE = 0.75
base.MIN_H1_POSTERIOR_TARGET_ACCURACY = 0.109619140625
base.STEP100_REFERENCE = {
    "val_h1_validation_nll": 2.3903664015233517,
    "val_h1_latent_relative_mse": 0.5923016034066677,
    "val_mean_overall_cosine": 0.5954537440303511,
    "val_h2_h4_token_agreement": 0.3614095052083333,
}
base.STEP100_REFERENCE_TOLERANCE = 1e-5


if __name__ == "__main__":
    base.main()
