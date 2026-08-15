"""Train EMA-SG H1 closure with a full post-update memory reread."""
from pathlib import Path

import train_byte256_complex_self_predicted_kv_h1_online_mse_ce_13m as base


base.EXPERIMENT_ID = (
    "EXP-20260801-byte256-complex-self-predicted-kv-"
    "post-update-full-read-h1-ema-sg-mse-ce-13m"
)
base.DEFAULT_RECORD = Path("experiments/records") / base.EXPERIMENT_ID
base.DEFAULT_OUTPUT = Path("outputs/experiments") / base.EXPERIMENT_ID
base.TEMPERED_READOUT = False
base.RESIDUAL_PRIOR = False
base.RECURRENT_HIDDEN_MODE = "post-update-full-read"
base.DETACH_LATENT_TARGET = True
base.EMA_TARGET_DECAY = 0.99
base.EMA_WARM_START_STEPS = 100
base.EVALUATE_WITH_EMA = True
base.REPORT_ONLINE_WITH_EMA = True
base.REPORT_STEPS = frozenset(
    (1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
)
base.EXPECTED_PARAMETERS = 13_216_352
base.MIN_H2_H4_TOKEN_AGREEMENT = 0.272135416667
base.MIN_MEAN_AR_STATE_COSINE = 0.0
base.MIN_H1_POSTERIOR_TARGET_ACCURACY = None
base.STEP100_REFERENCE = None


if __name__ == "__main__":
    base.main()
