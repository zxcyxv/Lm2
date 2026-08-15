"""Run the preregistered full-innovation complex KV experiment."""
from pathlib import Path

import train_byte256_complex_self_predicted_kv_h1_online_mse_ce_13m as base


base.EXPERIMENT_ID = (
    "EXP-20260801-byte256-complex-self-predicted-kv-full-innovation-"
    "h1-online-mse-ce-13m"
)
base.DEFAULT_RECORD = Path("experiments/records") / base.EXPERIMENT_ID
base.DEFAULT_OUTPUT = Path("outputs/experiments") / base.EXPERIMENT_ID
base.TEMPERED_READOUT = False
base.RESIDUAL_PRIOR = False
base.EXPECTED_PARAMETERS = 13_216_352


if __name__ == "__main__":
    base.main()
