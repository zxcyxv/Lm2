"""Continue branch-normalized H16 training to step 6000 with microbatch 64."""
import train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m as run


run.base.REPORT_STEPS = frozenset(
    (1001, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000)
)


if __name__ == "__main__":
    run.base.main()
