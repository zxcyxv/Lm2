"""Short end-to-end throughput run for the current fast H16 trainer."""
import train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m as run


run.base.REPORT_STEPS = frozenset((100,))


if __name__ == "__main__":
    run.base.main()
