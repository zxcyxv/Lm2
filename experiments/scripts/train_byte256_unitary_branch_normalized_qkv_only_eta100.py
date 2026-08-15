"""End-to-end ETA run for the selected packed-QKV eager-memory path."""
import train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m as run


run.base.FUSE_BRANCH_MEMORY_KERNEL = False
run.base.REPORT_STEPS = frozenset((100,))


if __name__ == "__main__":
    run.base.main()
