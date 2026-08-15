"""Compare the MVA (shared B/C) 121M run against the per-head 121M run.

Both runs use identical seeds, data order, schedule and width; the only
difference is whether the query/key projections are shared across heads
(Mamba's multi-value-attention layout) or produced per head.  ``pgnorm`` is
the raw ``hidden_phase`` gradient norm measured before clipping -- the
parameter group that carried 93% of the per-head run's squared gradient
norm at step 1000.
"""
from __future__ import annotations

import csv
from pathlib import Path

RUNS = (
    ("per-head", Path("outputs/time_varying_scan_mha/metrics.tsv")),
    ("MVA", Path("outputs/time_varying_scan_mva/metrics.tsv")),
)
FIELDS = ("train_ce", "gradient_norm", "phase_gradient_norm", "val_block_nll")
SHORT = {
    "train_ce": "ce",
    "gradient_norm": "gnorm",
    "phase_gradient_norm": "pgnorm",
    "val_block_nll": "block_nll",
}


def load(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    rows = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[int(row["step"])] = {
                field: float(row[field])
                for field in FIELDS
                if row.get(field) not in (None, "")
            }
    return rows


def main() -> None:
    loaded = [(label, load(path)) for label, path in RUNS]
    steps = sorted({step for _, rows in loaded for step in rows})
    if not steps:
        print("no metrics yet")
        return

    width = 4 + 3 + sum(11 * len(FIELDS) + 3 for _ in loaded)
    header = f"{'step':>5} | " + " | ".join(
        " ".join(f"{label[:4]}.{SHORT[f]:>9}" for f in FIELDS) for label, _ in loaded
    )
    print(header)
    print("-" * len(header))
    for step in steps:
        cells = []
        for _, rows in loaded:
            row = rows.get(step)
            cells.append(
                " ".join(
                    f"{row[f]:14.4f}" if row and f in row else " " * 14
                    for f in FIELDS
                )
            )
        print(f"{step:>5} | " + " | ".join(cells))

    print()
    for label, rows in loaded:
        if not rows:
            print(f"{label}: (no rows yet)")
            continue
        last = max(rows)
        row = rows[last]
        share = (
            row["phase_gradient_norm"] ** 2 / row["gradient_norm"] ** 2
            if row.get("gradient_norm")
            else float("nan")
        )
        print(
            f"{label:>9} @ step {last}: block_nll={row.get('val_block_nll', float('nan')):.4f} "
            f"gnorm={row.get('gradient_norm', float('nan')):.3f} "
            f"pgnorm={row.get('phase_gradient_norm', float('nan')):.3f} "
            f"-> hidden_phase is {share:.1%} of the squared gradient norm"
        )


if __name__ == "__main__":
    main()
