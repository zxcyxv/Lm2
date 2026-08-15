"""Compare the running 121M time-varying-scan job against the 13M baseline
record (EXP-20260803-byte256-unitary-time-varying-scan-h16-1000step-rtx5090)
-- same architecture (K1DecoderAblationLM + ComplexSelfPredictedKVTransition,
time-varying-scan rollout, H16/stride16), only width/peak-lr differ
(width=1344,lr=3e-4 vs width=4352,lr=1e-4). Run with no args; re-run to
refresh as the live log grows.
"""
import csv
import re
import sys
from pathlib import Path

LIVE_LOG = Path(
    "/tmp/claude-0/-workspace-Lm2/e54d4663-b6b2-449c-826b-2336543e672d/scratchpad/train_time_varying_scan.log"
)
BASELINE_METRICS = Path(
    "experiments/records/EXP-20260803-byte256-unitary-time-varying-scan-h16-1000step-rtx5090/metrics.tsv"
)

LINE_RE = re.compile(
    r"step=\s*(\d+)/\d+ ce=([\d.]+) gnorm=([\d.]+) block_nll=([\d.]+) h1=([\d.]+) h16=([\d.]+)"
)
STEP0_RE = re.compile(
    r"step=0/\d+ block_nll=([\d.]+) h1=([\d.]+) h16=([\d.]+)"
)


def load_baseline() -> dict[int, dict[str, float]]:
    rows = {}
    with BASELINE_METRICS.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            step = int(row["step"])
            rows[step] = {
                "train_ce": float(row["train_ce"]) if row["train_ce"] != "nan" else float("nan"),
                "gradient_norm": float(row["gradient_norm"]) if row["gradient_norm"] != "nan" else float("nan"),
                "h1_nll": float(row["h1_nll"]),
                "h16_nll": float(row["h16_nll"]),
                "block_nll": float(row["block_nll"]),
            }
    return rows


def load_live() -> dict[int, dict[str, float]]:
    rows = {}
    if not LIVE_LOG.exists():
        return rows
    for line in LIVE_LOG.read_text().splitlines():
        match = LINE_RE.search(line)
        if match:
            step, ce, gnorm, block_nll, h1, h16 = match.groups()
            rows[int(step)] = {
                "train_ce": float(ce),
                "gradient_norm": float(gnorm),
                "h1_nll": float(h1),
                "h16_nll": float(h16),
                "block_nll": float(block_nll),
            }
            continue
        match0 = STEP0_RE.search(line)
        if match0:
            block_nll, h1, h16 = match0.groups()
            rows[0] = {
                "train_ce": float("nan"),
                "gradient_norm": float("nan"),
                "h1_nll": float(h1),
                "h16_nll": float(h16),
                "block_nll": float(block_nll),
            }
    return rows


def main() -> None:
    baseline = load_baseline()
    live = load_live()
    steps = sorted(set(baseline) | set(live))

    header = f"{'step':>5} | {'121M ce':>9} {'121M gn':>9} {'121M h1':>9} {'121M h16':>9} {'121M blk':>9} | {'13M ce':>9} {'13M gn':>9} {'13M h1':>9} {'13M h16':>9} {'13M blk':>9}"
    print(header)
    print("-" * len(header))
    for step in steps:
        r121 = live.get(step)
        r13 = baseline.get(step)

        def fmt(row, key):
            return f"{row[key]:9.4f}" if row and row[key] == row[key] else " " * 9

        print(
            f"{step:>5} | {fmt(r121,'train_ce')} {fmt(r121,'gradient_norm')} {fmt(r121,'h1_nll')} {fmt(r121,'h16_nll')} {fmt(r121,'block_nll')} | "
            f"{fmt(r13,'train_ce')} {fmt(r13,'gradient_norm')} {fmt(r13,'h1_nll')} {fmt(r13,'h16_nll')} {fmt(r13,'block_nll')}"
        )

    latest_step = max(live) if live else 0
    print(f"\nlatest live step observed: {latest_step}")


if __name__ == "__main__":
    main()
