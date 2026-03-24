"""
Frozen-baseline entrypoint for CLOOME + ChemBERT concat model.

This wrapper reuses train_cloome_chembert_concat.py while enforcing:
  - CLOOME backbone is fully frozen for the whole run
  - backbone LR = 0.0

It also sets frozen-specific output paths unless the user provides their own.
"""

import sys

from train_cloome_chembert_concat import main as train_main


def _has_arg(flag: str) -> bool:
    return flag in sys.argv[1:]


if __name__ == "__main__":
    forced_args = [
        "--freeze-epochs", "1000000",  # effectively keep backbone frozen for full training
        "--backbone-lr", "0.0",
    ]

    # Use frozen-specific outputs unless caller overrides them.
    if not _has_arg("--output-model"):
        forced_args.extend(["--output-model", "results/checkpoints/cloome_concat_frozen.pt"])
    if not _has_arg("--output-metrics"):
        forced_args.extend(["--output-metrics", "results/metrics/cloome_concat_frozen_metrics.json"])

    sys.argv = [sys.argv[0], *sys.argv[1:], *forced_args]
    train_main()

