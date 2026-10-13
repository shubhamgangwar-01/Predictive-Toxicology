"""Benchmark descriptor models vs. graph neural networks under scaffold splits.

Usage:
    python scripts/run_benchmark.py                       # everything (4 datasets x 9 models x 3 seeds)
    python scripts/run_benchmark.py --datasets esol bbbp --models RF:ecfp GCN --seeds 0

Results are appended to results/benchmark_raw.csv (runs already present are skipped, so the
script can be interrupted and resumed). Seed-0 models are saved to models/<dataset>/ for the app.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from molrepbench.config import DATASETS, DESCRIPTOR_MODELS, GNN_MODELS, MODELS, RESULTS, SEEDS
from molrepbench.pipeline import prepare, train_and_evaluate

RAW = RESULTS / "benchmark_raw.csv"
PREDS = RESULTS / "preds"

# Fixed column schema. Classification and regression runs produce different metric columns;
# appending them to one CSV without a common schema silently shifts every later row under the
# wrong header, so each frame is reindexed to this before writing.
COLUMNS = ["dataset", "model", "features", "protocol", "seed", "task",
           "roc_auc", "pr_auc", "mcc", "pos_rate", "threshold",
           "rmse", "mae", "r2", "n_train", "n_test", "train_time_s"]


def all_model_keys():
    return [f"{f}:{x}" for f, x in DESCRIPTOR_MODELS] + [f"{g}:graph" for g in GNN_MODELS]


def parse_key(key):
    if ":" in key:
        fam, feat = key.split(":")
    else:
        fam, feat = key, "graph"
    return fam, feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=all_model_keys())
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--raw", default=str(RAW), help="CSV to append per-task results to")
    ap.add_argument("--protocol", default="balanced", choices=["balanced", "deterministic"],
                    help="scaffold-split protocol; must be fixed across a whole run")
    args = ap.parse_args()
    raw = Path(args.raw)
    torch.set_num_threads(args.threads)

    RESULTS.mkdir(exist_ok=True)
    PREDS.mkdir(exist_ok=True)
    done = set()
    if raw.exists():
        prev = pd.read_csv(raw)
        done = set(zip(prev.dataset, prev.model, prev.features, prev.seed))

    for name in args.datasets:
        bundle = prepare(name)
        for key in args.models:
            fam, feat = parse_key(key)
            for seed in args.seeds:
                if (name, fam, feat, seed) in done:
                    continue
                t0 = time.time()
                model, scaler, rows, preds = train_and_evaluate(bundle, fam, feat, seed,
                                                                 args.protocol)
                df = pd.DataFrame(rows).reindex(columns=COLUMNS)
                df.to_csv(raw, mode="a", header=not raw.exists(), index=False)
                np.savez(PREDS / f"{name}_{fam}_{feat}_s{seed}_{args.protocol}.npz", **preds)
                if seed == 0 and args.protocol == "balanced":
                    out = MODELS / name
                    out.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model, "scaler": scaler, "family": fam, "features": feat},
                               out / f"{fam}_{feat}.pt")
                key_metric = "roc_auc" if df["roc_auc"].notna().any() else "rmse"
                print(f"[{name}] {fam:11s} {feat:5s} seed={seed} "
                      f"{key_metric}={df[key_metric].mean():.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
