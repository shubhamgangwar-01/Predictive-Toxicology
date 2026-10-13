"""Audit every split for train/test leakage.

A scaffold split guarantees distinct Bemis-Murcko frameworks, but two different frameworks can still
belong to near-identical molecules (one added methyl, a changed ring heteroatom). If a large share of
test molecules have a near-duplicate in training, high scores reflect memorisation rather than
generalisation — so this is checked rather than assumed.

Writes results/leakage_audit.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from rdkit import DataStructs

from molrepbench.config import DATASETS, RESULTS
from molrepbench.features import ecfp_bitvect, mol_from_smiles
from molrepbench.pipeline import get_split, prepare

NEAR_DUPLICATE = 0.95   # Tanimoto above this is effectively the same compound
HIGH = 0.70             # a close analogue


def audit(name, seed, protocol):
    b = prepare(name)
    tr, va, te = get_split(b, seed, protocol)
    fps = [ecfp_bitvect(mol_from_smiles(s)) for s in b["smiles"]]
    tr_fps = [fps[i] for i in tr]
    sims = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], tr_fps)) for i in te])
    return {
        "dataset": name, "protocol": protocol, "seed": seed,
        "n_test": len(te),
        "mean_max_sim": round(float(sims.mean()), 3),
        "median_max_sim": round(float(np.median(sims)), 3),
        f"pct_above_{NEAR_DUPLICATE}": round(float((sims >= NEAR_DUPLICATE).mean() * 100), 1),
        f"pct_above_{HIGH}": round(float((sims >= HIGH).mean() * 100), 1),
    }


def main():
    rows = []
    for name in DATASETS:
        for protocol in ("balanced", "deterministic"):
            seeds = (0, 1, 2) if protocol == "balanced" else (0,)
            for seed in seeds:
                r = audit(name, seed, protocol)
                rows.append(r)
                print(f"{r['dataset']:8s} {r['protocol']:13s} seed={r['seed']} "
                      f"mean_max_sim={r['mean_max_sim']:.3f} "
                      f"near-dup={r[f'pct_above_{NEAR_DUPLICATE}']:.1f}% "
                      f"close={r[f'pct_above_{HIGH}']:.1f}%", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "leakage_audit.csv", index=False)
    print(f"\nWrote {RESULTS / 'leakage_audit.csv'}")


if __name__ == "__main__":
    main()
