"""Download, standardise and featurise all four datasets (cached in data/processed)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from molrepbench.config import DATASETS
from molrepbench.pipeline import get_split, prepare

if __name__ == "__main__":
    for name in DATASETS:
        t0 = time.time()
        b = prepare(name)
        tr, va, te = get_split(b, 0)
        Y = b["Y"]
        print(f"{name:8s} n={len(b['smiles']):5d} tasks={Y.shape[1]:2d} "
              f"split={len(tr)}/{len(va)}/{len(te)} "
              f"pos_rate={np.nanmean(Y):.3f} ({time.time()-t0:.0f}s)")
