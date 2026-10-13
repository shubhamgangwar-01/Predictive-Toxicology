"""Applicability domain (AD): is a query compound inside the chemical space the model was trained on?

Two complementary checks:
1. Structural similarity - mean Tanimoto similarity (ECFP4) of the query to its k=5 nearest
   training neighbours. Thresholds are percentiles of the same statistic computed leave-one-out
   on the training set itself, so they adapt to how diverse each dataset is.
2. Descriptor range - fraction of RDKit descriptors falling outside the min/max seen in training.
"""
from __future__ import annotations

import numpy as np
from rdkit import DataStructs

from .features import ecfp_bitvect, mol_from_smiles

K = 5
LEVELS = ("In domain", "Borderline", "Out of domain")


class ApplicabilityDomain:
    def __init__(self, k: int = K, low_pct: float = 5.0, mid_pct: float = 25.0, max_out_of_range: float = 0.10):
        self.k, self.low_pct, self.mid_pct, self.max_oor = k, low_pct, mid_pct, max_out_of_range

    def fit(self, train_smiles, desc_min=None, desc_max=None):
        self.fps = [ecfp_bitvect(mol_from_smiles(s)) for s in train_smiles]
        loo = []
        for i, fp in enumerate(self.fps):
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, self.fps))
            sims[i] = -1.0  # leave-one-out
            loo.append(np.sort(sims)[-self.k:].mean())
        loo = np.array(loo)
        self.t_low = float(np.percentile(loo, self.low_pct))
        self.t_mid = float(np.percentile(loo, self.mid_pct))
        self.desc_min, self.desc_max = desc_min, desc_max
        return self

    def knn_similarity(self, smiles_list):
        out = []
        for s in smiles_list:
            sims = np.array(DataStructs.BulkTanimotoSimilarity(ecfp_bitvect(mol_from_smiles(s)), self.fps))
            out.append(np.sort(sims)[-self.k:].mean())
        return np.array(out)

    def nearest_neighbours(self, smiles, train_smiles, n=3):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(ecfp_bitvect(mol_from_smiles(smiles)), self.fps))
        idx = np.argsort(-sims)[:n]
        return [(train_smiles[i], float(sims[i])) for i in idx]

    def out_of_range_fraction(self, desc_row):
        if self.desc_min is None or desc_row is None:
            return 0.0
        d = np.where(np.isnan(desc_row), self.desc_min, desc_row)
        return float(np.mean((d < self.desc_min) | (d > self.desc_max)))

    def level(self, sim: float, oor: float = 0.0) -> str:
        if sim < self.t_low:
            lvl = 2
        elif sim < self.t_mid:
            lvl = 1
        else:
            lvl = 0
        if oor > self.max_oor:
            lvl = max(lvl, 1 if oor <= 2 * self.max_oor else 2)
        return LEVELS[lvl]

    def assess(self, smiles_list, desc=None):
        sims = self.knn_similarity(smiles_list)
        oors = [self.out_of_range_fraction(desc[i] if desc is not None else None) for i in range(len(sims))]
        return [{"knn_sim": float(s), "out_of_range": o, "level": self.level(s, o)} for s, o in zip(sims, oors)]
