"""Featurise-once caching and a single train/evaluate routine shared by all models."""
from __future__ import annotations

import time

import numpy as np
import torch

from .config import DATA_PROCESSED, DATASETS
from .data import load_dataset, scaffold_split
from .features import DescriptorScaler, ecfp, mol_to_graph, rdkit_descriptors
from .metrics import best_mcc_threshold, classification_metrics, regression_metrics
from .models import make_model


def prepare(name: str) -> dict:
    """Load + featurise a dataset (cached to data/processed/<name>.pt)."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cache = DATA_PROCESSED / f"{name}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    spec = DATASETS[name]
    df = load_dataset(name)
    smiles = df["smiles"].tolist()
    Y = df[list(spec.tasks)].to_numpy(dtype=np.float32)
    bundle = {
        "name": name,
        "smiles": smiles,
        "Y": Y,
        "ecfp": ecfp(smiles),
        "rdkit": rdkit_descriptors(smiles),
        "graphs": [mol_to_graph(s, y) for s, y in zip(smiles, Y)],
    }
    torch.save(bundle, cache)
    return bundle


def get_split(bundle, seed, protocol="balanced"):
    return scaffold_split(bundle["smiles"], seed=seed, protocol=protocol)


def features_for(bundle, feat, tr, va, te):
    """Return (X_train, X_val, X_test, scaler) for the chosen representation."""
    if feat == "ecfp":
        X = bundle["ecfp"]
        return X[tr], X[va], X[te], None
    if feat == "rdkit":
        scaler = DescriptorScaler().fit(bundle["rdkit"][tr])
        R = scaler.transform(bundle["rdkit"])
        return R[tr], R[va], R[te], scaler
    if feat == "graph":
        G = bundle["graphs"]
        return [G[i] for i in tr], [G[i] for i in va], [G[i] for i in te], None
    raise ValueError(feat)


def train_and_evaluate(bundle, family, feat, seed, protocol="balanced"):
    """Train one model on one scaffold split; return (model, scaler, per-task metric rows, test preds)."""
    spec = DATASETS[bundle["name"]]
    tr, va, te = get_split(bundle, seed, protocol)
    Y = bundle["Y"]
    Xtr, Xva, Xte, scaler = features_for(bundle, feat, tr, va, te)
    t0 = time.time()
    model = make_model(family, spec.task_type, Y.shape[1], seed).fit(Xtr, Y[tr], Xva, Y[va])
    train_time = time.time() - t0
    Pva, Pte = model.predict(Xva), model.predict(Xte)

    rows, thresholds = [], []
    for t, task in enumerate(spec.tasks):
        mv, mt = ~np.isnan(Y[va, t]), ~np.isnan(Y[te, t])
        if spec.task_type == "classification":
            thr = best_mcc_threshold(Y[va][mv, t], Pva[mv, t])
            thresholds.append(thr)
            m = classification_metrics(Y[te][mt, t], Pte[mt, t], thr)
            m["threshold"] = thr
        else:
            m = regression_metrics(Y[te][mt, t], Pte[mt, t])
        m.update(dataset=bundle["name"], model=family, features=feat, seed=seed,
                 protocol=protocol, task=task,
                 n_train=int((~np.isnan(Y[tr, t])).sum()), n_test=int(mt.sum()), train_time_s=train_time)
        rows.append(m)
    model.thresholds = np.array(thresholds) if thresholds else None
    return model, scaler, rows, {"test_idx": te, "pred": Pte}
