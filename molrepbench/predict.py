"""Inference service: load the trained seed-0 models and score a SMILES across every endpoint.

This is the layer the Streamlit app talks to. It owns:
  * model loading (cached),
  * per-endpoint probability / value prediction with the frozen MCC threshold,
  * the applicability-domain confidence flag,
  * SHAP and attention attributions for the chosen molecule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from .applicability import ApplicabilityDomain
from .config import DATASETS, MODELS, TASK_SHORT
from .data import standardize_smiles
from .features import ecfp, mol_to_graph, rdkit_descriptors
from .models import DEVICE
from .pipeline import get_split, prepare

# Preference order per dataset when several trained models are available. The app picks the first
# that exists on disk, so it still works from a partial benchmark run. XGB_ecfp is preferred where
# it is competitive because it is the only configuration supporting exact TreeSHAP attribution.
PREFERENCE = {
    "tox21": ["XGB_ecfp", "RF_ecfp", "XGB_rdkit", "RF_rdkit", "AttentiveFP_graph", "GAT_graph",
              "GCN_graph", "DNN_ecfp", "DNN_rdkit"],
    "clintox": ["XGB_ecfp", "RF_ecfp", "XGB_rdkit", "RF_rdkit", "AttentiveFP_graph", "GAT_graph",
                "GCN_graph", "DNN_ecfp", "DNN_rdkit"],
    "bbbp": ["XGB_ecfp", "RF_ecfp", "XGB_rdkit", "RF_rdkit", "AttentiveFP_graph", "GAT_graph",
             "GCN_graph", "DNN_ecfp", "DNN_rdkit"],
    "esol": ["AttentiveFP_graph", "XGB_rdkit", "RF_rdkit", "GAT_graph", "GCN_graph", "DNN_rdkit",
             "XGB_ecfp", "RF_ecfp", "DNN_ecfp"],
}


def _resolve(dataset: str) -> str | None:
    """First preferred model actually present on disk for this dataset."""
    for key in PREFERENCE.get(dataset, []):
        if (MODELS / dataset / f"{key}.pt").exists():
            return key
    found = sorted((MODELS / dataset).glob("*.pt")) if (MODELS / dataset).is_dir() else []
    return found[0].stem if found else None


class _ModelChoice(dict):
    """Resolves lazily so the app reflects whatever the benchmark has finished training."""

    def __missing__(self, dataset):
        return _resolve(dataset)

    def get(self, dataset, default=None):
        return self[dataset] or default


DEFAULT_MODEL = _ModelChoice()


@dataclass
class EndpointResult:
    dataset: str
    task: str
    label: str
    task_type: str
    value: float                 # probability (classification) or predicted value (regression)
    threshold: float | None = None
    call: str = ""               # "Flagged" / "Clear" / formatted value


@dataclass
class ScreeningResult:
    query: str
    canonical: str
    endpoints: list = field(default_factory=list)
    confidence: dict = field(default_factory=dict)
    neighbours: list = field(default_factory=list)


@lru_cache(maxsize=None)
def load_model(dataset: str, key: str):
    """Load a trained checkpoint onto whatever device this machine has.

    `map_location=DEVICE` matters because the benchmark is trained on a Colab GPU while the app
    normally runs on a laptop CPU: without it, torch.load tries to restore CUDA tensors and raises.
    The network submodule is also moved explicitly, since map_location only relocates the tensors
    unpickled from the file, not the module's own device bookkeeping.
    """
    path = MODELS / dataset / f"{key}.pt"
    if not path.exists():
        return None
    bundle = torch.load(path, weights_only=False, map_location=DEVICE)
    net = getattr(bundle.get("model"), "net", None)
    if isinstance(net, torch.nn.Module):
        net.to(DEVICE).eval()
    return bundle


@lru_cache(maxsize=None)
def load_domain(dataset: str):
    """Applicability domain fitted on the seed-0 training split of this dataset."""
    bundle = prepare(dataset)
    tr, _, _ = get_split(bundle, 0)
    train_smiles = [bundle["smiles"][i] for i in tr]
    bundleload = load_model(dataset, DEFAULT_MODEL[dataset])
    scaler = bundleload.get("scaler") if bundleload else None
    desc_min = desc_max = None
    if scaler is None:
        from .features import DescriptorScaler
        scaler = DescriptorScaler().fit(bundle["rdkit"][tr])
    desc_min, desc_max = scaler.min_, scaler.max_
    ad = ApplicabilityDomain().fit(train_smiles, desc_min, desc_max)
    return ad, train_smiles


def featurise_one(smiles: str, features: str, scaler=None):
    if features == "ecfp":
        return ecfp([smiles])
    if features == "rdkit":
        X = rdkit_descriptors([smiles])
        return scaler.transform(X) if scaler is not None else X
    return [mol_to_graph(smiles)]


def screen(smiles: str, datasets=None) -> ScreeningResult:
    """Score one SMILES across every endpoint of every requested dataset."""
    canonical = standardize_smiles(smiles)
    if canonical is None:
        raise ValueError("Could not parse that SMILES string.")
    datasets = datasets or list(DATASETS)
    result = ScreeningResult(query=smiles, canonical=canonical)

    for name in datasets:
        spec = DATASETS[name]
        loaded = load_model(name, DEFAULT_MODEL[name])
        if loaded is None:
            continue
        model, scaler = loaded["model"], loaded.get("scaler")
        X = featurise_one(canonical, loaded["features"], scaler)
        P = model.predict(X)[0]
        thr = getattr(model, "thresholds", None)
        for t, task in enumerate(spec.tasks):
            if spec.task_type == "classification":
                th = float(thr[t]) if thr is not None and len(thr) > t else 0.5
                call = "Flagged" if P[t] >= th else "Clear"
                result.endpoints.append(EndpointResult(
                    dataset=name, task=task, label=spec.task_labels.get(task, task),
                    task_type="classification", value=float(P[t]), threshold=th, call=call))
            else:
                result.endpoints.append(EndpointResult(
                    dataset=name, task=TASK_SHORT.get(task, task),
                    label=spec.task_labels.get(task, task), task_type="regression",
                    value=float(P[t]), call=f"{P[t]:.2f} logS"))

    # Confidence flag from the Tox21 domain (the largest, most diverse training space)
    ad, train_smiles = load_domain("tox21")
    desc = rdkit_descriptors([canonical])
    verdict = ad.assess([canonical], desc)[0]
    result.confidence = verdict
    result.neighbours = ad.nearest_neighbours(canonical, train_smiles, n=3)
    return result


def explain_endpoint(smiles: str, dataset: str, task_index: int):
    """SHAP atom attribution for a fingerprint model endpoint."""
    from .explain import shap_atom_weights, top_bits
    canonical = standardize_smiles(smiles)
    loaded = load_model(dataset, DEFAULT_MODEL[dataset])
    if loaded is None or loaded["features"] != "ecfp" or loaded["family"] != "XGB":
        return None
    xgb = loaded["model"].models[task_index]
    weights, sv, bitinfo = shap_atom_weights(xgb, canonical)
    return {"weights": weights, "shap": sv, "bitinfo": bitinfo, "top_bits": top_bits(sv, bitinfo)}


def explain_attention(smiles: str, dataset: str = "esol"):
    """Attentive FP readout attention per atom, from the model trained on `dataset`."""
    canonical = standardize_smiles(smiles)
    loaded = load_model(dataset, "AttentiveFP_graph")
    if loaded is None:
        return None
    graph = mol_to_graph(canonical)
    return loaded["model"].atom_attention(graph)


def best_attention(smiles: str, datasets=None):
    """The most concentrated attention map available for this molecule, and where it came from.

    Attention informativeness turned out to be a property of the individual fitted model rather than
    of the Attentive FP architecture (see the report, section 7.2): on this benchmark the ClinTox
    model is sharply selective while the other three are indistinguishable from uniform. Rather than
    hard-coding that result, this picks whichever available model actually has something to say about
    the molecule in hand, so the app degrades gracefully if the models are retrained.

    Returns (dataset, attention, concentration) — or None when no Attentive FP model is on disk.
    """
    from .explain import attention_concentration

    best = None
    for name in (datasets or list(DATASETS)):
        att = explain_attention(smiles, name)
        if att is None:
            continue
        conc = attention_concentration(att)
        if best is None or conc["ratio"] > best[2]["ratio"]:
            best = (name, att, conc)
    return best
