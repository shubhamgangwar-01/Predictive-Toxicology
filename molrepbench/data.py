"""Dataset loading, SMILES standardisation and scaffold splitting."""
from __future__ import annotations

import gzip
import random
import shutil
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from .config import DATA_RAW, DATASETS, SPLIT_FRAC

RDLogger.DisableLog("rdApp.*")

_largest_frag = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_uncharger = rdMolStandardize.Uncharger()


def download(name: str) -> None:
    """Download a MoleculeNet CSV into data/raw if it is not already there."""
    spec = DATASETS[name]
    dest = DATA_RAW / spec.filename
    if dest.exists():
        return
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".download")
    urllib.request.urlretrieve(spec.url, tmp)
    if spec.url.endswith(".gz"):
        with gzip.open(tmp, "rb") as fin, open(dest, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        tmp.unlink()
    else:
        tmp.rename(dest)


def standardize_smiles(smiles: str) -> str | None:
    """Parse a SMILES, keep the largest organic fragment (drops salts/counter-ions),
    neutralise charges where possible and return canonical SMILES. None if unparsable."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        # Some MoleculeNet entries fail sanitisation (e.g. hypervalent N); try a partial sanitise
        mol = Chem.MolFromSmiles(smiles.strip(), sanitize=False)
        if mol is None:
            return None
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION,
                catchErrors=True,
            )
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        except Exception:
            return None
        if mol is None:
            return None
    try:
        mol = _largest_frag.choose(mol)
        mol = _uncharger.uncharge(mol)
    except Exception:
        pass
    if mol.GetNumAtoms() == 0:
        return None
    out = Chem.MolToSmiles(mol)
    # Only keep structures whose canonical SMILES round-trips through full sanitisation
    return out if Chem.MolFromSmiles(out) is not None else None


def load_dataset(name: str) -> pd.DataFrame:
    """Return a dataframe with columns ['smiles', *tasks]; labels may be NaN (Tox21)."""
    spec = DATASETS[name]
    download(name)
    df = pd.read_csv(DATA_RAW / spec.filename)
    out = pd.DataFrame({"raw_smiles": df[spec.smiles_col]})
    out["smiles"] = out["raw_smiles"].map(standardize_smiles)
    for t in spec.tasks:
        out[t] = pd.to_numeric(df[t], errors="coerce")
    out = out.dropna(subset=["smiles"])
    out = out.dropna(subset=list(spec.tasks), how="all")
    # Merge duplicate structures created by salt stripping: average labels
    # (for classification, a conflicting duplicate becomes 0.5 -> dropped).
    agg = out.groupby("smiles", sort=False)[list(spec.tasks)].mean().reset_index()
    if spec.task_type == "classification":
        for t in spec.tasks:
            v = agg[t]
            agg.loc[(v > 0) & (v < 1), t] = np.nan
    agg = agg.dropna(subset=list(spec.tasks), how="all").reset_index(drop=True)
    return agg


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return smiles


def scaffold_split(smiles: list[str], seed: int = 0, frac=SPLIT_FRAC, protocol: str = "balanced"):
    """Scaffold split: no Bemis-Murcko scaffold appears in more than one subset, so the
    test set contains chemotypes never seen in training.

    Two protocols exist in the literature and they are NOT interchangeable - on BBBP they
    differ by ~0.2 ROC-AUC, more than any model difference in this benchmark. Mixing them
    across seeds would make the mean meaningless, so the protocol is fixed for a whole run:

    "balanced" (default, Chemprop style)
        Scaffold sets larger than half the validation/test quota go to train first, so the
        few very generic frameworks cannot dominate the held-out sets; the rest are shuffled
        with `seed`. Every seed is a genuinely different draw of the same protocol, which is
        what makes averaging over seeds meaningful.

    "deterministic" (DeepChem / original MoleculeNet)
        Scaffold sets sorted largest first, no randomness. The test set is therefore the tail
        of rare, structurally unusual scaffolds, which is markedly harder and ignores `seed`.
        This is the protocol behind the commonly quoted MoleculeNet numbers.
    """
    n = len(smiles)
    n_train, n_val = int(frac[0] * n), int(frac[1] * n)
    n_test = n - n_train - n_val
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[murcko_scaffold(s)].append(i)
    sets = list(groups.values())

    if protocol == "deterministic":
        sets = sorted(sets, key=lambda x: (len(x), x[0]), reverse=True)
    elif protocol == "balanced":
        rng = random.Random(seed)
        big = [s for s in sets if len(s) > n_val / 2 or len(s) > n_test / 2]
        small = [s for s in sets if not (len(s) > n_val / 2 or len(s) > n_test / 2)]
        rng.shuffle(big)
        rng.shuffle(small)
        sets = big + small
    else:
        raise ValueError(f"Unknown split protocol: {protocol!r}")

    train, val, test = [], [], []
    for s in sets:
        if len(train) + len(s) <= n_train:
            train += s
        elif len(val) + len(s) <= n_val:
            val += s
        else:
            test += s
    return np.array(sorted(train)), np.array(sorted(val)), np.array(sorted(test))
