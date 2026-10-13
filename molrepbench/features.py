"""Molecular featurisation: ECFP4 fingerprints, RDKit 2D descriptors and PyG graphs."""
from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from torch_geometric.data import Data

ECFP_BITS = 2048
ECFP_RADIUS = 2  # radius 2 == ECFP4 (diameter 4)

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)
DESCRIPTOR_NAMES = [name for name, _ in Descriptors._descList]
_DESC_FUNCS = [fn for _, fn in Descriptors._descList]


def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


# --------------------------------------------------------------------- ECFP4
def ecfp_bitvect(mol):
    return _morgan.GetFingerprint(mol)


def ecfp(smiles_list) -> np.ndarray:
    out = np.zeros((len(smiles_list), ECFP_BITS), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        out[i] = _morgan.GetFingerprintAsNumPy(mol_from_smiles(s))
    return out


def ecfp_with_bitinfo(mol):
    """ECFP4 bits plus the (atom, radius) environments that set each bit."""
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    fp = _morgan.GetFingerprintAsNumPy(mol, additionalOutput=ao)
    return fp.astype(np.float32), ao.GetBitInfoMap()


def tanimoto_matrix(query_fps, ref_fps) -> np.ndarray:
    """Tanimoto similarity of each query bitvect to every reference bitvect."""
    return np.array([DataStructs.BulkTanimotoSimilarity(q, ref_fps) for q in query_fps])


# ---------------------------------------------------------- RDKit descriptors
def rdkit_descriptors(smiles_list) -> np.ndarray:
    out = np.zeros((len(smiles_list), len(DESCRIPTOR_NAMES)), dtype=np.float64)
    for i, s in enumerate(smiles_list):
        mol = mol_from_smiles(s)
        for j, fn in enumerate(_DESC_FUNCS):
            try:
                out[i, j] = fn(mol)
            except Exception:
                out[i, j] = np.nan
    out[~np.isfinite(out)] = np.nan
    # Guard against absurd values (e.g. Ipc overflow) that break scaling
    out = np.clip(out, -1e6, 1e6)
    return out.astype(np.float32)


class DescriptorScaler:
    """Median imputation + robust standardisation fitted on the training split only."""

    def fit(self, X):
        self.median_ = np.nanmedian(X, axis=0)
        self.median_ = np.where(np.isnan(self.median_), 0.0, self.median_)
        Xi = np.where(np.isnan(X), self.median_, X)
        self.mean_ = Xi.mean(0)
        self.std_ = Xi.std(0)
        self.std_[self.std_ < 1e-8] = 1.0
        # Feature range seen in training: used by the applicability-domain check
        self.min_, self.max_ = Xi.min(0), Xi.max(0)
        return self

    def impute(self, X):
        return np.where(np.isnan(X), self.median_, X).astype(np.float32)

    def transform(self, X):
        Z = (self.impute(X) - self.mean_) / self.std_
        return np.clip(Z, -10, 10).astype(np.float32)


# --------------------------------------------------------------- graph (PyG)
ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "Na", "K", "Ca", "Fe",
                "Zn", "Cu", "Sn", "As", "Hg", "Pt", "Al", "H"]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_STEREO = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
               Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]


def _one_hot(value, choices, other=True):
    vec = [float(value == c) for c in choices]
    if other:
        vec.append(float(value not in choices))
    return vec


def atom_features(atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOM_SYMBOLS)                      # 25
        + _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])              # 7
        + _one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])         # 6
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)           # 7
        + _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])             # 6
        + _one_hot(atom.GetChiralTag(), CHIRAL_TAGS)                  # 4
        + [float(atom.GetIsAromatic()), float(atom.IsInRing()),       # 2
           atom.GetMass() / 100.0]                                    # 1
    )


def bond_features(bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES, other=False)         # 4
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]     # 2
        + _one_hot(bond.GetStereo(), BOND_STEREO)                     # 5
    )


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def mol_to_graph(smiles: str, y=None) -> Data:
    mol = mol_from_smiles(smiles)
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]
        dst += [j, i]
        eattr += [f, f]
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float32)
    else:  # single-atom molecules
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_DIM), dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    if y is not None:
        data.y = torch.tensor(np.asarray(y, dtype=np.float32)).view(1, -1)
    return data
