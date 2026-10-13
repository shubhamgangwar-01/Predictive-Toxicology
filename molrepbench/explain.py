"""Interpretability: SHAP on fingerprint models mapped back to atoms, and GNN attention maps."""
from __future__ import annotations

import io

import numpy as np
import shap
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D

from .features import ecfp_with_bitinfo, mol_from_smiles


def tree_shap_values(xgb_model, X):
    """SHAP values (log-odds / output units) for one fitted XGBoost task model."""
    explainer = shap.TreeExplainer(xgb_model)
    return explainer.shap_values(X), explainer.expected_value


def shap_atom_weights(xgb_model, smiles: str):
    """Distribute the SHAP value of every ECFP4 bit present in the molecule over the
    atoms of the circular environment(s) that set that bit. Returns (atom_weights, bit_shap, bitinfo)."""
    mol = mol_from_smiles(smiles)
    fp, bitinfo = ecfp_with_bitinfo(mol)
    sv, _ = tree_shap_values(xgb_model, fp.reshape(1, -1))
    sv = np.asarray(sv).reshape(-1)
    weights = np.zeros(mol.GetNumAtoms())
    for bit, envs in bitinfo.items():
        atoms = set()
        for center, radius in envs:
            atoms.add(center)
            if radius > 0:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)
                for b in env:
                    bond = mol.GetBondWithIdx(b)
                    atoms.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        for a in atoms:
            weights[a] += sv[bit] / len(atoms)
    return weights, sv, bitinfo


def top_bits(sv, bitinfo, n=6):
    """Bits present in the molecule ranked by |SHAP|."""
    present = [(b, sv[b]) for b in bitinfo]
    return sorted(present, key=lambda x: -abs(x[1]))[:n]


def bit_environment(mol, bit, bitinfo):
    """Atoms and bonds of the first circular environment that sets `bit` in this molecule."""
    center, radius = bitinfo[bit][0]
    if radius == 0:
        return [center], []
    bond_ids = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center))
    atoms = {center}
    for b in bond_ids:
        bond = mol.GetBondWithIdx(b)
        atoms.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    return sorted(atoms), bond_ids


def draw_bit(smiles, bit, bitinfo, size=(300, 260)) -> bytes:
    """Draw the molecule with the environment behind `bit` highlighted.

    Replaces RDKit's Draw.DrawMorganBit, whose Cairo path raises an ArgumentError in
    RDKit 2025.03 and whose bit-info format differs from rdFingerprintGenerator's.
    """
    mol = mol_from_smiles(smiles)
    atoms, bonds = bit_environment(mol, bit, bitinfo)
    center = bitinfo[bit][0][0]
    colors = {a: (1.0, 0.72, 0.30) for a in atoms}
    colors[center] = (0.95, 0.45, 0.20)   # the environment's centre atom
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().highlightBondWidthMultiplier = 16
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=atoms, highlightAtomColors=colors,
        highlightBonds=bonds, highlightBondColors={b: (1.0, 0.72, 0.30) for b in bonds})
    d.FinishDrawing()
    return d.GetDrawingText()


def atom_heatmap_png(smiles: str, weights, size=(520, 420), colormap="coolwarm") -> bytes:
    """Atom-level heat map: red = pushes the prediction up (toxic / positive), blue = down."""
    mol = mol_from_smiles(smiles)
    weights = np.asarray(weights, dtype=float)
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().addAtomIndices = False
    if np.allclose(weights, 0):
        d.DrawMolecule(mol)
    else:
        from matplotlib import cm
        SimilarityMaps.GetSimilarityMapFromWeights(mol, list(weights), draw2d=d,
                                                   colorMap=cm.get_cmap(colormap) if colormap else None,
                                                   contourLines=6, alpha=0.4)
    d.FinishDrawing()
    return d.GetDrawingText()


# An atom reaches full colour on the absolute attention scale at this multiple of uniform (1/n).
# 3x is a deliberate choice: it is well inside the range a genuinely peaked readout reaches (the
# ClinTox model exceeds it) while leaving a near-uniform readout visibly blank.
ABS_REFERENCE = 3.0


def attention_concentration(attention) -> dict:
    """How far the attention is from uniform — i.e. whether it explains anything at all.

    A readout that assigns every atom ~1/n carries no information, and rendering it as a heat map
    would show a flat picture that *looks* like an explanation. Returns the max/min ratio and the
    normalised entropy (1.0 = perfectly uniform), plus a verdict used by the app and the figures.
    """
    a = np.asarray(attention, dtype=float)
    a = a / max(a.sum(), 1e-12)
    n = len(a)
    ratio = float(a.max() / max(a.min(), 1e-12))
    ent = float(-(a * np.log(np.clip(a, 1e-12, None))).sum() / np.log(n)) if n > 1 else 1.0
    return {"ratio": ratio, "entropy": ent, "n_atoms": n,
            "informative": bool(ratio >= 1.5 and ent < 0.99)}


def attention_png(smiles: str, attention, size=(520, 420), absolute=False) -> bytes:
    """Attention map: darker orange = more attention from the model's readout.

    `absolute=True` scales against uniform attention (1/n per atom) instead of stretching each
    molecule's own min-max range. This matters when comparing models: min-max normalisation turns a
    flat readout into a vivid picture that looks like an explanation, because it amplifies whatever
    numerical noise separates the atoms. On the absolute scale an atom reaches full colour at
    `ABS_REFERENCE` times uniform, so a model that attends to nothing correctly renders as nothing.
    """
    mol = mol_from_smiles(smiles)
    a = np.asarray(attention, dtype=float)
    if absolute:
        a = a / max(a.sum(), 1e-12)
        a = np.clip((a * len(a) - 1.0) / (ABS_REFERENCE - 1.0), 0.0, 1.0)
    else:
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    colors = {i: (1.0, 1.0 - 0.6 * float(w), 1.0 - 0.95 * float(w)) for i, w in enumerate(a)}
    radii = {i: 0.3 + 0.25 * float(w) for i, w in enumerate(a)}
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=list(colors), highlightAtomColors=colors,
                                       highlightAtomRadii=radii, highlightBonds=[])
    d.FinishDrawing()
    return d.GetDrawingText()


def mol_png(smiles: str, size=(400, 300)) -> bytes:
    img = Draw.MolToImage(mol_from_smiles(smiles), size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
