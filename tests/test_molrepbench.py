"""Unit tests for the parts of MolRepBench where a silent bug would corrupt every result.

    python -m pytest tests/ -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from molrepbench.applicability import ApplicabilityDomain
from molrepbench.data import murcko_scaffold, scaffold_split, standardize_smiles
from molrepbench.features import (ATOM_DIM, BOND_DIM, DescriptorScaler, ecfp, ecfp_with_bitinfo,
                                  mol_from_smiles, mol_to_graph, rdkit_descriptors)
from molrepbench.metrics import best_mcc_threshold, classification_metrics, masked_mean_auc

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


# ------------------------------------------------------------- standardisation
def test_standardize_canonicalises_equivalent_smiles():
    """The same molecule written two ways must collapse to one string, or it can land
    in both the train and the test split and inflate every score."""
    assert standardize_smiles("C1=CC=CC=C1") == standardize_smiles("c1ccccc1")


def test_standardize_strips_salt():
    """Propranolol hydrochloride and propranolol must give the same canonical structure."""
    assert standardize_smiles("[Cl].CC(C)NCC(O)COc1cccc2ccccc12") == \
           standardize_smiles("CC(C)NCC(O)COc1cccc2ccccc12")


def test_standardize_rejects_garbage():
    for bad in ["not_a_smiles", "", "   ", None, "C(((C"]:
        assert standardize_smiles(bad) is None


def test_standardize_output_always_reparses():
    """Anything we return must survive full RDKit sanitisation downstream."""
    for s in [ASPIRIN, CAFFEINE, "[Na+].[Cl-]", "CC(=O)[O-]"]:
        out = standardize_smiles(s)
        if out is not None:
            assert mol_from_smiles(out) is not None


# ------------------------------------------------------------- scaffold splits
@pytest.fixture(scope="module")
def smiles_sample():
    from molrepbench.data import load_dataset
    return load_dataset("bbbp")["smiles"].tolist()


def test_split_is_a_partition(smiles_sample):
    tr, va, te = scaffold_split(smiles_sample, seed=0)
    allidx = np.concatenate([tr, va, te])
    assert len(allidx) == len(smiles_sample)
    assert len(set(allidx.tolist())) == len(smiles_sample)


def test_no_scaffold_crosses_splits(smiles_sample):
    """The whole point of a scaffold split: a chemotype in test must be absent from train."""
    tr, va, te = scaffold_split(smiles_sample, seed=1)
    sc = {i: murcko_scaffold(s) for i, s in enumerate(smiles_sample)}
    s_tr = {sc[i] for i in tr}
    assert not (s_tr & {sc[i] for i in te})
    assert not (s_tr & {sc[i] for i in va})


def test_split_proportions(smiles_sample):
    tr, va, te = scaffold_split(smiles_sample, seed=2)
    n = len(smiles_sample)
    assert 0.74 < len(tr) / n < 0.86
    assert len(va) / n < 0.16 and len(te) / n < 0.16


def test_balanced_protocol_varies_with_seed_deterministic_does_not(smiles_sample):
    """Mixing the two protocols across seeds silently averages two different experiments —
    the bug this test exists to prevent."""
    a = scaffold_split(smiles_sample, seed=0, protocol="balanced")[2]
    b = scaffold_split(smiles_sample, seed=1, protocol="balanced")[2]
    assert not np.array_equal(a, b)
    d0 = scaffold_split(smiles_sample, seed=0, protocol="deterministic")[2]
    d1 = scaffold_split(smiles_sample, seed=7, protocol="deterministic")[2]
    assert np.array_equal(d0, d1)


def test_split_is_reproducible(smiles_sample):
    a = scaffold_split(smiles_sample, seed=3)
    b = scaffold_split(smiles_sample, seed=3)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# ----------------------------------------------------------------- featurisation
def test_ecfp_shape_and_binary():
    X = ecfp([ASPIRIN, CAFFEINE])
    assert X.shape == (2, 2048)
    assert set(np.unique(X)).issubset({0.0, 1.0})


def test_ecfp_is_deterministic_and_discriminative():
    assert np.array_equal(ecfp([ASPIRIN]), ecfp([ASPIRIN]))
    assert not np.array_equal(ecfp([ASPIRIN]), ecfp([CAFFEINE]))


def test_bitinfo_atoms_are_in_range():
    """Every environment SHAP attributes to must point at a real atom."""
    mol = mol_from_smiles(ASPIRIN)
    fp, bitinfo = ecfp_with_bitinfo(mol)
    assert fp.sum() == len(bitinfo) or fp.sum() >= 1
    for envs in bitinfo.values():
        for center, radius in envs:
            assert 0 <= center < mol.GetNumAtoms()
            assert radius >= 0


def test_descriptors_finite_after_scaling():
    X = rdkit_descriptors([ASPIRIN, CAFFEINE, "CCCCCCCCCCCCCCCCCCCC"])
    sc = DescriptorScaler().fit(X)
    Z = sc.transform(X)
    assert np.isfinite(Z).all()
    assert np.abs(Z).max() <= 10.0 + 1e-6


def test_scaler_fits_on_train_only():
    """Imputation and scaling statistics must not see the test rows."""
    train = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    sc = DescriptorScaler().fit(train)
    assert np.allclose(sc.mean_, [2.0, 3.0])
    out = sc.transform(np.array([[np.nan, 100.0]], dtype=np.float32))
    assert np.isfinite(out).all()


def test_graph_structure():
    g = mol_to_graph(ASPIRIN)
    mol = mol_from_smiles(ASPIRIN)
    assert g.x.shape == (mol.GetNumAtoms(), ATOM_DIM)
    assert g.edge_attr.shape[1] == BOND_DIM
    # undirected: every bond contributes two directed edges
    assert g.edge_index.shape[1] == 2 * mol.GetNumBonds()
    assert int(g.edge_index.max()) < mol.GetNumAtoms()


def test_single_atom_molecule_has_no_edges():
    g = mol_to_graph("C")
    assert g.x.shape[0] == 1
    assert g.edge_index.shape == (2, 0)
    assert g.edge_attr.shape == (0, BOND_DIM)


# ----------------------------------------------------------------------- metrics
def test_masked_auc_ignores_nan():
    Y = np.array([[1.0, np.nan], [0.0, 1.0], [1.0, 0.0], [0.0, np.nan]])
    P = np.array([[0.9, 0.5], [0.1, 0.8], [0.8, 0.2], [0.2, 0.4]])
    assert masked_mean_auc(Y, P) == pytest.approx(1.0)


def test_metrics_undefined_for_single_class():
    m = classification_metrics(np.zeros(10), np.random.rand(10))
    assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"])


def test_threshold_beats_default_on_imbalanced_data():
    """A 0.5 cut-off is meaningless at a 5% positive rate; the tuned threshold must do better."""
    rng = np.random.default_rng(0)
    y = np.zeros(400)
    y[:20] = 1
    p = np.where(y == 1, rng.uniform(0.15, 0.45, 400), rng.uniform(0.0, 0.2, 400))
    thr = best_mcc_threshold(y, p)
    assert classification_metrics(y, p, thr)["mcc"] >= classification_metrics(y, p, 0.5)["mcc"]


# ------------------------------------------------------------ applicability domain
def test_domain_flags_dissimilar_compound():
    """A compound unlike anything in training must not come back 'In domain'."""
    train = [ASPIRIN, "CC(=O)Nc1ccc(O)cc1", "CC(=O)Oc1ccccc1C(=O)OC", "c1ccccc1C(=O)O",
             "CC(=O)Oc1ccccc1", "OC(=O)c1ccccc1O"]
    ad = ApplicabilityDomain(k=3).fit(train)
    far = ad.assess(["[Pt](Cl)(Cl)(N)N"])[0]
    near = ad.assess([ASPIRIN])[0]
    assert near["knn_sim"] > far["knn_sim"]
    assert far["level"] in ("Borderline", "Out of domain")


def test_domain_similarity_is_bounded():
    ad = ApplicabilityDomain(k=2).fit([ASPIRIN, CAFFEINE, "CCO"])
    for s in ad.knn_similarity([ASPIRIN, "CCCC"]):
        assert 0.0 <= s <= 1.0


# --------------------------------------------------------------- attention diagnostics
def test_attention_concentration_separates_uniform_from_peaked():
    from molrepbench.explain import attention_concentration
    flat = attention_concentration(np.full(12, 1 / 12))
    assert not flat["informative"]
    assert flat["entropy"] > 0.999 and flat["ratio"] < 1.01

    peaked = np.full(12, 0.01)
    peaked[0] = 0.89
    sharp = attention_concentration(peaked)
    assert sharp["informative"]
    assert sharp["ratio"] > 1.5 and sharp["entropy"] < 0.99


def test_attention_concentration_is_scale_invariant():
    """The diagnostic normalises internally, so an unnormalised readout scores the same."""
    from molrepbench.explain import attention_concentration
    a = np.array([0.5, 0.2, 0.2, 0.1])
    assert (attention_concentration(a)["ratio"]
            == pytest.approx(attention_concentration(a * 7.3)["ratio"]))


def test_absolute_attention_scale_blanks_a_uniform_map():
    """Regression test for the figure that made flat attention look like an explanation.

    Min-max normalisation stretches a readout varying in the fourth decimal place across the full
    colour range; the absolute scale must leave it uncoloured while still showing a real peak.
    """
    from rdkit import Chem

    from molrepbench.explain import attention_png

    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    n = Chem.MolFromSmiles(smiles).GetNumAtoms()
    nearly_uniform = np.full(n, 1 / n) + np.linspace(0, 1e-4, n)
    peaked = np.full(n, 0.01)
    peaked[0] = 1.0

    flat_abs = attention_png(smiles, nearly_uniform, absolute=True)
    flat_rel = attention_png(smiles, nearly_uniform, absolute=False)
    sharp_abs = attention_png(smiles, peaked, absolute=True)

    # All three are valid PNGs ...
    for png in (flat_abs, flat_rel, sharp_abs):
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # ... but only the absolute rendering of flat attention is visually empty, which is the point.
    assert flat_abs != flat_rel
    assert flat_abs != sharp_abs
