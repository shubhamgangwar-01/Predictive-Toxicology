# MolRepBench

**Interpretable Toxicity & ADMET Screening from Molecular Structure**

Shubham Gangwar · MS23BTECH11027 · Predictive Toxicology / AI in Drug Discovery

Give it a SMILES string; it returns a toxicity and ADMET risk panel across 16 endpoints, the
substructure that drove each prediction, and a confidence flag saying whether the compound is close
enough to the training data for the answer to mean anything.

---

## Quick start

```bash
git clone https://github.com/shubhamgangwar-01/Predictive-Toxicology.git
cd Predictive-Toxicology
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

That is all. The trained models are committed to this repository, so nothing needs retraining. The first
prediction rebuilds the feature cache from `data/raw/` and takes a minute or two; everything after
that is instant.

Try `CC(C)(c1ccc(O)cc1)c1ccc(O)cc1` (bisphenol A) — it should come back flagged on the estrogen
receptor endpoints, with both phenol rings highlighted red in the "Why this prediction" tab.

To use it from Python instead of the browser:

```python
from molrepbench.predict import screen

res = screen("CC(=O)Oc1ccccc1C(=O)O")          # aspirin
print(res.confidence["level"])                  # In domain / Borderline / Out of domain
for e in sorted(res.endpoints, key=lambda e: -e.value):
    print(f"{e.label:45s} {e.value:.3f}  {e.call}")
```

---

## How it works

**1 · Standardisation.** Every SMILES goes through the same pipeline in training and at inference:
keep the largest organic fragment, neutralise charges, canonicalise, and discard anything whose
canonical form does not survive re-parsing. Duplicates are removed. This matters because a model
trained on canonical structures will quietly misread a salt or a tautomer written differently.

**2 · Three representations.** The project exists to compare them:

| Representation | What it is |
| --- | --- |
| ECFP4 fingerprint | 2048-bit circular fingerprint, radius 2 — presence/absence of substructures |
| RDKit descriptors | ~210 physicochemical and topological properties (logP, TPSA, ring counts, …) |
| Molecular graph | Atoms as nodes (58 features), bonds as edges (11 features) |

**3 · Nine models.** Random Forest, XGBoost and a deep neural network on each of the two descriptor
representations, plus three graph neural networks — GCN, GAT and Attentive FP — on the raw graph.

**4 · Scaffold splitting.** Molecules are grouped by Bemis-Murcko scaffold and whole groups are
assigned to train / validation / test, so the test set contains chemotypes the model has never seen.
A random split would let close analogues straddle the boundary and inflate every score. Two
protocols are implemented (`balanced`, the Chemprop-style shuffled one, and `deterministic`, the
DeepChem largest-first one) because *which one you pick changes the results more than which model
you pick* — see the results below.

**5 · Imbalance-aware metrics.** ROC-AUC, PR-AUC and MCC are all reported. Tox21 assays run as low
as 3% actives, where ROC-AUC stays respectable while the model's actual yes/no decisions are near
chance. The MCC threshold is tuned on validation and then frozen before touching test.

**6 · Explanations.** For the fingerprint models, TreeSHAP gives an exact contribution per bit;
because each bit is a circular atom environment, that value is redistributed over the atoms of the
environment, producing a signed per-atom map — red raises predicted risk, blue lowers it. For
Attentive FP, the readout attention is read off the final timestep. Attention is *measured before it
is shown*: a map that is statistically indistinguishable from uniform is flagged rather than
displayed as an explanation (see "attention" below).

**7 · Applicability domain.** A QSAR model is only valid inside the chemical space it was trained
on. Every prediction carries a badge computed from the mean Tanimoto similarity to the five nearest
training molecules, thresholded against the training set's own leave-one-out distribution, plus a
check on how many RDKit descriptors fall outside the training range. The three nearest training
compounds are shown so you can judge the comparison yourself.

---

## What the benchmark found

Full grid: 9 model configurations × 4 datasets × 3 scaffold splits = 108 training runs.

| Dataset | Best descriptor model | Best graph network | Gap | Split s.d. |
| --- | --- | --- | --- | --- |
| Tox21 (ROC-AUC) | RF + RDKit **0.838** | Attentive FP 0.827 | 0.011 | 0.031 |
| ClinTox (ROC-AUC) | RF + ECFP4 **0.787** | Attentive FP 0.700 | 0.087 | 0.065 |
| BBBP (ROC-AUC) | RF + ECFP4 **0.943** | GCN 0.940 | 0.004 | 0.040 |
| ESOL (RMSE, lower better) | DNN + RDKit **0.803** | Attentive FP 0.831 | 0.028 | 0.142 |

Descriptor models lead on all four, but three of the four margins fall inside one standard deviation
of the split-to-split spread. The defensible claim is parity at lower training cost, not superiority.

**The evaluation protocol mattered more than the architecture.** Changing only the scaffold-split
protocol moved ROC-AUC by 0.123 on average and 0.168 on BBBP — a dataset where the two model
families differ by 0.004. The effect shrinks as datasets grow (Tox21, at 7,611 molecules, moves by
only 0.042), so it is a small-data pathology. A leakage audit confirmed neither protocol leaks:
near-duplicate rates stay between 0% and 4.4% in every split.

**Representation should match the endpoint.** On ESOL, RDKit descriptors and the graph networks take
the top six places (RMSE 0.80–0.99) while all three ECFP4 models sit at 1.48–1.65. Solubility is
governed by bulk physicochemistry that a binary substructure fingerprint cannot encode.

**Attention was informative on one fitted model in four.** The ClinTox Attentive FP has sharply
concentrated readout attention (max/min 5.1–32.2); the Tox21, BBBP and ESOL models are
indistinguishable from uniform (max/min 1.03–1.10, normalised entropy > 0.9999). Training-set size
does not explain the pattern, and the sharp model is the weakest classifier of the four. So
concentration is a property of a *fitted model*, not of the architecture — which is why every map is
scored before display. Maps are drawn on an absolute scale (full colour = 3× uniform) because
min-max normalisation renders a meaningless readout as a convincing picture.

---

## Retraining from scratch

Not required — the models are included — but fully reproducible:

```bash
python scripts/prepare_data.py          # standardise + featurise (cached to data/processed/)
python scripts/run_benchmark.py         # 4 datasets x 9 models x 3 scaffold splits
python scripts/make_report_assets.py    # aggregate into summary tables and figures
python scripts/audit_leakage.py         # train/test near-duplicate audit
python -m pytest tests/ -q              # 24 unit tests
```

`run_benchmark.py` appends results incrementally and skips runs already present, so it can be
interrupted and resumed. Useful flags:

```bash
python scripts/run_benchmark.py --datasets esol bbbp --models RF:ecfp AttentiveFP --seeds 0
python scripts/run_benchmark.py --protocol deterministic --raw results/raw_deterministic.csv
```

It trains on a GPU automatically when one is available; override with `MOLREP_DEVICE=cpu` or `cuda`.
Random Forest and XGBoost always train on CPU, since scikit-learn has no GPU backend.

---

## Layout

```
app/streamlit_app.py      the screening application
molrepbench/              the package
  config.py               dataset specs and paths
  data.py                 download, standardisation, scaffold splitting
  features.py             ECFP4, RDKit descriptors, graph construction, scaling
  models.py               the nine model configurations and their training loops
  metrics.py              ROC-AUC / PR-AUC / MCC with threshold tuning
  pipeline.py             featurisation cache, splits, train-and-evaluate
  explain.py              SHAP-to-atom mapping, attention maps and diagnostics
  applicability.py        the domain-of-validity check
  predict.py              inference service the app talks to
scripts/                  prepare_data, run_benchmark, make_report_assets, audit_leakage
tests/                    24 unit tests
data/raw/                 the four MoleculeNet CSVs, so training reproduces offline
models/                   trained weights the app loads
results/                  benchmark tables and figures
```

**On `models/`:** the repository carries the XGBoost fingerprint models (every default endpoint, and the
only configuration supporting exact TreeSHAP) and the four Attentive FP models (attention maps). The
Random Forest checkpoints are not committed because they come to 1.2 GB on their own — the forests are
unpruned — and `scripts/run_benchmark.py` regenerates them. Nothing in the app depends on them.

`results/raw_gpu.csv` holds all 432 individual benchmark runs; every other table is derived from it.
