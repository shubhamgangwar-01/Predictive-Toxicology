"""MolRepBench - interpretable toxicity & ADMET screening from a SMILES string.

Run with:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from molrepbench.config import DATASETS, RESULTS
from molrepbench.explain import (atom_heatmap_png, attention_png,
                                 draw_bit, mol_png)
from molrepbench.predict import DEFAULT_MODEL, best_attention, explain_endpoint, screen

st.set_page_config(page_title="MolRepBench", page_icon="🧪", layout="wide")

EXAMPLES = {
    "Aspirin (safe reference)": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine (CNS active)": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Estradiol (ER agonist)": "C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc34)[C@@H]1CC[C@@H]2O",
    "Bisphenol A (endocrine disruptor)": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
    "Rotenone (mitochondrial toxin)": "COc1cc2c(cc1OC)C(=O)[C@@H]1[C@H]2CO[C@@H]2[C@@H]1c1ccc(C=C(C)C)o1",
    "Thalidomide (clinical failure)": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Imatinib (large kinase inhibitor)": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
}

BADGE = {"In domain": "🟢", "Borderline": "🟡", "Out of domain": "🔴"}


@st.cache_data(show_spinner=False)
def run_screen(smiles: str):
    r = screen(smiles)
    return {
        "canonical": r.canonical,
        "endpoints": [vars(e) for e in r.endpoints],
        "confidence": r.confidence,
        "neighbours": r.neighbours,
    }


@st.cache_data(show_spinner=False)
def benchmark_table():
    path = RESULTS / "benchmark_summary.csv"
    return pd.read_csv(path) if path.exists() else None


def risk_panel(endpoints, confidence):
    cls = [e for e in endpoints if e["task_type"] == "classification"]
    reg = [e for e in endpoints if e["task_type"] == "regression"]
    flagged = [e for e in cls if e["call"] == "Flagged"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Endpoints flagged", f"{len(flagged)} / {len(cls)}")
    if reg:
        c2.metric("Aqueous solubility", f"{reg[0]['value']:.2f} logS")
    lvl = confidence["level"]
    c3.metric("Confidence", f"{BADGE[lvl]} {lvl}", f"Tanimoto {confidence['knn_sim']:.2f}")

    if lvl == "Out of domain":
        st.warning("This compound sits outside the training chemical space. "
                   "Every number below should be treated as unreliable.", icon="⚠️")
    elif lvl == "Borderline":
        st.info("This compound is at the edge of the training chemical space - interpret with care.", icon="ℹ️")

    rows = []
    for e in cls:
        rows.append({
            "Dataset": e["dataset"], "Endpoint": e["label"], "Assay": e["task"],
            "Probability": e["value"], "Cut-off": e["threshold"], "Call": e["call"],
        })
    df = pd.DataFrame(rows).sort_values("Probability", ascending=False)
    st.dataframe(
        df.style.format({"Probability": "{:.3f}", "Cut-off": "{:.3f}"})
          .background_gradient(subset=["Probability"], cmap="Reds", vmin=0, vmax=1),
        width="stretch", hide_index=True,
    )
    st.caption("Cut-offs maximise MCC on the validation split of a scaffold split, not a naive 0.5 - "
               "the assays are heavily imbalanced (3-16% actives).")
    return df


def main():
    st.title("🧪 MolRepBench")
    st.markdown("**Interpretable toxicity & ADMET screening from molecular structure.** "
                "Enter a SMILES string to get a per-endpoint risk panel with a confidence flag, "
                "then see which substructures drove each prediction.")

    with st.sidebar:
        st.header("Input")
        choice = st.selectbox("Example compounds", ["(type my own)"] + list(EXAMPLES))
        default = EXAMPLES.get(choice, "CC(=O)Oc1ccccc1C(=O)O")
        smiles = st.text_area("SMILES", value=default, height=90)
        go = st.button("Screen compound", type="primary", width="stretch")
        st.divider()
        st.caption("**Models in use**")
        for d, m in DEFAULT_MODEL.items():
            st.caption(f"`{d}` → {m.replace('_', ' + ')}")

    tab_screen, tab_explain, tab_bench, tab_about = st.tabs(
        ["Risk panel", "Why this prediction", "Benchmark", "About"])

    if go or smiles:
        try:
            res = run_screen(smiles.strip())
        except Exception as exc:
            st.error(f"Could not screen that compound: {exc}")
            return

        with tab_screen:
            left, right = st.columns([1, 2])
            with left:
                st.image(mol_png(res["canonical"], (380, 300)), caption="Standardised structure")
                st.code(res["canonical"], language=None)
            with right:
                df = risk_panel(res["endpoints"], res["confidence"])

            st.subheader("Nearest training compounds")
            nb = pd.DataFrame(res["neighbours"], columns=["Training SMILES", "Tanimoto similarity"])
            st.dataframe(nb.style.format({"Tanimoto similarity": "{:.3f}"}),
                         width="stretch", hide_index=True)
            st.download_button("Download risk panel (CSV)", df.to_csv(index=False),
                               file_name="molrepbench_panel.csv", mime="text/csv")

        with tab_explain:
            st.subheader("Substructure attribution")
            cls = [e for e in res["endpoints"] if e["task_type"] == "classification"]
            labels = [f"{e['dataset']} · {e['label']} (p={e['value']:.2f})" for e in cls]
            pick = st.selectbox("Endpoint", labels, index=int(np.argmax([e["value"] for e in cls])))
            e = cls[labels.index(pick)]
            spec = DATASETS[e["dataset"]]
            ti = list(spec.tasks).index(e["task"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**SHAP on ECFP4 bits → atoms**")
                exp = explain_endpoint(res["canonical"], e["dataset"], ti)
                if exp is None:
                    st.info("SHAP attribution is available for the fingerprint + XGBoost models.")
                else:
                    st.image(atom_heatmap_png(res["canonical"], exp["weights"]))
                    st.caption("Red raises the predicted risk, blue lowers it. Atom weights are the "
                               "SHAP value of each ECFP4 bit spread over the atoms that set it.")
                    bits = pd.DataFrame(exp["top_bits"], columns=["ECFP4 bit", "SHAP value"])
                    st.dataframe(bits.style.format({"SHAP value": "{:+.3f}"}),
                                 width="stretch", hide_index=True)
                    top = exp["top_bits"][0][0] if exp["top_bits"] else None
                    if top is not None:
                        try:
                            st.image(draw_bit(res["canonical"], top, exp["bitinfo"]),
                                     caption=f"Most influential environment (bit {top})")
                        except Exception:
                            pass
            with col2:
                st.markdown("**Attentive FP readout attention**")
                picked = best_attention(res["canonical"])
                if picked is None:
                    st.info("Train the Attentive FP model to see attention maps.")
                else:
                    att_dataset, att, conc = picked
                    st.caption(f"Showing the Attentive FP model trained on **{att_dataset}** — the "
                               f"most concentrated of the available readouts for this molecule.")
                    if not conc["informative"]:
                        st.warning(
                            f"This model's readout attention is essentially uniform "
                            f"(max/min = {conc['ratio']:.2f} across {conc['n_atoms']} atoms, "
                            f"normalised entropy {conc['entropy']:.3f}). It therefore explains "
                            f"nothing, and it is the most concentrated readout available for this "
                            f"molecule — so none of the trained graph models has an attention-based "
                            f"explanation to offer here. The map is shown only to make that visible. "
                            f"Use the SHAP attribution instead.", icon="⚠️")
                    # Absolute scale, deliberately: min-max normalisation would stretch a readout
                    # varying in the fourth decimal place across the full colour range, drawing a
                    # vivid picture of nothing directly above the warning that says so.
                    st.image(attention_png(res["canonical"], att, absolute=True))
                    st.caption("Darker orange = atoms the graph network's molecule-level readout "
                               "attends to most, on an absolute scale where full colour is 3x "
                               "uniform attention — so a model that attends to nothing shows "
                               "nothing. Attention marks where the model looked, which is not the "
                               "same as what drove the output.")

    with tab_bench:
        st.subheader("Representation benchmark under scaffold splits")
        tbl = benchmark_table()
        if tbl is None:
            st.info("Run `python scripts/run_benchmark.py` then `python scripts/make_report_assets.py`.")
        else:
            st.dataframe(tbl, width="stretch", hide_index=True)
            st.caption("Mean ± s.d. over 3 scaffold splits (seeds 0-2). Scaffold splitting puts whole "
                       "Bemis-Murcko chemotypes in the test set, so these numbers measure generalisation "
                       "to new chemistry, not interpolation.")
        for fig, cap in [("fig_representation.png", "Descriptor models vs. graph networks"),
                         ("fig_imbalance.png", "ROC-AUC vs. PR-AUC under class imbalance")]:
            p = RESULTS / "figures" / fig
            if p.exists():
                st.image(str(p), caption=cap)

    with tab_about:
        st.markdown(f"""
### What this does
MolRepBench predicts **toxicity and ADMET risk directly from a SMILES string** and shows *why*.

**Datasets** (MoleculeNet)
{chr(10).join(f"- `{k}` — {v.description}" for k, v in DATASETS.items())}

**Representations benchmarked**
- ECFP4 circular fingerprints (2048 bits, radius 2)
- RDKit 2D descriptors (~210 physicochemical/topological features)
- Molecular graphs: GCN, GAT, Attentive FP

**Evaluation** — scaffold splits (80/10/10), 3 seeds, ROC-AUC + PR-AUC + MCC.
PR-AUC and MCC are reported because ROC-AUC flatters a model on assays with 3% actives.

**Interpretability** — TreeSHAP on fingerprint bits mapped back onto atoms, and
Attentive FP's readout attention per atom. Attention was informative on only one of the four
fitted models (ClinTox), so every map is scored for concentration before it is shown and a
near-uniform one is flagged rather than presented as an explanation.

**Confidence flag** — mean Tanimoto similarity to the 5 nearest training molecules,
thresholded at the 5th/25th percentile of the training set's own leave-one-out distribution,
plus a check on how many RDKit descriptors fall outside the training range.

---
Shubham Gangwar · MS23BTECH11027 · Predictive Toxicology / AI in Drug Discovery
""")


if __name__ == "__main__":
    main()
