"""Aggregate benchmark_raw.csv into the summary tables and figures used by the report, deck and app.

Produces in results/:
  benchmark_summary.csv       mean +/- s.d. per dataset x model x metric
  benchmark_pivot_*.csv       one table per dataset, ready to paste into the report
  tox21_per_task.csv          per-assay ROC-AUC for the best descriptor model vs. best GNN
  representation_winner.csv   which representation wins on each dataset and by how much
  figures/fig_representation.png, fig_imbalance.png, fig_tox21_tasks.png, fig_dataset_stats.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from molrepbench.config import DATASETS, FIGURES, GNN_MODELS, RESULTS

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

GREY, ACCENT = "#9aa0a6", "#1a73e8"
DATASET_ORDER = ["tox21", "clintox", "bbbp", "esol"]
MODEL_ORDER = ["RF:ecfp", "RF:rdkit", "XGB:ecfp", "XGB:rdkit", "DNN:ecfp", "DNN:rdkit",
               "GCN:graph", "GAT:graph", "AttentiveFP:graph"]


def load_raw(protocol: str = "balanced") -> pd.DataFrame:
    """Concatenate every raw_*.csv shard, keeping ONE split protocol.

    The two scaffold-split protocols are different experiments (see molrepbench.data), so
    averaging across them would be meaningless; the headline benchmark is the balanced one.
    """
    shards = sorted(RESULTS.glob("raw_*.csv")) + sorted(RESULTS.glob("benchmark_raw.csv"))
    if not shards:
        raise SystemExit("No raw results found. Run scripts/run_benchmark.py first.")
    df = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    if "protocol" not in df.columns:
        df["protocol"] = "balanced"
    df = df[df.protocol == protocol].copy()
    df = df.drop_duplicates(subset=["dataset", "model", "features", "protocol", "seed", "task"],
                            keep="last")
    df["key"] = df.model + ":" + df.features
    df["representation"] = np.where(df.features == "graph", "Graph (GNN)",
                            np.where(df.features == "ecfp", "ECFP4 fingerprint", "RDKit descriptors"))
    df["family"] = np.where(df.model.isin(GNN_MODELS), "Graph neural network", "Descriptor model")
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Average over tasks within a seed first, then over seeds (so a 12-task dataset is not
    dominated by whichever assays happen to be easy)."""
    metrics = ["roc_auc", "pr_auc", "mcc", "rmse", "mae", "r2"]
    present = [m for m in metrics if m in df.columns]
    per_seed = df.groupby(["dataset", "key", "model", "features", "representation",
                           "family", "seed"], as_index=False)[present].mean()
    agg = per_seed.groupby(["dataset", "key", "model", "features", "representation", "family"],
                           as_index=False)[present].agg(["mean", "std"])
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    return agg.rename(columns={c: c.replace("_mean", "") for c in agg.columns if c.endswith("_mean")})


def fmt(mean, std):
    if pd.isna(mean):
        return "-"
    return f"{mean:.3f} ± {std:.3f}" if pd.notna(std) else f"{mean:.3f}"


def write_pivots(summary: pd.DataFrame):
    tables = {}
    for name in DATASET_ORDER:
        sub = summary[summary.dataset == name].copy()
        if sub.empty:
            continue
        clf = DATASETS[name].task_type == "classification"
        cols = ["roc_auc", "pr_auc", "mcc"] if clf else ["rmse", "mae", "r2"]
        rows = []
        for key in MODEL_ORDER:
            r = sub[sub.key == key]
            if r.empty:
                continue
            r = r.iloc[0]
            row = {"Model": key.replace(":", " + ").replace("graph", "graph"),
                   "Representation": r.representation}
            for c in cols:
                row[c.upper().replace("_", "-")] = fmt(r[c], r.get(f"{c}_std"))
            rows.append(row)
        t = pd.DataFrame(rows)
        t.to_csv(RESULTS / f"benchmark_pivot_{name}.csv", index=False)
        tables[name] = t
    return tables


def winners(summary: pd.DataFrame) -> pd.DataFrame:
    """Best descriptor model vs. best GNN on each dataset's headline metric."""
    rows = []
    for name in DATASET_ORDER:
        sub = summary[summary.dataset == name]
        if sub.empty:
            continue
        clf = DATASETS[name].task_type == "classification"
        metric = "roc_auc" if clf else "rmse"
        better = (lambda s: s.loc[s[metric].idxmax()]) if clf else (lambda s: s.loc[s[metric].idxmin()])
        d = sub[sub.family == "Descriptor model"]
        g = sub[sub.family == "Graph neural network"]
        if d.empty or g.empty:
            continue
        bd, bg = better(d), better(g)
        gap = bd[metric] - bg[metric]
        if clf:
            win = "Descriptor" if gap > 0 else "GNN"
        else:
            win = "Descriptor" if gap < 0 else "GNN"
        rows.append({"dataset": name, "metric": metric,
                     "best_descriptor": bd.key, "descriptor_score": round(bd[metric], 3),
                     "best_gnn": bg.key, "gnn_score": round(bg[metric], 3),
                     "winner": win, "absolute_gap": round(abs(gap), 3),
                     "n_molecules": None})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "representation_winner.csv", index=False)
    return out


# ------------------------------------------------------------------ figures
def fig_representation(summary: pd.DataFrame):
    """Headline metric per model, grouped by dataset; GNNs in the accent colour."""
    sets = [d for d in DATASET_ORDER if not summary[summary.dataset == d].empty]
    fig, axes = plt.subplots(1, len(sets), figsize=(3.1 * len(sets), 3.2))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, sets):
        sub = summary[summary.dataset == name].set_index("key").reindex(
            [k for k in MODEL_ORDER if k in set(summary[summary.dataset == name].key)])
        clf = DATASETS[name].task_type == "classification"
        metric = "roc_auc" if clf else "rmse"
        vals, errs = sub[metric].values, sub.get(f"{metric}_std", pd.Series(0, index=sub.index)).values
        colors = [ACCENT if f == "Graph neural network" else GREY for f in sub.family]
        y = np.arange(len(sub))
        ax.barh(y, vals, xerr=np.nan_to_num(errs), color=colors, height=0.68,
                error_kw=dict(ecolor="#5f6368", lw=0.8, capsize=2))
        ax.set_yticks(y)
        ax.set_yticklabels([k.replace(":", "\n") for k in sub.index], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"{name}  ({'ROC-AUC' if clf else 'RMSE'})", fontsize=9)
        if clf:
            ax.set_xlim(0.5, 1.0)
            ax.axvline(0.5, color="#dadce0", lw=0.8)
        # Place each value past the end of its error bar, not at the bar end, so the two do not
        # overprint; widen the axis if the longest label would otherwise fall outside it.
        e = np.nan_to_num(errs)
        pad = 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0]) if clf else 0.012 * (vals + e).max()
        for yi, v, ev in zip(y, vals, e):
            ax.text(v + ev + pad, yi, f"{v:.3f}", va="center", fontsize=6.5, color="#202124")
        right = float((vals + e).max() + pad * 7)
        if clf:
            ax.set_xlim(0.5, max(1.0, right))
        else:
            ax.set_xlim(0, right)
    fig.suptitle("Graph networks (blue) do not beat descriptor models (grey) on small datasets",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_representation.png")
    plt.close(fig)


def fig_imbalance(df: pd.DataFrame):
    """ROC-AUC flatters models on rare-active assays; PR-AUC and MCC do not."""
    t = df[(df.dataset == "tox21") & df.roc_auc.notna()]
    if t.empty:
        return
    per_task = t.groupby("task", as_index=False)[["roc_auc", "pr_auc", "mcc", "pos_rate"]].mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    a1.scatter(per_task.pos_rate * 100, per_task.roc_auc, s=34, color=GREY, label="ROC-AUC")
    a1.scatter(per_task.pos_rate * 100, per_task.pr_auc, s=34, color=ACCENT, label="PR-AUC")
    for _, r in per_task.iterrows():
        a1.annotate(r.task, (r.pos_rate * 100, r.pr_auc), fontsize=5.5,
                    xytext=(2, -5), textcoords="offset points", color="#5f6368")
    a1.set_xlabel("Positive (active) rate in the assay, %")
    a1.set_ylabel("Score")
    a1.set_title("ROC-AUC stays high as actives vanish;\nPR-AUC collapses", fontsize=9)
    a1.legend(fontsize=7, frameon=False)
    a1.set_ylim(0, 1)

    order = per_task.sort_values("pos_rate")
    y = np.arange(len(order))
    a2.barh(y - 0.2, order.roc_auc, height=0.38, color=GREY, label="ROC-AUC")
    a2.barh(y + 0.2, order.pr_auc, height=0.38, color=ACCENT, label="PR-AUC")
    a2.set_yticks(y)
    a2.set_yticklabels([f"{t} ({p*100:.0f}%)" for t, p in zip(order.task, order.pos_rate)], fontsize=6)
    a2.set_xlabel("Score (mean over all models and seeds)")
    a2.set_title("Tox21 assays, rarest actives first", fontsize=9)
    a2.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_imbalance.png")
    plt.close(fig)


def fig_tox21_tasks(df: pd.DataFrame, summary: pd.DataFrame):
    """Per-assay ROC-AUC: best descriptor model vs. best GNN."""
    s = summary[summary.dataset == "tox21"]
    t = df[(df.dataset == "tox21") & df.roc_auc.notna()]
    if s.empty or t.empty:
        return
    d = s[s.family == "Descriptor model"]
    g = s[s.family == "Graph neural network"]
    if d.empty or g.empty:
        return
    bd, bg = d.loc[d.roc_auc.idxmax(), "key"], g.loc[g.roc_auc.idxmax(), "key"]
    piv = (t[t.key.isin([bd, bg])].groupby(["task", "key"], as_index=False).roc_auc.mean()
           .pivot(index="task", columns="key", values="roc_auc").sort_values(bd))
    piv.to_csv(RESULTS / "tox21_per_task.csv")
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    y = np.arange(len(piv))
    ax.barh(y - 0.2, piv[bd], height=0.38, color=GREY, label=bd)
    ax.barh(y + 0.2, piv[bg], height=0.38, color=ACCENT, label=bg)
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("ROC-AUC (mean over 3 scaffold splits)")
    ax.set_title(f"Per-assay Tox21: {bd} vs. {bg}", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_tox21_tasks.png")
    plt.close(fig)


def fig_protocol(pc):
    """The split protocol moves scores more than the model does."""
    if pc is None or pc.empty:
        return
    fig, axes = plt.subplots(1, pc.dataset.nunique(), figsize=(3.6 * pc.dataset.nunique(), 3.2),
                             squeeze=False)
    for ax, (name, sub) in zip(axes[0], pc.groupby("dataset", sort=False)):
        sub = sub.sort_values("balanced_roc_auc")
        y = np.arange(len(sub))
        ax.barh(y - 0.2, sub.balanced_roc_auc, height=0.38, color=ACCENT, label="balanced (Chemprop)")
        ax.barh(y + 0.2, sub.deterministic_roc_auc, height=0.38, color=GREY,
                label="deterministic (DeepChem)")
        ax.set_yticks(y)
        ax.set_yticklabels(sub.model, fontsize=7)
        ax.set_xlim(0.3, 1.0)
        ax.axvline(0.5, color="#dadce0", lw=0.8)
        ax.set_xlabel("ROC-AUC")
        gap = sub.difference.mean()
        ax.set_title(f"{name} — mean gap {gap:+.2f}", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Changing only the scaffold-split protocol moves ROC-AUC further than changing the model",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_protocol.png")
    plt.close(fig)


def fig_dataset_stats():
    """Dataset size and class imbalance - the reason this benchmark exists."""
    from molrepbench.pipeline import prepare
    rows = []
    for name in DATASET_ORDER:
        try:
            b = prepare(name)
        except Exception:
            continue
        Y = b["Y"]
        rows.append({"dataset": name, "n": len(b["smiles"]), "tasks": Y.shape[1],
                     "pos_rate": float(np.nanmean(Y)) if DATASETS[name].task_type == "classification" else np.nan})
    if not rows:
        return
    st = pd.DataFrame(rows)
    st.to_csv(RESULTS / "dataset_stats.csv", index=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    a1.bar(st.dataset, st.n, color=GREY)
    for x, v in zip(st.dataset, st.n):
        a1.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=7)
    a1.set_ylabel("Molecules after standardisation")
    a1.set_title("Every dataset is small by deep-learning standards", fontsize=9)
    cl = st.dropna(subset=["pos_rate"])
    a2.bar(cl.dataset, cl.pos_rate * 100, color=ACCENT)
    for x, v in zip(cl.dataset, cl.pos_rate * 100):
        a2.text(x, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=7)
    a2.set_ylabel("Mean active rate, %")
    a2.set_title("Actives are rare (Tox21)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_dataset_stats.png")
    plt.close(fig)


def protocol_comparison():
    """Table 'the split protocol matters more than the model': same models, two protocols."""
    try:
        bal, det = load_raw("balanced"), load_raw("deterministic")
    except SystemExit:
        return None
    if det.empty:
        return None
    rows = []
    for name in DATASET_ORDER:
        if DATASETS[name].task_type != "classification":
            continue
        b = bal[(bal.dataset == name) & (bal.seed == 0)]
        d = det[det.dataset == name]
        keys = sorted(set(b.key) & set(d.key))
        if not keys:
            continue
        for k in keys:
            bv = b[b.key == k].roc_auc.mean()
            dv = d[d.key == k].roc_auc.mean()
            rows.append({"dataset": name, "model": k, "balanced_roc_auc": round(bv, 3),
                         "deterministic_roc_auc": round(dv, 3), "difference": round(bv - dv, 3)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(RESULTS / "protocol_comparison.csv", index=False)
    return out


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_raw("balanced")
    summary = summarise(df)
    summary.to_csv(RESULTS / "benchmark_summary.csv", index=False)
    write_pivots(summary)
    w = winners(summary)
    fig_representation(summary)
    fig_imbalance(df)
    fig_tox21_tasks(df, summary)
    fig_dataset_stats()
    pc = protocol_comparison()
    fig_protocol(pc)
    if pc is not None and not pc.empty:
        print("\nSplit-protocol comparison (same models, two scaffold protocols):")
        print(pc.to_string(index=False))
    print(f"\n{len(df)} per-task rows, {df.key.nunique()} models, {df.dataset.nunique()} datasets")
    cols = [c for c in ["dataset", "key", "roc_auc", "pr_auc", "mcc", "rmse", "r2"] if c in summary]
    print(summary[cols].to_string(index=False))
    print("\nRepresentation winners:")
    print(w.to_string(index=False) if not w.empty else "(waiting for both families on a dataset)")


if __name__ == "__main__":
    main()
