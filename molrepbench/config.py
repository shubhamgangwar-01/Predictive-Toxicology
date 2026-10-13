"""Central configuration: paths, dataset specifications and endpoint metadata."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"

DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    url: str
    smiles_col: str
    tasks: tuple
    task_type: str  # "classification" | "regression"
    description: str = ""
    # Human readable meaning of a positive label / regression target per task
    task_labels: dict = field(default_factory=dict)


DATASETS = {
    "tox21": DatasetSpec(
        name="tox21",
        filename="tox21.csv",
        url=f"{DATA_URL}/tox21.csv.gz",
        smiles_col="smiles",
        tasks=(
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
        ),
        task_type="classification",
        description="Tox21 2014 challenge: 12 nuclear-receptor (NR) and stress-response (SR) assays.",
        task_labels={
            "NR-AR": "Androgen receptor agonism",
            "NR-AR-LBD": "Androgen receptor (ligand-binding domain)",
            "NR-AhR": "Aryl hydrocarbon receptor activation",
            "NR-Aromatase": "Aromatase inhibition",
            "NR-ER": "Estrogen receptor alpha agonism",
            "NR-ER-LBD": "Estrogen receptor (ligand-binding domain)",
            "NR-PPAR-gamma": "PPAR-gamma activation",
            "SR-ARE": "Oxidative stress (antioxidant response element)",
            "SR-ATAD5": "Genotoxicity (ATAD5)",
            "SR-HSE": "Heat-shock stress response",
            "SR-MMP": "Mitochondrial membrane potential disruption",
            "SR-p53": "DNA damage (p53 activation)",
        },
    ),
    "clintox": DatasetSpec(
        name="clintox",
        filename="clintox.csv",
        url=f"{DATA_URL}/clintox.csv.gz",
        smiles_col="smiles",
        tasks=("FDA_APPROVED", "CT_TOX"),
        task_type="classification",
        description="ClinTox: FDA-approved drugs vs. drugs that failed clinical trials for toxicity.",
        task_labels={
            "FDA_APPROVED": "Resembles FDA-approved drugs",
            "CT_TOX": "Clinical-trial toxicity failure",
        },
    ),
    "bbbp": DatasetSpec(
        name="bbbp",
        filename="BBBP.csv",
        url=f"{DATA_URL}/BBBP.csv",
        smiles_col="smiles",
        tasks=("p_np",),
        task_type="classification",
        description="BBBP: blood-brain barrier penetration (1 = penetrant).",
        task_labels={"p_np": "Blood-brain barrier penetration"},
    ),
    "esol": DatasetSpec(
        name="esol",
        filename="delaney-processed.csv",
        url=f"{DATA_URL}/delaney-processed.csv",
        smiles_col="smiles",
        tasks=("measured log solubility in mols per litre",),
        task_type="regression",
        description="ESOL (Delaney): measured aqueous solubility, log10(mol/L).",
        task_labels={"measured log solubility in mols per litre": "Aqueous solubility, logS (mol/L)"},
    ),
}

# Short display names for tasks (ESOL's column name is long)
TASK_SHORT = {"measured log solubility in mols per litre": "logS", "p_np": "BBB+"}

SEEDS = (0, 1, 2)
SPLIT_FRAC = (0.8, 0.1, 0.1)

# Model registry: (model family, feature set)
DESCRIPTOR_MODELS = [
    ("RF", "ecfp"), ("RF", "rdkit"),
    ("XGB", "ecfp"), ("XGB", "rdkit"),
    ("DNN", "ecfp"), ("DNN", "rdkit"),
]
GNN_MODELS = ["GCN", "GAT", "AttentiveFP"]
