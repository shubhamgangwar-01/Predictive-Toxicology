"""Model zoo with one interface: fit(train, val) and predict(X) -> (n_samples, n_tasks).

Descriptor models (RF, XGBoost, DNN) consume a dense feature matrix (ECFP4 or RDKit
descriptors). Graph models (GCN, GAT, Attentive FP) consume lists of PyG `Data` graphs.
Missing labels (NaN, common in Tox21) are masked out of every loss.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (AttentiveFP, GATConv, GCNConv, global_add_pool, global_max_pool,
                                global_mean_pool)
from xgboost import XGBClassifier, XGBRegressor

from .features import ATOM_DIM, BOND_DIM
from .metrics import masked_mean_auc

def _pick_device() -> torch.device:
    """CUDA when available (Colab GPU), else CPU.

    Override with MOLREP_DEVICE=cpu|cuda|mps. MPS is not selected automatically: several PyG
    scatter kernels silently fall back or misbehave on it, and these graphs are small enough that
    CPU is competitive anyway.
    """
    forced = os.environ.get("MOLREP_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = _pick_device()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ======================================================================
# Tree ensembles: one model per task, trained on that task's labelled rows
# ======================================================================
class TreeModel:
    def __init__(self, family: str, task_type: str, seed: int = 0):
        self.family, self.task_type, self.seed = family, task_type, seed
        self.models = []

    def _make(self, y_train):
        clf = self.task_type == "classification"
        if self.family == "RF":
            if clf:
                return RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                              class_weight="balanced_subsample", n_jobs=-1,
                                              random_state=self.seed)
            return RandomForestRegressor(n_estimators=500, max_features=0.33, n_jobs=-1,
                                         random_state=self.seed)
        # XGBoost; early stopping on the validation split
        common = dict(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
                      colsample_bytree=0.5, min_child_weight=1, reg_lambda=1.0, tree_method="hist",
                      n_jobs=-1, random_state=self.seed, early_stopping_rounds=50)
        if clf:
            pos = max(y_train.sum(), 1)
            return XGBClassifier(**common, eval_metric="aucpr",
                                 scale_pos_weight=float((len(y_train) - pos) / pos))
        return XGBRegressor(**common, eval_metric="rmse")

    def fit(self, X, Y, Xv, Yv):
        self.models = []
        for t in range(Y.shape[1]):
            m, mv = ~np.isnan(Y[:, t]), ~np.isnan(Yv[:, t])
            model = self._make(Y[m, t])
            if self.family == "XGB":
                model.fit(X[m], Y[m, t], eval_set=[(Xv[mv], Yv[mv, t])], verbose=False)
            else:
                model.fit(X[m], Y[m, t])
            self.models.append(model)
        return self

    def predict(self, X):
        if self.task_type == "classification":
            return np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])
        return np.column_stack([m.predict(X) for m in self.models])


# ======================================================================
# Shared neural-network training loop (masked multi-task loss)
# ======================================================================
class _NNTrainer:
    lr = 1e-3
    weight_decay = 1e-5
    max_epochs = 150
    patience = 30
    batch_size = 64

    def __init__(self, task_type: str, n_tasks: int, seed: int = 0):
        self.task_type, self.n_tasks, self.seed = task_type, n_tasks, seed
        self.y_mean, self.y_std = 0.0, 1.0
        self.history = []

    # subclasses implement: build(), _batches(data, shuffle), _forward(batch) -> (pred, y)
    def _loss(self, pred, y):
        mask = ~torch.isnan(y)
        y0 = torch.nan_to_num(y)
        if self.task_type == "classification":
            loss = F.binary_cross_entropy_with_logits(pred, y0, pos_weight=self.pos_weight, reduction="none")
        else:
            loss = (pred - y0) ** 2
        return (loss * mask).sum() / mask.sum().clamp(min=1)

    def _score(self, Yv, Pv):
        if self.task_type == "classification":
            return masked_mean_auc(Yv, Pv)
        return -float(np.sqrt(np.nanmean((Yv - Pv) ** 2)))

    def _fit_loop(self, train, val, Y, Yv):
        set_seed(self.seed)
        if self.task_type == "classification":
            pos = np.nansum(Y, 0)
            neg = np.sum(~np.isnan(Y), 0) - pos
            self.pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 50),
                                           dtype=torch.float32, device=DEVICE)
        else:
            self.y_mean, self.y_std = float(np.nanmean(Y)), float(np.nanstd(Y))
        self.net = self.build().to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8)
        best, best_state, wait = -np.inf, None, 0
        for epoch in range(self.max_epochs):
            self.net.train()
            for batch in self._batches(train, shuffle=True):
                pred, y = self._forward(batch)
                if self.task_type == "regression":
                    y = (y - self.y_mean) / self.y_std
                loss = self._loss(pred, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
            score = self._score(Yv, self._predict(val))
            sched.step(score)
            self.history.append(score)
            if score > best + 1e-4:
                best, best_state, wait = score, copy.deepcopy(self.net.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_val, self.epochs_run = best, epoch + 1
        return self

    @torch.no_grad()
    def _predict(self, data):
        self.net.eval()
        outs = []
        for batch in self._batches(data, shuffle=False):
            pred, _ = self._forward(batch)
            outs.append(pred)
        logits = torch.cat(outs)
        if self.task_type == "classification":
            return torch.sigmoid(logits).cpu().numpy()  # numerically stable
        return logits.cpu().numpy() * self.y_std + self.y_mean


# ======================================================================
# DNN on fingerprints / descriptors
# ======================================================================
class MLP(nn.Module):
    def __init__(self, d_in, n_out, hidden=(512, 128), dropout=0.3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DNNModel(_NNTrainer):
    def build(self):
        return MLP(self.d_in, self.n_tasks)

    def _batches(self, data, shuffle):
        X, Y = data
        idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
        for i in range(0, len(X), self.batch_size):
            b = idx[i:i + self.batch_size]
            if shuffle and len(b) < 2:  # BatchNorm needs > 1 sample
                continue
            yield torch.from_numpy(X[b]), torch.from_numpy(Y[b])

    def _forward(self, batch):
        x, y = batch
        return self.net(x.to(DEVICE)), y.to(DEVICE)

    def fit(self, X, Y, Xv, Yv):
        self.d_in = X.shape[1]
        X, Xv = X.astype(np.float32), Xv.astype(np.float32)
        return self._fit_loop((X, Y.astype(np.float32)), (Xv, Yv.astype(np.float32)), Y, Yv)

    def predict(self, X):
        X = X.astype(np.float32)
        return self._predict((X, np.zeros((len(X), self.n_tasks), np.float32)))


# ======================================================================
# Graph neural networks
# ======================================================================
class GCNNet(nn.Module):
    """Kipf & Welling GCN, 3 layers, sum+mean+max readout (sum keeps molecule size information)."""

    def __init__(self, n_out, hidden=128, layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(ATOM_DIM if i == 0 else hidden, hidden) for i in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_out))

    def forward(self, data, return_attention=False):
        x = data.x
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(F.relu(norm(conv(x, data.edge_index))), self.dropout, self.training)
        g = torch.cat([global_add_pool(x, data.batch), global_mean_pool(x, data.batch), global_max_pool(x, data.batch)], 1)
        return self.head(g)


class GATNet(nn.Module):
    """Velickovic et al. GAT with bond features in the attention (edge_dim), 3 layers x 4 heads."""

    def __init__(self, n_out, hidden=64, heads=4, layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(layers):
            d_in = ATOM_DIM if i == 0 else hidden * heads
            self.convs.append(GATConv(d_in, hidden, heads=heads, edge_dim=BOND_DIM, dropout=0.1))
            self.norms.append(nn.BatchNorm1d(hidden * heads))
        self.dropout = dropout
        d = hidden * heads
        self.head = nn.Sequential(nn.Linear(3 * d, d // 2), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(d // 2, n_out))

    def forward(self, data, return_attention=False):
        x, att = data.x, None
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if return_attention and i == len(self.convs) - 1:
                h, (ei, alpha) = conv(x, data.edge_index, data.edge_attr, return_attention_weights=True)
                att = (ei, alpha.mean(1))
            else:
                h = conv(x, data.edge_index, data.edge_attr)
            x = F.dropout(F.elu(norm(h)), self.dropout, self.training)
        g = torch.cat([global_add_pool(x, data.batch), global_mean_pool(x, data.batch), global_max_pool(x, data.batch)], 1)
        out = self.head(g)
        if return_attention:
            # atom importance = attention the atom receives from its neighbours (excluding self loops)
            ei, a = att
            keep = ei[0] != ei[1]
            imp = torch.zeros(x.size(0)).index_add_(0, ei[0][keep], a[keep])
            return out, imp
        return out


class AttentiveFPNet(AttentiveFP):
    """PyG Attentive FP (Xiong et al. 2019) that can also return the readout attention:
    how strongly the molecule-level 'super node' attends to each atom at the final step."""

    def forward(self, data, return_attention=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()
        row = torch.arange(batch.size(0), device=batch.device)
        mol_edges = torch.stack([row, batch], dim=0)
        out = global_add_pool(x, batch).relu_()
        alpha = None
        for _ in range(self.num_timesteps):
            h, (_, alpha) = self.mol_conv((x, out), mol_edges, return_attention_weights=True)
            h = F.dropout(F.elu_(h), p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin2(out)
        if return_attention:
            return out, alpha.view(-1)
        return out


class GNNModel(_NNTrainer):
    def __init__(self, arch: str, task_type: str, n_tasks: int, seed: int = 0):
        super().__init__(task_type, n_tasks, seed)
        self.arch = arch

    def build(self):
        if self.arch == "GCN":
            return GCNNet(self.n_tasks)
        if self.arch == "GAT":
            return GATNet(self.n_tasks)
        if self.arch == "AttentiveFP":
            return AttentiveFPNet(in_channels=ATOM_DIM, hidden_channels=200, out_channels=self.n_tasks,
                                  edge_dim=BOND_DIM, num_layers=2, num_timesteps=2, dropout=0.2)
        raise ValueError(self.arch)

    def _batches(self, graphs, shuffle):
        g = torch.Generator().manual_seed(self.seed + len(self.history))
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=shuffle,
                            generator=g if shuffle else None)
        yield from loader

    def _forward(self, batch):
        batch = batch.to(DEVICE)
        return self.net(batch), batch.y

    def fit(self, graphs, Y, graphs_val, Yv):
        return self._fit_loop(graphs, graphs_val, Y, Yv)

    def predict(self, graphs):
        return self._predict(graphs)

    @torch.no_grad()
    def atom_attention(self, graph):
        """Per-atom attention scores (sum to ~1) for a single molecule graph."""
        if self.arch == "GCN":
            raise ValueError("GCN has no attention mechanism")
        self.net.eval()
        batch = next(iter(DataLoader([graph], batch_size=1))).to(DEVICE)
        out, imp = self.net(batch, return_attention=True)
        imp = imp.cpu().numpy()
        return imp / max(imp.sum(), 1e-9)


def make_model(family: str, task_type: str, n_tasks: int, seed: int = 0):
    if family in ("RF", "XGB"):
        return TreeModel(family, task_type, seed)
    if family == "DNN":
        return DNNModel(task_type, n_tasks, seed)
    return GNNModel(family, task_type, n_tasks, seed)
