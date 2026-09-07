"""
Full PoC training pipeline: converts loaded Elliptic data into PyTorch
Geometric format, trains all 3 models, evaluates on low-drift vs
high-drift periods.

Run this after data_loader.py has loaded merged/edges successfully.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score

from data_loader import load_elliptic, build_strict_inductive_splits, TRAIN_TIMESTEPS, TEST_TIMESTEPS
from models import train_random_forest, StandardGCN, NeighbourAwareGCN
from evaluate import evaluate_predictions, compare_across_drift


def build_pyg_graph(df, edges_df, id_to_idx=None):
    """Convert a dataframe of nodes + an edgelist into PyG-ready tensors."""
    df = df.reset_index(drop=True)
    if id_to_idx is None:
        id_to_idx = {tx_id: i for i, tx_id in enumerate(df["txId"])}

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float)

    # Keep only edges where both endpoints exist in this node set
    mask = edges_df["txId1"].isin(id_to_idx) & edges_df["txId2"].isin(id_to_idx)
    valid_edges = edges_df[mask]
    src = valid_edges["txId1"].map(id_to_idx).values
    dst = valid_edges["txId2"].map(id_to_idx).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    y = torch.tensor(df["label"].fillna(-1).values, dtype=torch.long)
    labelled_mask = torch.tensor(df["label"].notna().values)

    return x, edge_index, y, labelled_mask, id_to_idx


def train_gcn(model, x, edge_index, y, labelled_mask, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # class-balanced loss weight, since illicit is a small minority
    n_pos = (y[labelled_mask] == 1).sum().item()
    n_neg = (y[labelled_mask] == 0).sum().item()
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[labelled_mask], y[labelled_mask])
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"  epoch {epoch}: loss={loss.item():.4f}")
    return model


def get_probs(model, x, edge_index, labelled_mask):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        probs = torch.softmax(out, dim=1)[:, 1]
    return probs[labelled_mask].numpy()


def run_poc(data_dir="data"):
    print("=== Loading data ===")
    merged, edges = load_elliptic(data_dir)
    train_df, test_df, train_edges = build_strict_inductive_splits(merged, edges)

    # Split test period further into low-drift tail (34-40) vs high-drift (41-49)
    # to sharpen the comparison, per the README's finer-grained plan
    low_drift_test = test_df[test_df["timestep"].isin(range(34, 41))]
    high_drift_test = test_df[test_df["timestep"].isin(range(41, 50))]

    print(f"\nLow-drift test window: {len(low_drift_test)} nodes, "
          f"illicit rate={low_drift_test['label'].mean():.4f}")
    print(f"High-drift test window: {len(high_drift_test)} nodes, "
          f"illicit rate={high_drift_test['label'].mean():.4f}")

    # ---------- Random Forest baseline ----------
    print("\n=== Training Random Forest ===")
    feature_cols = [c for c in train_df.columns if c.startswith("feat_")]
    train_labelled = train_df[train_df["label"].notna()]
    rf = train_random_forest(train_labelled[feature_cols], train_labelled["label"])

    rf_results = {}
    for name, subset in [("low_drift", low_drift_test), ("high_drift", high_drift_test)]:
        sub_labelled = subset[subset["label"].notna()]
        probs = rf.predict_proba(sub_labelled[feature_cols])[:, 1]
        rf_results[name] = evaluate_predictions(sub_labelled["label"].values, probs)

    # ---------- Build full graph for GCN training (train + both test windows) ----------
    print("\n=== Building graph tensors ===")
    full_df = pd.concat([train_df, low_drift_test, high_drift_test]).reset_index(drop=True)
    x, edge_index, y, labelled_mask, id_to_idx = build_pyg_graph(full_df, edges)

    train_node_mask = torch.tensor(full_df["timestep"].isin(TRAIN_TIMESTEPS).values) & labelled_mask
    low_drift_mask = torch.tensor(full_df["timestep"].isin(range(34, 41)).values) & labelled_mask
    high_drift_mask = torch.tensor(full_df["timestep"].isin(range(41, 50)).values) & labelled_mask

        # --- NEW: check feature normalization ---
    print("\n=== Feature stats (raw, before normalization) ===")
    print("Min:", x.min().item(), "Max:", x.max().item())
    print("Mean:", x.mean().item(), "Std:", x.std().item())
    print("Per-feature std range:", x.std(dim=0).min().item(), "to", x.std(dim=0).max().item())

    # --- NEW: standardize features (fit on train nodes only — avoids leakage) ---
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x[train_node_mask].numpy())
    x = torch.tensor(scaler.transform(x.numpy()), dtype=torch.float)


    # ---------- Standard GCN ----------
    print("\n=== Training Standard GCN ===")
    gcn = StandardGCN(in_channels=x.shape[1])
    gcn = train_gcn(gcn, x, edge_index, y, train_node_mask)

    gcn_results = {
        "low_drift": evaluate_predictions(y[low_drift_mask].numpy(), get_probs(gcn, x, edge_index, low_drift_mask)),
        "high_drift": evaluate_predictions(y[high_drift_mask].numpy(), get_probs(gcn, x, edge_index, high_drift_mask)),
    }
    
    # ---------- Neighbour-Aware GCN ----------
    print("\n=== Training Neighbour-Aware GCN ===")
    na_gcn = NeighbourAwareGCN(in_channels=x.shape[1])
    na_gcn = train_gcn(na_gcn, x, edge_index, y, train_node_mask)

    na_gcn_results = {
        "low_drift": evaluate_predictions(y[low_drift_mask].numpy(), get_probs(na_gcn, x, edge_index, low_drift_mask)),
        "high_drift": evaluate_predictions(y[high_drift_mask].numpy(), get_probs(na_gcn, x, edge_index, high_drift_mask)),
    }

        # --- NEW: diagnostic check before trusting H2 ---
    std_low_f1 = gcn_results["low_drift"]["f1"]
    na_low_f1 = na_gcn_results["low_drift"]["f1"]
    gap = std_low_f1 - na_low_f1
    print("\n=== DIAGNOSTIC CHECK ===")
    print(f"Standard GCN low-drift F1:        {std_low_f1:.4f}")
    print(f"Neighbour-Aware GCN low-drift F1: {na_low_f1:.4f}")
    print(f"Gap: {gap:.4f}  -> {'OK, proceed to trust H2' if gap < 0.15 else 'STILL BROKEN, do not trust H2 yet'}")

    # ---------- Compare ----------
    print("\n=== RESULTS ===")
    all_results = {
        "Random Forest": rf_results,
        "Standard GCN": gcn_results,
        "Neighbour-Aware GCN": na_gcn_results,
    }
    compare_across_drift(all_results)
    return all_results, gcn, na_gcn, x, edge_index, y, low_drift_mask, high_drift_mask


if __name__ == "__main__":
    run_poc()
