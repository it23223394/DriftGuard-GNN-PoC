"""
Loads the Elliptic Bitcoin Dataset and builds a strict-inductive,
leakage-free train/test split following Maganti (2026)'s protocol.

Expected files in data/ (from the Kaggle Elliptic dataset):
    elliptic_txs_features.csv
    elliptic_txs_classes.csv
    elliptic_txs_edgelist.csv
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

# Maganti's split point: train on early timesteps, test on the drift period
TRAIN_TIMESTEPS = list(range(1, 35))   # low-drift / pre-shift
TEST_TIMESTEPS = list(range(35, 50))   # high-drift / post-shift


def load_elliptic(data_dir="data"):
    """Load raw Elliptic CSVs and merge into a single dataframe."""
    features = pd.read_csv(f"{data_dir}/elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(f"{data_dir}/elliptic_txs_classes.csv")
    edges = pd.read_csv(f"{data_dir}/elliptic_txs_edgelist.csv")

    # First column = txId, second column = timestep, rest = 165 features
    features.columns = ["txId", "timestep"] + [f"feat_{i}" for i in range(165)]
    classes.columns = ["txId", "class"]

    # class: 1 = illicit, 2 = licit, "unknown" = unlabeled
    merged = features.merge(classes, on="txId", how="left")
    merged["label"] = merged["class"].map({"1": 1, "2": 0, 1: 1, 2: 0})

    return merged, edges


def build_strict_inductive_splits(merged, edges):
    """
    Build train graph (timesteps 1-34 only) and full graph (all timesteps),
    ensuring the training graph never contains test-period nodes or edges.
    This is the key leakage-free requirement from Maganti (2026).
    """
    train_df = merged[merged["timestep"].isin(TRAIN_TIMESTEPS)]
    test_df = merged[merged["timestep"].isin(TEST_TIMESTEPS)]

    train_ids = set(train_df["txId"])
    # Only keep edges where BOTH endpoints are in the training window —
    # this is what prevents test-period structure leaking into training
    train_edges = edges[
        edges["txId1"].isin(train_ids) & edges["txId2"].isin(train_ids)
    ]

    print(f"Train nodes: {len(train_df)}, Train edges: {len(train_edges)}")
    print(f"Test nodes (high-drift): {len(test_df)}")
    print(f"Train labelled: {train_df['label'].notna().sum()}, "
          f"illicit rate: {train_df['label'].mean():.4f}")
    print(f"Test labelled: {test_df['label'].notna().sum()}, "
          f"illicit rate: {test_df['label'].mean():.4f}  <-- compare to train rate")

    return train_df, test_df, train_edges


if __name__ == "__main__":
    merged, edges = load_elliptic()
    train_df, test_df, train_edges = build_strict_inductive_splits(merged, edges)
    print("\nIf test illicit rate is dramatically lower than train illicit rate,")
    print("that confirms the ~39x drift documented in Maganti (2026).")
