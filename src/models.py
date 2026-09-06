"""
Three models for the PoC, exactly matching supervisor-approved scope:
  1. RandomForestBaseline  - no graph, feature-only
  2. StandardGCN           - static, equal trust to all neighbours
  3. NeighbourAwareGCN     - reliability-reweighted message passing

FIXES applied vs. original:
  1. reliability clamp -> rescale (no edge fully zeroed out)
  2. added self-loops (each node keeps its own signal, like GCNConv does)
  3. switched aggr to "mean" so node degree doesn't distort message scale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops
from sklearn.ensemble import RandomForestClassifier


# ---------- 1. Baseline: Random Forest (no graph) ----------
def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    return clf


# ---------- 2. Standard GCN (static, equal-trust baseline) ----------
class StandardGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ---------- 3. Neighbour-Aware GCN (the core new contribution) ----------
class ReliabilityWeightedConv(MessagePassing):
    """
    Like GCNConv, but scales each edge's message by a reliability score
    computed as cosine similarity between the two connected nodes' raw
    features. Recalculated fresh each forward pass (not fixed after training),
    which is what makes this "drift-triggered" rather than static like
    CARE-GNN/PC-GNN/GraphConsis/H2-FDetector.

    FIX: uses mean aggregation (not raw add) so node degree doesn't distort
    message scale, and adds self-loops with reliability=1.0 so each node
    always retains its own signal, matching GCNConv's behaviour.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="mean")
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, raw_features):
        # Add self-loops so every node keeps some of its own signal,
        # same as GCNConv does by default.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        reliability = F.cosine_similarity(raw_features[row], raw_features[col], dim=-1)

        # FIX: rescale instead of clamp-to-zero, so no edge is fully discarded.
        # [-1, 1] -> [0, 1]
        reliability = (reliability + 1) / 2

        # Self-loop edges (row == col) get full trust (reliability = 1.0),
        # since add_self_loops appends them at the end with feature similarity 1.0
        # already (a node is maximally similar to itself), so no extra step needed.

        x = self.lin(x)
        return self.propagate(edge_index, x=x, reliability=reliability)

    def message(self, x_j, reliability):
        return x_j * reliability.unsqueeze(-1)


class NeighbourAwareGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2):
        super().__init__()
        self.conv1 = ReliabilityWeightedConv(in_channels, hidden_channels)
        self.conv2 = ReliabilityWeightedConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        raw = x  # keep original features for reliability scoring at every layer
        h = F.relu(self.conv1(x, edge_index, raw))
        h = F.dropout(h, p=0.5, training=self.training)
        out = self.conv2(h, edge_index, raw)
        return out