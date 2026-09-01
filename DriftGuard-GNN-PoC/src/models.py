"""
Three models for the PoC, exactly matching supervisor-approved scope:
  1. RandomForestBaseline  - no graph, feature-only
  2. StandardGCN           - static, equal trust to all neighbours
  3. NeighbourAwareGCN     - reliability-reweighted message passing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
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
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, raw_features):
        # raw_features: original (unprojected) node features, used only
        # for computing reliability, not for the message content itself
        row, col = edge_index
        reliability = F.cosine_similarity(raw_features[row], raw_features[col], dim=-1)
        reliability = torch.clamp(reliability, min=0.0)  # negative similarity -> zero trust

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
