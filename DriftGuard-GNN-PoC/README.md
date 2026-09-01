# DriftGuard-GNN — Quick Proof of Concept

**SE4012 Research Project | IT23223394**

## Purpose
Test whether static neighbour-trust assumptions in graph-based fraud detection
(as used in CARE-GNN, PC-GNN, GraphConsis, H2-FDetector) break down under
temporal drift, and whether a drift-triggered, recalculated neighbour-reliability
score recovers lost performance.

## Supervisor-approved scope (do not expand beyond this for the PoC)
- Dataset: Elliptic Bitcoin Dataset only
- Models: (1) Random Forest baseline, (2) standard GCN (static, equal trust),
  (3) neighbour-aware GCN (reliability reweighting, recalculated per timestep)

## Hypothesis
H1: Standard GCN performance degrades more in high-drift periods than in
low-drift periods.
H2: Neighbour-aware GCN recovers some of that degradation, relative to
standard GCN, in the high-drift period specifically.

## Neighbour reliability — working definition
For an edge (i, j):

    reliability(i, j) = cosine_similarity(x_i, x_j)

recalculated at every timestep (not fixed once at training time), where
x_i and x_j are the raw feature vectors of transactions i and j.

This reliability score scales that edge's contribution during GCN message
passing (see `src/models.py`).

## Drift period definition
Following Maganti (2026), the Elliptic dataset's known ~39x fraud-rate drop
occurs late in its 49-timestep range. We split:
- **Low-drift period**: timesteps 1–34 (training window)
- **High-drift period**: timesteps 35–49 (test window, post-shift)

This matches Maganti's own train/test split, keeping our PoC directly
comparable to the published rigor benchmark.

## Repository structure
```
DriftGuard-GNN-PoC/
├── data/                # Elliptic CSVs go here (not committed — see .gitignore)
├── src/
│   ├── data_loader.py   # loads + builds strict-inductive graph splits
│   ├── models.py        # RF baseline, standard GCN, neighbour-aware GCN
│   ├── train.py         # training loop for both GCN variants
│   └── evaluate.py      # F1/AUPRC evaluation, low-drift vs high-drift comparison
├── notebooks/
│   └── poc_run.ipynb    # Colab-friendly notebook tying it all together
├── results/             # output metrics/plots land here
├── requirements.txt
└── README.md
```

## Status log
- [ ] Data loaded, strict-inductive split rebuilt (no test-period leakage)
- [ ] Random Forest baseline trained and evaluated
- [ ] Standard GCN trained and evaluated
- [ ] Neighbour-aware GCN (reliability reweighting) implemented and evaluated
- [ ] Low-drift vs high-drift comparison complete
- [ ] Results written up for supervisor update
