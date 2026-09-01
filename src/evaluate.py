"""
Evaluates all three models on low-drift vs high-drift test periods,
and reports the comparison that answers the PoC's core question:

  Does the standard GCN degrade more under drift than the neighbour-aware GCN?
"""

from sklearn.metrics import f1_score, average_precision_score
import pandas as pd


def evaluate_predictions(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auprc = average_precision_score(y_true, y_pred_proba)
    return {"f1": f1, "auprc": auprc}


def compare_across_drift(results_dict):
    """
    results_dict format:
    {
        "Random Forest": {"low_drift": {...}, "high_drift": {...}},
        "Standard GCN": {"low_drift": {...}, "high_drift": {...}},
        "Neighbour-Aware GCN": {"low_drift": {...}, "high_drift": {...}},
    }
    """
    rows = []
    for model_name, periods in results_dict.items():
        low = periods["low_drift"]
        high = periods["high_drift"]
        f1_drop = low["f1"] - high["f1"]
        rows.append({
            "Model": model_name,
            "F1 (low-drift)": round(low["f1"], 4),
            "F1 (high-drift)": round(high["f1"], 4),
            "F1 drop": round(f1_drop, 4),
            "AUPRC (low-drift)": round(low["auprc"], 4),
            "AUPRC (high-drift)": round(high["auprc"], 4),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv("results/drift_comparison.csv", index=False)

    print("\n--- Interpretation guide ---")
    print("H1 confirmed if: Standard GCN's F1 drop > Random Forest's F1 drop")
    print("H2 confirmed if: Neighbour-Aware GCN's F1 drop < Standard GCN's F1 drop")
    return df
