from __future__ import annotations
import warnings
import pandas as pd
from fairlearn.metrics import (MetricFrame, demographic_parity_ratio, selection_rate)
from sklearn.metrics import accuracy_score

LOW_SAMPLE_THRESHOLD = 30

def collect(y_true, y_pred, sensitive_features) -> dict:
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    sf     = pd.Series(sensitive_features).reset_index(drop=True)

    group_counts  = sf.value_counts()
    small_groups  = group_counts[group_counts < LOW_SAMPLE_THRESHOLD].index.tolist()
    low_sample_warning = len(small_groups) > 0
    if low_sample_warning:
        warnings.warn(
            f"BiasOps: groups {small_groups} have < {LOW_SAMPLE_THRESHOLD} samples. ",
            UserWarning, stacklevel=3)

    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_true, y_pred=y_pred, sensitive_features=sf)

    try:
        air = demographic_parity_ratio(y_true, y_pred, sensitive_features=sf)
    except Exception:
        air = None

    try:
        dp_gap = float(mf.difference(method="between_groups")["selection_rate"])
    except Exception:
        dp_gap = None

    by_group_sr   = mf.by_group["selection_rate"]
    by_group_acc  = mf.by_group["accuracy"]

    per_group = {}
    for group in sf.unique():
        per_group[str(group)] = {
            "selection_rate": round(float(by_group_sr.get(group, float("nan"))), 4),
            "accuracy":       round(float(by_group_acc.get(group, float("nan"))), 4),
            "n":              int((sf == group).sum()),
        }

    return {
        "adverse_impact_ratio":     round(air, 4) if air is not None else None,
        "demographic_parity_gap":   round(dp_gap, 4) if dp_gap is not None else None,
        "min_group_selection_rate": round(float(by_group_sr.min()), 4),
        "overall_accuracy":         round(float(accuracy_score(y_true, y_pred)), 4),
        "worst_group_accuracy":     round(float(by_group_acc.min()), 4),
        "max_group_selection_rate": round(float(by_group_sr.max()), 4),
        "worst_selection_rate_group": str(by_group_sr.idxmin()),
        "best_selection_rate_group":  str(by_group_sr.idxmax()),
        "worst_accuracy_group":       str(by_group_acc.idxmin()),
        "per_group":                  per_group,
        "low_sample_warning":         low_sample_warning,
        "small_groups":               [str(g) for g in small_groups],
    }
