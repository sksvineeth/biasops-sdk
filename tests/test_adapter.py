"""Tests for biasops.adapter — fairlearn metric collection."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from biasops.adapter import collect, LOW_SAMPLE_THRESHOLD


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_data(n=200, seed=42):
    rng = np.random.RandomState(seed)
    groups = rng.choice(["male", "female"], size=n)
    y_true = rng.randint(0, 2, size=n)
    y_pred = y_true.copy()
    # flip 10% of predictions to create imperfect accuracy
    flip = rng.choice(n, size=n // 10, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    return y_true, y_pred, groups


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCollectBasic:
    def test_returns_dict(self):
        y_true, y_pred, sf = _make_data()
        result = collect(y_true, y_pred, sf)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        y_true, y_pred, sf = _make_data()
        result = collect(y_true, y_pred, sf)
        required = {
            "adverse_impact_ratio", "demographic_parity_gap",
            "min_group_selection_rate", "overall_accuracy",
            "worst_group_accuracy", "max_group_selection_rate",
            "worst_selection_rate_group", "best_selection_rate_group",
            "worst_accuracy_group", "per_group",
            "low_sample_warning", "small_groups",
        }
        assert required.issubset(result.keys())

    def test_air_between_zero_and_one(self):
        y_true, y_pred, sf = _make_data()
        result = collect(y_true, y_pred, sf)
        air = result["adverse_impact_ratio"]
        assert air is not None
        assert 0.0 <= air <= 1.5  # can exceed 1.0 if minority favoured

    def test_overall_accuracy_reasonable(self):
        y_true, y_pred, sf = _make_data()
        result = collect(y_true, y_pred, sf)
        assert 0.85 <= result["overall_accuracy"] <= 1.0  # 10% flip

    def test_per_group_has_both_groups(self):
        y_true, y_pred, sf = _make_data()
        result = collect(y_true, y_pred, sf)
        assert "male" in result["per_group"]
        assert "female" in result["per_group"]

    def test_per_group_fields(self):
        y_true, y_pred, sf = _make_data()
        result = collect(y_true, y_pred, sf)
        for group in result["per_group"].values():
            assert "selection_rate" in group
            assert "accuracy" in group
            assert "n" in group
            assert group["n"] > 0

    def test_values_are_rounded(self):
        y_true, y_pred, sf = _make_data()
        result = collect(y_true, y_pred, sf)
        # all floats should be 4 decimal places
        air = result["adverse_impact_ratio"]
        assert air == round(air, 4)

    def test_no_low_sample_warning_with_enough_data(self):
        y_true, y_pred, sf = _make_data(n=200)
        result = collect(y_true, y_pred, sf)
        assert result["low_sample_warning"] is False
        assert result["small_groups"] == []

    def test_low_sample_warning_fires(self):
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 1, 1]
        sf = ["a", "a", "a", "b", "b"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = collect(y_true, y_pred, sf)
            assert result["low_sample_warning"] is True
            assert len(result["small_groups"]) > 0

    def test_identical_predictions_air_is_one(self):
        n = 100
        y_true = np.ones(n, dtype=int)
        y_pred = np.ones(n, dtype=int)
        sf = np.array(["a"] * 50 + ["b"] * 50)
        result = collect(y_true, y_pred, sf)
        assert result["adverse_impact_ratio"] == 1.0
