"""Tests for biasops.evaluator — policy loading and rule evaluation."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from biasops.evaluator import load_policy, evaluate, _is_effective


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

class TestLoadPolicy:
    def test_load_bundled_eeoc(self):
        policy = load_policy("eeoc_title7_hiring_disparate_impact")
        assert policy["id"] == "EEOC-TITLE7-001"

    def test_load_bundled_nyc(self):
        policy = load_policy("nyc_local_law_144")
        assert policy["id"] == "NYC-LL144-001"

    def test_load_bundled_illinois(self):
        policy = load_policy("illinois_hb3773")
        assert policy["id"] == "IL-HB3773-001"

    def test_load_bundled_colorado(self):
        policy = load_policy("colorado_ai_act_hiring")
        assert policy["id"] == "CO-SB24205-001"

    def test_load_bundled_eu_ai_act(self):
        policy = load_policy("eu_ai_act_high_risk_system")
        assert policy["id"] == "EU-AI-ACT-001"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_policy("nonexistent_policy_xyz")

    def test_load_from_custom_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_data = {
                "id": "TEST-001", "name": "Test", "version": "1.0.0",
                "rules": [{"id": "R1", "name": "check", "type": "numeric",
                            "metric_id": "overall_accuracy", "threshold": 0.5,
                            "operator": ">=", "severity": "BLOCK"}],
            }
            path = Path(tmpdir) / "test_policy.yaml"
            with open(path, "w") as f:
                yaml.dump(policy_data, f)
            loaded = load_policy("test_policy", policy_dir=tmpdir)
            assert loaded["id"] == "TEST-001"


# ---------------------------------------------------------------------------
# Effective date
# ---------------------------------------------------------------------------

class TestEffectiveDate:
    def test_no_effective_date_is_effective(self):
        assert _is_effective({}) is True

    def test_past_date_is_effective(self):
        assert _is_effective({"effective_date": "2020-01-01"}) is True

    def test_future_date_not_effective(self):
        assert _is_effective({"effective_date": "2099-12-31"}) is False


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------

def _simple_policy(threshold=0.80, operator=">=", severity="BLOCK", metric_id="adverse_impact_ratio"):
    return {
        "id": "TEST-001", "name": "Test Policy", "version": "1.0.0",
        "rules": [{
            "id": "R1", "name": "AIR check", "type": "numeric",
            "metric_id": metric_id, "threshold": threshold,
            "operator": operator, "severity": severity,
        }],
    }


class TestEvaluate:
    def test_pass_when_above_threshold(self):
        metrics = {"adverse_impact_ratio": 0.85}
        results = evaluate(metrics, [_simple_policy(threshold=0.80)])
        assert len(results) == 1
        assert results[0]["passed"] is True
        assert results[0]["status"] == "PASS"

    def test_block_when_below_threshold(self):
        metrics = {"adverse_impact_ratio": 0.65}
        results = evaluate(metrics, [_simple_policy(threshold=0.80)])
        assert len(results) == 1
        assert results[0]["passed"] is False
        assert results[0]["status"] == "BLOCK"

    def test_pass_at_exact_threshold(self):
        metrics = {"adverse_impact_ratio": 0.80}
        results = evaluate(metrics, [_simple_policy(threshold=0.80)])
        assert results[0]["passed"] is True

    def test_warn_severity(self):
        metrics = {"adverse_impact_ratio": 0.65}
        results = evaluate(metrics, [_simple_policy(threshold=0.80, severity="WARN")])
        assert results[0]["status"] == "WARN"

    def test_skipped_when_metric_missing(self):
        metrics = {}
        results = evaluate(metrics, [_simple_policy()])
        assert results[0]["status"] == "SKIPPED"
        assert results[0]["passed"] is None

    def test_less_than_operator(self):
        metrics = {"demographic_parity_gap": 0.05}
        policy = _simple_policy(threshold=0.10, operator="<=", metric_id="demographic_parity_gap")
        results = evaluate(metrics, [policy])
        assert results[0]["passed"] is True

    def test_multiple_policies(self):
        metrics = {"adverse_impact_ratio": 0.90, "overall_accuracy": 0.75}
        p1 = _simple_policy(threshold=0.80, metric_id="adverse_impact_ratio")
        p2 = _simple_policy(threshold=0.70, metric_id="overall_accuracy")
        p2["id"] = "TEST-002"
        results = evaluate(metrics, [p1, p2])
        assert len(results) == 2
        assert all(r["passed"] is True for r in results)

    def test_future_policy_skipped(self):
        policy = _simple_policy()
        policy["effective_date"] = "2099-12-31"
        metrics = {"adverse_impact_ratio": 0.50}
        results = evaluate(metrics, [policy])
        assert len(results) == 0

    def test_passed_is_python_bool_not_numpy(self):
        """Ensure bool() cast avoids numpy.bool_ identity issues."""
        import numpy as np
        metrics = {"adverse_impact_ratio": np.float64(0.85)}
        results = evaluate(metrics, [_simple_policy(threshold=0.80)])
        assert type(results[0]["passed"]) is bool

    def test_metric_ids_multi_binding(self):
        policy = {
            "id": "MULTI-001", "name": "Multi", "version": "1.0.0",
            "rules": [{
                "id": "R1", "name": "Multi check", "type": "numeric",
                "threshold": 0.80, "operator": ">=", "severity": "BLOCK",
                "metric_ids": [
                    {"id": "air", "metric_id": "adverse_impact_ratio"},
                    {"id": "acc", "metric_id": "overall_accuracy"},
                ],
            }],
        }
        metrics = {"adverse_impact_ratio": 0.85, "overall_accuracy": 0.90}
        results = evaluate(metrics, [policy])
        assert len(results) == 2
        assert all(r["passed"] is True for r in results)
