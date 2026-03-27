"""Tests for biasops.artifact — audit artifact builder."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path


from biasops.artifact import build, write, BIASOPS_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics():
    return {
        "adverse_impact_ratio": 0.72,
        "demographic_parity_gap": 0.15,
        "min_group_selection_rate": 0.35,
        "overall_accuracy": 0.88,
        "worst_group_accuracy": 0.82,
        "max_group_selection_rate": 0.50,
        "worst_selection_rate_group": "female",
        "best_selection_rate_group": "male",
        "worst_accuracy_group": "female",
        "per_group": {"male": {"selection_rate": 0.50, "accuracy": 0.90, "n": 100},
                      "female": {"selection_rate": 0.35, "accuracy": 0.82, "n": 100}},
        "low_sample_warning": False,
        "small_groups": [],
    }

def _rule_results_pass():
    return [
        {"rule_id": "R1", "policy_id": "TEST-001", "name": "AIR check",
         "metric_id": "adverse_impact_ratio", "value": 0.85, "threshold": 0.80,
         "operator": ">=", "status": "PASS", "severity": "BLOCK", "passed": True,
         "article": "", "citation": "", "confidence": "?", "stage": ""},
    ]

def _rule_results_block():
    return [
        {"rule_id": "R1", "policy_id": "TEST-001", "name": "AIR check",
         "metric_id": "adverse_impact_ratio", "value": 0.65, "threshold": 0.80,
         "operator": ">=", "status": "BLOCK", "severity": "BLOCK", "passed": False,
         "article": "Title VII", "citation": "EEOC Uniform Guidelines", "confidence": "high", "stage": ""},
    ]

def _policies():
    return [{"id": "TEST-001", "name": "Test", "version": "1.0.0", "jurisdiction": "US"}]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuild:
    def test_pass_status(self):
        artifact = build(_metrics(), _rule_results_pass(), _policies(), ["gender"])
        assert artifact["overall_status"] == "PASS"
        assert artifact["block_count"] == 0

    def test_block_status(self):
        artifact = build(_metrics(), _rule_results_block(), _policies(), ["gender"])
        assert artifact["overall_status"] == "BLOCK"
        assert artifact["block_count"] == 1

    def test_has_sha256(self):
        artifact = build(_metrics(), _rule_results_pass(), _policies(), ["gender"])
        assert "sha256" in artifact
        assert len(artifact["sha256"]) == 64

    def test_has_run_id(self):
        artifact = build(_metrics(), _rule_results_pass(), _policies(), ["gender"])
        assert len(artifact["run_id"]) == 8

    def test_has_timestamp(self):
        artifact = build(_metrics(), _rule_results_pass(), _policies(), ["gender"])
        assert "timestamp" in artifact
        assert "T" in artifact["timestamp"]

    def test_version_matches(self):
        artifact = build(_metrics(), _rule_results_pass(), _policies(), ["gender"])
        assert artifact["biasops_version"] == BIASOPS_VERSION

    def test_policies_loaded(self):
        artifact = build(_metrics(), _rule_results_pass(), _policies(), ["gender"])
        assert len(artifact["policies_loaded"]) == 1
        assert artifact["policies_loaded"][0]["id"] == "TEST-001"

    def test_excludes_internal_metric_keys(self):
        artifact = build(_metrics(), _rule_results_pass(), _policies(), ["gender"])
        metrics = artifact["metrics"]
        assert "worst_selection_rate_group" not in metrics
        assert "best_selection_rate_group" not in metrics
        assert "worst_accuracy_group" not in metrics

    def test_block_count_is_int_not_numpy(self):
        """Regression: numpy.bool_ in list comprehension gave wrong count."""
        import numpy as np
        rr = _rule_results_block()
        rr[0]["passed"] = np.bool_(False)  # simulate numpy bool
        artifact = build(_metrics(), rr, _policies(), ["gender"])
        assert isinstance(artifact["block_count"], int)
        assert artifact["block_count"] == 1


class TestWrite:
    def test_writes_json_file(self):
        artifact = build(_metrics(), _rule_results_pass(), _policies(), ["gender"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write(artifact, output_dir=tmpdir)
            assert Path(path).exists()
            with open(path) as f:
                data = json.load(f)
            assert data["run_id"] == artifact["run_id"]

    def test_filename_contains_run_id(self):
        artifact = build(_metrics(), _rule_results_pass(), _policies(), ["gender"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write(artifact, output_dir=tmpdir)
            assert artifact["run_id"] in Path(path).name
