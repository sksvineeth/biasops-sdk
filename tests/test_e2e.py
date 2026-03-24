"""End-to-end tests for biasops.eval() — synthetic hiring dataset."""
from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from biasops.biasops import eval as biasops_eval, BiasOpsBlockError


# ---------------------------------------------------------------------------
# Synthetic hiring dataset
# ---------------------------------------------------------------------------
# skill_score is correlated with gender:
#   males   draw from Uniform(0.45, 0.85)
#   females draw from Uniform(0.20, 0.65)
# This creates a biased model that will fail AIR < 0.80.
# DataFrame is shuffled before train/test split to ensure gender balance.
# ---------------------------------------------------------------------------

@pytest.fixture()
def hiring_data():
    rng = np.random.RandomState(42)
    n = 400

    gender = rng.choice(["male", "female"], size=n)
    experience = rng.uniform(0, 20, size=n)

    # Biased skill score — males get much higher range to guarantee AIR < 0.80
    skill = np.where(
        gender == "male",
        rng.uniform(0.55, 0.95, size=n),
        rng.uniform(0.05, 0.45, size=n),
    )

    # Hiring decision based on skill + experience (biased inputs → biased labels)
    score = 0.7 * skill + 0.3 * (experience / 20)
    hired = (score >= 0.50).astype(int)

    df = pd.DataFrame({
        "gender": gender,
        "experience": experience,
        "skill_score": skill,
        "hired": hired,
    })

    # Shuffle before split — critical for gender balance in train/test
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split = int(len(df) * 0.7)
    train = df.iloc[:split]
    test  = df.iloc[split:]

    features = ["experience", "skill_score"]
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(train[features], train["hired"])

    X_test = test[["gender", "experience", "skill_score"]].reset_index(drop=True)
    y_test = test["hired"].reset_index(drop=True)
    y_pred = pd.Series(model.predict(test[features]), name="hired").reset_index(drop=True)

    return model, X_test, y_test, y_pred


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvalEndToEnd:
    def test_raises_block_on_biased_model(self, hiring_data):
        model, X_test, y_test, y_pred = hiring_data
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(BiasOpsBlockError) as exc_info:
                biasops_eval(
                    model=model, X_test=X_test, y_test=y_test, y_pred=y_pred,
                    policies="eeoc_title7_hiring_disparate_impact",
                    protected_cols=["gender"],
                    output_dir=tmpdir,
                )
            assert len(exc_info.value.violations) > 0
            assert exc_info.value.artifact_path is not None

    def test_warn_only_does_not_raise(self, hiring_data):
        model, X_test, y_test, y_pred = hiring_data
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = biasops_eval(
                model=model, X_test=X_test, y_test=y_test, y_pred=y_pred,
                policies="eeoc_title7_hiring_disparate_impact",
                protected_cols=["gender"],
                output_dir=tmpdir,
                warn_only=True,
            )
            assert artifact["overall_status"] in ("BLOCK", "WARN")

    def test_artifact_has_correct_structure(self, hiring_data):
        model, X_test, y_test, y_pred = hiring_data
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = biasops_eval(
                model=model, X_test=X_test, y_test=y_test, y_pred=y_pred,
                policies="eeoc_title7_hiring_disparate_impact",
                protected_cols=["gender"],
                output_dir=tmpdir,
                warn_only=True,
            )
            assert "biasops_version" in artifact
            assert "run_id" in artifact
            assert "sha256" in artifact
            assert "metrics" in artifact
            assert "rule_results" in artifact
            assert "policies_loaded" in artifact

    def test_multiple_policies(self, hiring_data):
        model, X_test, y_test, y_pred = hiring_data
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = biasops_eval(
                model=model, X_test=X_test, y_test=y_test, y_pred=y_pred,
                policies=["eeoc_title7_hiring_disparate_impact", "nyc_local_law_144"],
                protected_cols=["gender"],
                output_dir=tmpdir,
                warn_only=True,
            )
            policy_ids = [p["id"] for p in artifact["policies_loaded"]]
            assert "EEOC-TITLE7-001" in policy_ids
            assert "NYC-LL144-001" in policy_ids

    def test_missing_protected_col_raises(self, hiring_data):
        model, X_test, y_test, y_pred = hiring_data
        with pytest.raises(ValueError, match="not in X_test"):
            biasops_eval(
                model=model, X_test=X_test, y_test=y_test, y_pred=y_pred,
                policies="eeoc_title7_hiring_disparate_impact",
                protected_cols=["race"],  # not in dataframe
            )

    def test_no_protected_cols_raises(self, hiring_data):
        model, X_test, y_test, y_pred = hiring_data
        with pytest.raises(ValueError, match="protected_cols is required"):
            biasops_eval(
                model=model, X_test=X_test, y_test=y_test, y_pred=y_pred,
                policies="eeoc_title7_hiring_disparate_impact",
                protected_cols=None,
            )

    def test_string_policy_accepted(self, hiring_data):
        model, X_test, y_test, y_pred = hiring_data
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = biasops_eval(
                model=model, X_test=X_test, y_test=y_test, y_pred=y_pred,
                policies="eeoc_title7_hiring_disparate_impact",  # string, not list
                protected_cols="gender",  # string, not list
                output_dir=tmpdir,
                warn_only=True,
            )
            assert artifact is not None

    def test_string_protected_col_accepted(self, hiring_data):
        model, X_test, y_test, y_pred = hiring_data
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = biasops_eval(
                model=model, X_test=X_test, y_test=y_test, y_pred=y_pred,
                policies=["eeoc_title7_hiring_disparate_impact"],
                protected_cols="gender",
                output_dir=tmpdir,
                warn_only=True,
            )
            assert artifact["protected_cols"] == ["gender"]

    def test_block_error_has_artifact(self, hiring_data):
        model, X_test, y_test, y_pred = hiring_data
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(BiasOpsBlockError) as exc_info:
                biasops_eval(
                    model=model, X_test=X_test, y_test=y_test, y_pred=y_pred,
                    policies="eeoc_title7_hiring_disparate_impact",
                    protected_cols=["gender"],
                    output_dir=tmpdir,
                )
            assert exc_info.value.artifact is not None
            assert exc_info.value.artifact["overall_status"] == "BLOCK"

    def test_all_five_policies_load(self, hiring_data):
        model, X_test, y_test, y_pred = hiring_data
        all_policies = [
            "eeoc_title7_hiring_disparate_impact",
            "nyc_local_law_144",
            "illinois_hb3773",
            "colorado_ai_act_hiring",
            "eu_ai_act_high_risk_system",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = biasops_eval(
                model=model, X_test=X_test, y_test=y_test, y_pred=y_pred,
                policies=all_policies,
                protected_cols=["gender"],
                output_dir=tmpdir,
                warn_only=True,
            )
            loaded_ids = [p["id"] for p in artifact["policies_loaded"]]
            assert len(loaded_ids) == 5
