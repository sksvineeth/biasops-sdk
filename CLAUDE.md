# BiasOps SDK — Agent Instructions

## What Is This Repo?

The pip-installable Python SDK for BiasOps. One-line integration into any ML eval loop: `biasops.eval()` computes fairness metrics, evaluates them against YAML policy rules, and produces signed audit artifacts. Blocks deployment (exit code 1) if any BLOCK rule fails.

- **PyPI**: https://pypi.org/project/biasops/
- **GitHub**: github.com/sksvineeth/biasops-sdk
- **Install**: `pip install biasops`
- **Current version**: 0.1.0

## Architecture

```
biasops-sdk/
├── biasops/                    # Main package
│   ├── __init__.py             # Public API exports: eval, BiasOpsBlockError, collect, etc.
│   ├── biasops.py              # eval() function — the public API entry point
│   ├── adapter.py              # fairlearn MetricFrame wrapper → dict of metrics
│   ├── evaluator.py            # YAML policy loader + rule evaluation engine
│   ├── artifact.py             # Audit artifact builder (JSON with SHA-256 signature)
│   ├── cli.py                  # Click CLI: `biasops check`
│   └── policies/               # 5 bundled YAML policies
│       ├── POLICIES_VERSION
│       ├── eeoc_title7_hiring_disparate_impact.yaml
│       ├── nyc_local_law_144.yaml
│       ├── illinois_hb3773.yaml
│       ├── colorado_ai_act_hiring.yaml
│       └── eu_ai_act_high_risk_system.yaml
├── tests/                      # 51 tests across 4 files
│   ├── test_adapter.py         # 10 tests — fairlearn metric collection
│   ├── test_evaluator.py       # 14 tests — policy loading + rule evaluation
│   ├── test_artifact.py        # 11 tests — artifact building + writing
│   └── test_e2e.py             # 10 tests — full eval() with synthetic hiring data
├── demo.ipynb                  # Jupyter demo: biased vs mitigated model
├── pyproject.toml              # Package config
├── README.md
├── .github/workflows/ci.yml   # CI: ruff + pytest on 3.10/3.11/3.12
└── .gitignore
```

## Data Flow

```
biasops.eval()
  │
  ├── 1. Validate inputs (protected_cols exist in X_test)
  ├── 2. adapter.collect(y_test, y_pred, sensitive_features)
  │       └── fairlearn MetricFrame → dict with 12 metric keys
  ├── 3. evaluator.load_policy(name) for each policy
  │       └── YAML search: policy_dir → bundled policies/ (rglob)
  ├── 4. evaluator.evaluate(metrics, policies, domain)
  │       └── For each rule: _normalise_rules → operator check → PASS/BLOCK/WARN/SKIPPED
  ├── 5. artifact.build(metrics, rule_results, policies, protected_cols)
  │       └── JSON with sha256 signature, run_id, timestamp
  ├── 6. artifact.write(artifact, output_dir)
  │       └── biasops-run-{run_id}.json
  └── 7. If any BLOCK and not warn_only → raise BiasOpsBlockError
```

## Key Metric IDs (adapter.py output keys)

These are the binding keys between adapter output and policy rule `metric_id`:

| Key | Description | Source |
|-----|-------------|--------|
| `adverse_impact_ratio` | Min/max group selection rate ratio (4/5ths rule) | `fairlearn.metrics.demographic_parity_ratio` |
| `demographic_parity_gap` | Absolute difference in selection rates | `MetricFrame.difference()` |
| `min_group_selection_rate` | Lowest group selection rate | `MetricFrame.by_group["selection_rate"].min()` |
| `max_group_selection_rate` | Highest group selection rate | `MetricFrame.by_group["selection_rate"].max()` |
| `overall_accuracy` | Overall prediction accuracy | `sklearn.metrics.accuracy_score` |
| `worst_group_accuracy` | Lowest group accuracy | `MetricFrame.by_group["accuracy"].min()` |
| `per_group` | Dict of per-group stats (selection_rate, accuracy, n) | Computed |
| `low_sample_warning` | True if any group has n < 30 | Computed |
| `small_groups` | List of group names with n < 30 | Computed |

Internal keys excluded from artifact: `worst_selection_rate_group`, `best_selection_rate_group`, `worst_accuracy_group`

## eval() API

```python
biasops.eval(
    model,                          # Fitted model object (for metadata)
    X_test,                         # DataFrame with protected_cols + features
    y_test,                         # Ground truth labels
    y_pred,                         # Model predictions
    policies="eeoc_title7_...",     # str or list[str] — policy names
    protected_cols=["gender"],      # str or list[str] — column names in X_test
    domain=None,                    # Optional domain for threshold overrides
    output_dir=".",                 # Where to write artifact JSON
    warn_only=False,                # If True, never raise BiasOpsBlockError
    policy_dir=None,                # Optional custom policy directory
) -> dict                           # Returns artifact dict
```

## Code Conventions

### Python Style
- `from __future__ import annotations` in every file
- Snake case for functions/variables, PascalCase for classes
- Type hints on function signatures
- No unused imports (ruff enforced)
- No multi-statement lines (ruff E701/E702)
- `bool()` cast on numpy comparison results — critical to avoid `numpy.bool_` identity issues with `is False` checks
- Use `r.get("passed") is not None and not bool(r.get("passed"))` pattern instead of `r.get("passed") is False`
- All float metrics rounded to 4 decimal places

### Testing
- pytest with `--cov=biasops`
- Synthetic hiring dataset in e2e tests uses extreme bias (males 0.55-0.95, females 0.05-0.45 skill scores) to guarantee AIR < 0.80
- Tests must not depend on network access
- 51 tests total, all must pass

### Git Rules
- NEVER add Co-authored-by trailers
- NEVER modify git config
- Conventional commits: `feat:`, `fix:`, `docs:`, `ci:`, `test:`

## CI Pipeline

GitHub Actions on push/PR to `main`:
1. Matrix: Python 3.10, 3.11, 3.12
2. `pip install -e ".[dev]"`
3. `ruff check .`
4. `pytest --cov=biasops`

## Dependencies

```
fairlearn>=0.10.0    # Bias metrics (MetricFrame, demographic_parity_ratio)
scikit-learn>=1.3.0  # accuracy_score, model training
pandas>=2.0.0        # DataFrame handling
pyyaml>=6.0          # Policy YAML parsing
click>=8.1.0         # CLI framework
joblib>=1.3.0        # Model serialization for CLI
```

Dev: `pytest>=7.0`, `pytest-cov`, `ruff`

## Common Operations

### Run tests locally
```bash
pip install -e ".[dev]"
ruff check .
pytest --cov=biasops -v
```

### Add a new bundled policy
1. Add YAML file to `biasops/policies/`
2. Add test in `test_evaluator.py::TestLoadPolicy`
3. Update `test_e2e.py::test_all_five_policies_load` if count changes
4. Bump version in `pyproject.toml` and `artifact.py::BIASOPS_VERSION`

### Publish to PyPI
```bash
pip install build twine
python -m build
twine upload dist/*
```

## Related Repos

- **biasops-policy-marketplace** (`~/biasops-policy-marketplace/`) — Source of truth for policy YAML files
- **biasops-saas** (`~/biasops-saas/`) — SaaS platform
- **biasOps-site** (`~/biasOps-site/`) — Landing page

## Important Warnings

- Running from `~/biasops-policy-marketplace/` will shadow this package's `biasops` module with the marketplace's `biasops/` package. Always test from a different directory or a clean venv.
- `biasops.eval` shadows Python's builtin `eval` — intentional, imported as `from biasops import eval as biasops_eval` in tests.
- The `BiasOpsBlockError` contains the full artifact dict and artifact path for programmatic access.
- `BIASOPS_VERSION` in `artifact.py` must match `version` in `pyproject.toml`.
