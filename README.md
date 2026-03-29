# BiasOps - SDK

Policy-as-code enforcement for AI governance.

## Install
```bash
pip install biasops
```

## Usage (one line added to your eval loop)

```python
import biasops

biasops.eval(
    model=model,
    X_test=X_test,
    y_test=y_test,
    y_pred=y_pred,
    policies=['eeoc_title7_hiring_disparate_impact', 'eu_ai_act_high_risk_system'],
    protected_cols=['gender'],
    domain='hiring_and_recruitment_ai',
)
```

## What it checks
- EEOC Title VII 4/5ths rule (29 CFR §1607.4(D))
- EU AI Act Article 10 bias examination
- NYC Local Law 144, Illinois HB3773, Colorado AI Act

## Output
Blocks deployment if any BLOCK rule fails (exit code 1).
Writes a signed audit artifact for compliance documentation.

## Policy Marketplace
[github.com/sksvineeth/biasops-policy-marketplace](https://github.com/sksvineeth/biasops-policy-marketplace)
