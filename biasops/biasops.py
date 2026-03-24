from __future__ import annotations
import textwrap
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from .adapter   import collect
from .artifact  import build, write
from .evaluator import evaluate, load_policy

class BiasOpsBlockError(Exception):
    def __init__(self, violations, artifact_path, artifact):
        self.violations    = violations
        self.artifact_path = artifact_path
        self.artifact      = artifact
        lines = ["", f"BiasOps — {len(violations)} BLOCK violation(s)", "─"*60]
        for v in violations:
            val = f"{v['value']:.4f}" if v.get("value") is not None else "N/A"
            thr = f"{v['threshold']:.2f}" if v.get("threshold") is not None else "N/A"
            lines.append(f"  {v['rule_id']:<22}  {v.get('metric_id',''):<35}  {val}  (need {v.get('operator','>=')} {thr})")
            cite = v.get("citation","")
            if cite:
                lines.append(f"  {'':22}  {textwrap.shorten(cite, 70)}")
            lines.append("")
        lines += [f"Artifact: {artifact_path}", "─"*60]
        super().__init__("\n".join(lines))

def eval(model, X_test, y_test, y_pred,
         policies="eeoc_title7_hiring_disparate_impact",
         protected_cols=None, domain=None,
         output_dir=".", warn_only=False, policy_dir=None) -> dict:
    if isinstance(policies, str):
        policies = [policies]
    if protected_cols is None:
        raise ValueError("BiasOps: protected_cols is required.")
    if isinstance(protected_cols, str):
        protected_cols = [protected_cols]

    y_test = pd.Series(y_test).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    loaded_policies = [load_policy(name, policy_dir=policy_dir) for name in policies]

    primary_col = protected_cols[0]
    if primary_col not in X_test.columns:
        raise ValueError(f"BiasOps: column '{primary_col}' not in X_test. Available: {list(X_test.columns)}")

    metrics      = collect(y_test, y_pred, X_test[primary_col])
    rule_results = evaluate(metrics, loaded_policies, domain=domain)
    artifact     = build(metrics, rule_results, loaded_policies, protected_cols, domain)
    artifact_path = write(artifact, output_dir)

    _print_summary(rule_results, artifact, artifact_path)

    blocks = [r for r in rule_results if r.get("status") == "BLOCK" and r.get("passed") is False]
    if blocks and not warn_only:
        raise BiasOpsBlockError(blocks, artifact_path, artifact)
    return artifact

def _print_summary(rule_results, artifact, artifact_path):
    status = artifact["overall_status"]
    colors = {"BLOCK": "\033[91m", "WARN": "\033[93m", "PASS": "\033[92m"}
    reset  = "\033[0m"
    icons  = {"PASS": "✓", "BLOCK": "✗", "WARN": "!", "SKIPPED": "–"}
    print(f"\nBiasOps v{artifact['biasops_version']}  run:{artifact['run_id']}")
    print(f"Policies: {', '.join(p['id'] for p in artifact['policies_loaded'])}")
    print("─"*72)
    for r in rule_results:
        s = r.get("status","?")
        c = colors.get(s, "")
        val = f"{r['value']:.4f}" if r.get("value") is not None else "—"
        thr = f"(need {r.get('operator','>=')} {r['threshold']:.2f})" if r.get("threshold") is not None else ""
        print(f"  {c}{icons.get(s,'?')} {s:<8}{reset}  {r.get('rule_id',''):<24}  {r.get('metric_id',''):<35}  {val:<8}  {thr}")
    print("─"*72)
    sc = colors.get(status,"")
    print(f"Status: {sc}{status}{reset}  ·  {artifact['block_count']} block  {artifact['warn_count']} warn  {artifact['skip_count']} skipped")
    print(f"Artifact: {artifact_path}\n")
