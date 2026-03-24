from __future__ import annotations
import hashlib, json, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BIASOPS_VERSION = "0.1.0"

def build(metrics, rule_results, policies, protected_cols, domain=None) -> dict:
    violations = [r for r in rule_results if r.get("status") in ("BLOCK","WARN") and r.get("passed") is not None and not bool(r.get("passed"))]
    blocks = [r for r in violations if r.get("status") == "BLOCK"]
    warns  = [r for r in violations if r.get("status") == "WARN"]
    skipped = [r for r in rule_results if r.get("status") == "SKIPPED"]

    artifact = {
        "biasops_version":  BIASOPS_VERSION,
        "run_id":           str(uuid.uuid4())[:8],
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "policies_loaded":  [{"id": p.get("id",""), "name": p.get("name",""),
                               "version": p.get("version",""), "jurisdiction": p.get("jurisdiction","")}
                              for p in policies],
        "protected_cols":   protected_cols,
        "domain":           domain,
        "overall_status":   "BLOCK" if blocks else ("WARN" if warns else "PASS"),
        "block_count":      len(blocks),
        "warn_count":       len(warns),
        "skip_count":       len(skipped),
        "metrics":          {k: (round(v,4) if isinstance(v,float) else v)
                             for k,v in metrics.items()
                             if k not in {"worst_selection_rate_group","best_selection_rate_group",
                                          "worst_accuracy_group","max_group_selection_rate"}},
        "rule_results":     [{k: v for k,v in r.items()
                               if k in {"rule_id","policy_id","name","metric_id","value",
                                        "threshold","operator","status","severity","passed",
                                        "article","citation","confidence","stage","skip_reason"}}
                              for r in rule_results],
        "summary":          {"blocks": [{"rule_id": r.get("rule_id"), "policy_id": r.get("policy_id"),
                                           "metric_id": r.get("metric_id"), "value": r.get("value"),
                                           "threshold": r.get("threshold"), "status": r.get("status"),
                                           "citation": r.get("citation","")[:120]}
                                          for r in blocks],
                              "warns":  [{"rule_id": r.get("rule_id"), "value": r.get("value")}
                                          for r in warns]},
    }
    content = json.dumps(artifact, sort_keys=True, default=str).encode()
    artifact["sha256"] = hashlib.sha256(content).hexdigest()
    return artifact

def write(artifact, output_dir=".") -> str:
    path = Path(output_dir) / f"biasops-run-{artifact['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    return str(path)
