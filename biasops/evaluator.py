from __future__ import annotations
import logging
from datetime import date, datetime
from pathlib import Path
import yaml

logger = logging.getLogger("biasops")
POLICIES_DIR = Path(__file__).parent / "policies"
NUMERIC_TYPES = {"numeric", "conditional_numeric"}
OPERATORS = {
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
    ">": lambda v, t: v > t,
    "<": lambda v, t: v < t,
    "==": lambda v, t: v == t,
}

def load_policy(policy_name: str, policy_dir=None) -> dict:
    search_dirs = []
    if policy_dir:
        search_dirs.append(Path(policy_dir))
    search_dirs.append(POLICIES_DIR)
    for d in search_dirs:
        for suffix in [".yaml", ".yml", ""]:
            candidate = d / f"{policy_name}{suffix}"
            if candidate.exists():
                with open(candidate) as f:
                    return yaml.safe_load(f)
        for p in d.rglob("*.yaml"):
            if p.stem.replace("-", "_") == policy_name.replace("-", "_"):
                with open(p) as f:
                    return yaml.safe_load(f)
    raise FileNotFoundError(f"BiasOps: policy '{policy_name}' not found. Searched: {[str(d) for d in search_dirs]}")

def _is_effective(policy: dict) -> bool:
    eff = policy.get("effective_date")
    if not eff:
        return True
    if isinstance(eff, str):
        eff = datetime.fromisoformat(eff).date()
    elif isinstance(eff, datetime):
        eff = eff.date()
    return date.today() >= eff

def _resolve_threshold(rule_or_sub: dict, domain, fallback=None):
    if domain:
        overrides = (rule_or_sub.get("domain_overrides") or
                     (fallback or {}).get("domain_overrides") or {})
        domain_override = overrides.get(domain, {})
        if isinstance(domain_override, dict):
            mid = rule_or_sub.get("metric_id", "")
            if mid in domain_override:
                val = domain_override[mid]
                return None if val is None else float(val)
            if "threshold" in domain_override:
                val = domain_override["threshold"]
                return None if val is None else float(val)
    threshold = rule_or_sub.get("threshold", (fallback or {}).get("threshold"))
    if threshold is None:
        return None
    if isinstance(threshold, dict):
        return None
    return float(threshold)

def _normalise_rules(rule: dict, domain) -> list:
    rtype = rule.get("type", "")
    if rtype not in NUMERIC_TYPES:
        return []
    condition = rule.get("condition", "")
    if condition and "generative" in condition:
        return []
    base = {
        "rule_id":    rule["id"],
        "name":       rule["name"],
        "article":    rule.get("article", ""),
        "operator":   rule.get("operator", ">="),
        "severity":   rule.get("severity", "BLOCK"),
        "stage":      rule.get("stage", ""),
        "citation":   rule.get("threshold_provenance", {}).get("source", ""),
        "confidence": rule.get("threshold_provenance", {}).get(
                           "confidence",
                           rule.get("threshold_provenance", {}).get("default_confidence", "?")),
    }
    pairs = []
    if "metric_id" in rule:
        threshold = _resolve_threshold(rule, domain)
        if threshold is not None:
            pairs.append({**base, "metric_id": rule["metric_id"], "threshold": threshold})
    elif "metric_ids" in rule:
        for m in rule["metric_ids"]:
            threshold = _resolve_threshold(m, domain, fallback=rule)
            if threshold is not None:
                pairs.append({
                    **base,
                    "rule_id":   f"{rule['id']}.{m.get('id', m['metric_id'])}",
                    "metric_id": m["metric_id"],
                    "threshold": threshold,
                    "severity":  m.get("severity", base["severity"]),
                    "operator":  m.get("operator", base["operator"]),
                })
    return pairs

def evaluate(metrics: dict, policies: list, domain=None) -> list:
    results = []
    for policy in policies:
        pid = policy.get("id", "unknown")
        if not _is_effective(policy):
            logger.info(f"BiasOps: policy '{pid}' not yet in effect — skipped")
            continue
        for rule in policy.get("rules", []):
            for r in _normalise_rules(rule, domain):
                value = metrics.get(r["metric_id"])
                if value is None:
                    results.append({**r, "policy_id": pid, "value": None,
                                    "passed": None, "status": "SKIPPED",
                                    "skip_reason": f"metric '{r['metric_id']}' not computed"})
                    continue
                op_fn  = OPERATORS.get(r["operator"], OPERATORS[">="])
                passed = bool(op_fn(value, r["threshold"]))
                results.append({**r, "policy_id": pid, "value": value,
                                "passed": passed,
                                "status": "PASS" if passed else r["severity"]})
    return results
