"""Recommend local or commercial inference from task complexity and workload evidence.

Offline planner: no inference, network requests, API keys, or prompt uploads.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def default_policy_path(home=None):
    """Use the existing editable home policy, with a bundled install fallback."""
    if home is None:
        from .runtime import project_home
        home = project_home()
    candidate = Path(home) / 'benchmarks/decision_policy.json'
    return candidate if candidate.is_file() else ROOT / 'data/decision_policy.json'
LEVELS = {"low": 0, "medium": 1, "high": 2}
TASK_COMPLEXITY = {
    "extract": "low", "classify": "low", "format": "low",
    "summarize": "medium", "translate": "medium", "general": "medium",
    "code": "high", "reason": "high",
}


def numeric(value, name, minimum=0, maximum=None, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum
            or (maximum is not None and value > maximum)
            or (integer and not isinstance(value, int))):
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def wilson_lower(correct, cases):
    """Lower end of a descriptive two-sided 95% Wilson interval."""
    z = 1.959963984540054
    p = correct / cases
    return (p + z*z/(2*cases) - z*math.sqrt(p*(1-p)/cases + z*z/(4*cases*cases))) / (1 + z*z/cases)


def validate_policy(policy):
    if policy["schema_version"] != 1:
        raise ValueError("Unsupported policy schema_version")
    if policy["local_max_complexity"] not in LEVELS:
        raise ValueError("Invalid local_max_complexity")
    numeric(policy["minimum_success_rate"], "minimum_success_rate", 0, 1)
    numeric(policy["minimum_cases"], "minimum_cases", 1, integer=True)
    for name in ("local", "commercial"):
        model = policy["models"][name]
        for field in ("model", "configuration"):
            if not isinstance(model[field], str) or not model[field].strip():
                raise ValueError(f"{name}.{field} must be nonempty")
        numeric(model["context_tokens"], f"{name}.context_tokens", 1, integer=True)
        for field in ("cost_per_request_usd", "monthly_fixed_cost_usd"):
            if model[field] is not None:
                numeric(model[field], f"{name}.{field}")
        if not isinstance(model["evaluations"], list):
            raise ValueError(f"{name}.evaluations must be a list")
        identities = set()
        for evidence in model["evaluations"]:
            if evidence["complexity"] not in LEVELS:
                raise ValueError("Invalid evaluation complexity")
            for field in ("workload", "model", "configuration", "dataset_id", "source"):
                if not isinstance(evidence[field], str) or not evidence[field].strip():
                    raise ValueError(f"Evaluation {field} must be nonempty")
            if not isinstance(evidence["representative"], bool):
                raise ValueError("Evaluation representative must be boolean")
            numeric(evidence["cases"], "evaluation cases", 1, integer=True)
            numeric(evidence["correct"], "evaluation correct", 0, evidence["cases"], integer=True)
            if evidence["p95_latency_s"] is not None:
                numeric(evidence["p95_latency_s"], "evaluation p95_latency_s", 0)
            identity = (evidence["workload"], evidence["complexity"], evidence["model"], evidence["configuration"])
            if identity in identities:
                raise ValueError(f"Duplicate evaluation: {identity}")
            identities.add(identity)


def decide(policy, *, task="general", complexity="auto", workload="default",
           allow_cloud=False, offline=False, validation_failed=False,
           multistep=False, high_stakes=False, input_tokens=0, output_tokens=128,
           monthly_requests=1000, max_monthly_cost_usd=None, max_latency_s=None):
    validate_policy(policy)
    if task not in TASK_COMPLEXITY or complexity not in ("auto", *LEVELS):
        raise ValueError("Unknown task or complexity")
    for value, name, minimum in ((input_tokens, "input_tokens", 0),
                                 (output_tokens, "output_tokens", 1),
                                 (monthly_requests, "monthly_requests", 1)):
        numeric(value, name, minimum, integer=True)
    for value, name in ((max_monthly_cost_usd, "max_monthly_cost_usd"), (max_latency_s, "max_latency_s")):
        if value is not None:
            numeric(value, name)
    level = TASK_COMPLEXITY[task] if complexity == "auto" else complexity
    reasons = [f"Task {task}: {level} complexity ({'task default' if complexity == 'auto' else 'explicit input'})."]
    if multistep or high_stakes:
        level = "high"
        reasons.append("Multiple reasoning steps or high consequences raise complexity to high.")
    cloud_enabled = allow_cloud and not offline
    candidates = {}
    for name in ("local", "commercial"):
        model = policy["models"][name]
        excluded, missing, notes = [], [], []
        evidence = next((e for e in model["evaluations"]
                         if e["workload"] == workload and e["complexity"] == level
                         and e["model"] == model["model"]
                         and e["configuration"] == model["configuration"]), None)
        quality = "unknown"
        lower = None
        if evidence and evidence["representative"] and evidence["cases"] >= policy["minimum_cases"]:
            lower = wilson_lower(evidence["correct"], evidence["cases"])
            quality = "pass" if lower >= policy["minimum_success_rate"] else "fail"
            if quality == "fail":
                excluded.append("Quality evidence does not clear the configured success threshold.")
        else:
            notes.append("Need enough representative, graded cases for this workload, complexity, model and configuration.")
        if name == "commercial" and not cloud_enabled:
            excluded.append("Cloud use is disabled or offline operation is required.")
        if name == "local":
            if validation_failed:
                excluded.append("The local answer failed task validation; escalate instead of retrying blindly.")
            if LEVELS[level] > LEVELS[policy["local_max_complexity"]] and quality != "pass":
                excluded.append("Above the local complexity ceiling without passing workload evidence.")
        if input_tokens + output_tokens > model["context_tokens"]:
            excluded.append("Input plus output exceeds the configured context capacity.")
        variable, fixed = model["cost_per_request_usd"], model["monthly_fixed_cost_usd"]
        monthly = fixed + monthly_requests * variable if variable is not None and fixed is not None else None
        if max_monthly_cost_usd is not None:
            if monthly is None:
                missing.append("Monthly budget cannot be checked until variable and fixed costs are supplied.")
            elif monthly > max_monthly_cost_usd:
                excluded.append("Estimated monthly cost exceeds the budget.")
        latency = evidence["p95_latency_s"] if evidence else None
        if max_latency_s is not None:
            if latency is None or quality == "unknown":
                missing.append("Need representative p95 end-to-end latency for this workload.")
            elif latency > max_latency_s:
                excluded.append("Measured p95 end-to-end latency exceeds the limit.")
        candidates[name] = {
            "model": model["model"], "configuration": model["configuration"],
            "quality": quality, "success_rate_lower_95": lower,
            "dataset_id": evidence["dataset_id"] if evidence else None,
            "evidence_source": evidence["source"] if evidence else None,
            "estimated_monthly_cost_usd": monthly, "p95_latency_s": latency,
            "excluded": excluded, "missing": missing, "notes": notes,
        }
    eligible = [n for n, c in candidates.items() if not c["excluded"] and not c["missing"]]
    supported = [n for n in eligible if candidates[n]["quality"] == "pass"]
    selected = None
    if supported:
        selected = supported[0]
        reasons.append("Prefer a candidate with passing workload quality evidence.")
        if len(supported) == 2:
            a, b = (candidates[n] for n in supported)
            if a["dataset_id"] != b["dataset_id"]:
                return {"status": "benchmark_required", "recommendation": None, "model": None,
                        "complexity": level, "workload": workload, "candidates": candidates,
                        "reasons": reasons + ["Re-evaluate both models on the same dataset and grading rubric before comparing."],
                        "executes_inference": False}
            if all(c["estimated_monthly_cost_usd"] is not None for c in (a, b)):
                selected = min(supported, key=lambda n: candidates[n]["estimated_monthly_cost_usd"])
                reasons.append("Both clear quality requirements; choose the lower estimated monthly cost.")
            else:
                reasons.append("Both clear quality requirements; prefer local until comparable costs are supplied.")
    elif eligible:
        selected = eligible[0]
        reasons.append("Use the complexity policy provisionally; task quality is not established.")
    status = ("supported" if selected in supported else "provisional") if selected else (
        "benchmark_required" if any(not c["excluded"] and c["missing"] for c in candidates.values()) else "blocked")
    if not selected:
        reasons.append("No candidate currently meets the policy and available constraints; inspect candidate reasons.")
    if high_stakes:
        reasons.append("Human review is required for high-consequence answers regardless of model choice.")
    if input_tokens == 0:
        reasons.append("Input size is unspecified; context fit must be checked before execution.")
        if status == "supported":
            status = "provisional"
    return {"status": status, "recommendation": selected,
            "model": candidates[selected]["model"] if selected else None,
            "complexity": level, "workload": workload, "reasons": reasons,
            "candidates": candidates, "executes_inference": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=default_policy_path())
    parser.add_argument("--task", choices=TASK_COMPLEXITY, default="general")
    parser.add_argument("--complexity", choices=("auto", *LEVELS), default="auto")
    parser.add_argument("--workload", default="default")
    parser.add_argument("--allow-cloud", action="store_true", help="allow commercial recommendations")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--validation-failed", action="store_true", help="local answer failed a task-specific validator")
    parser.add_argument("--multistep", action="store_true")
    parser.add_argument("--high-stakes", action="store_true")
    parser.add_argument("--input-tokens", type=int, default=0, help="estimated input including instructions/history; 0 means unspecified")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--monthly-requests", type=int, default=1000)
    parser.add_argument("--max-monthly-cost-usd", type=float)
    parser.add_argument("--max-latency-s", type=float)
    parser.add_argument("--output", type=Path, help="write a new JSON audit file; existing files are never overwritten")
    args = parser.parse_args()
    try:
        raw = args.policy.read_bytes()
        policy = json.loads(raw.decode("utf-8-sig"))
        request = {k: v for k, v in vars(args).items() if k not in ("policy", "output")}
        result = decide(policy, **request)
        result["request"] = request
        result["policy_sha256"] = hashlib.sha256(raw).hexdigest()
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as destination:
                destination.write(serialized)
        print(serialized, end="")
        return 0 if result["recommendation"] else 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
