"""Content-bound evidence import and fail-closed Auto qualification.

A SHA256 hash proves identity, not truth. The configured trusted digest must be
an independently reviewed retained artifact. A request cannot supply trust.
"""
import hashlib
import json
from pathlib import Path
import argparse

from .contract import ExecutionError
from .evaluation import digest, summarize
from .policy import wilson_lower
from .validators import VERSION

REQUIRED_IDENTITIES = ("model_sha256", "quantization", "runtime_profile", "prompt_version", "validator_version", "policy_sha256", "dataset_sha256")


def import_evidence(path):
    raw = Path(path).read_bytes()
    artifact = json.loads(raw)
    if not isinstance(artifact, dict) or artifact.get("source_kind") not in ("measured", "fake"):
        raise ValueError("Evidence must identify a measured or fake retained run")
    if artifact.get("schema_version") != 1 or artifact.get("dataset_sha256") != digest(artifact["dataset"]):
        raise ValueError("Evidence dataset identity is missing or inconsistent")
    if artifact.get("validator_version") != VERSION:
        raise ValueError("Evidence validator version is stale")
    if artifact.get("summary") != summarize(artifact["dataset"], artifact["records"]):
        raise ValueError("Evidence summary differs from retained graded records")
    cases = {x["id"]: x for x in artifact["dataset"]["cases"]}
    for row in artifact["records"]:
        case = cases[row["case_id"]]
        if row.get("prompt_sha256") != digest(case["prompt"]) or row.get("validator_sha256") != digest(case["validator"]):
            raise ValueError("Evidence prompt or rubric hash differs")
    return artifact, hashlib.sha256(raw).hexdigest()


def verify_qualification(path, trusted_sha256, request, destination, policy):
    try:
        if path is None or not trusted_sha256:
            raise ValueError("No independently reviewed retained evaluation is configured")
        artifact, sha = import_evidence(path)
        if sha not in trusted_sha256:
            raise ValueError("Evaluation content hash has not been trusted by a reviewer")
        dataset = artifact["dataset"]
        if artifact.get("source_kind") != "measured" or dataset["synthetic"] or not dataset["representative"]:
            raise ValueError("Fixture, synthetic or unrepresentative data cannot qualify Auto")
        if dataset.get("workload") != request.workload or not dataset.get("held_out"):
            raise ValueError("Evidence does not describe this named held-out workload")
        if artifact.get("policy_sha256") != digest(policy):
            raise ValueError("Policy changed after evaluation")
        identity = artifact.get("identities", {}).get(destination, {})
        for name in REQUIRED_IDENTITIES:
            if not identity.get(name) or identity.get(name) != request.identity.get(name):
                raise ValueError(f"Missing or stale qualification identity: {name}")
        if identity["policy_sha256"] != digest(policy) or identity["dataset_sha256"] != artifact["dataset_sha256"] or identity["validator_version"] != VERSION:
            raise ValueError("Qualification is not bound to policy, dataset and validator")
        if identity["runtime_profile"] != request.local_profile and destination == "local":
            raise ValueError("Local runtime profile changed")
        model = policy["models"][destination]
        if identity.get("model") != model["model"] or identity.get("configuration") != model["configuration"]:
            raise ValueError("Model/configuration differs from policy")
        summary = artifact["summary"].get(destination)
        if not summary or summary["cases"] < policy["minimum_cases"] or summary["false_acceptances"]:
            raise ValueError("Insufficient cases or false acceptances")
        if wilson_lower(summary["correct"], summary["cases"]) < policy["minimum_success_rate"]:
            raise ValueError("Held-out quality is below the configured floor")
        # Require paired ordinary-code or commercial/local baseline measurements
        # and a fixed objective recorded before evaluation.
        baseline_name = "commercial" if destination == "local" else "local"
        baseline = artifact["summary"].get(baseline_name)
        objective = dataset.get("acceptance", {}).get("objective")
        if not baseline or baseline["cases"] != summary["cases"]:
            raise ValueError("Missing paired meaningful baseline")
        if objective == "latency":
            if summary["p95_latency_s"] >= baseline["p95_latency_s"]:
                raise ValueError("No measured p95 latency benefit against paired baseline")
        elif objective == "cost":
            ours, theirs = summary["cost_per_successful_task_usd"], baseline["cost_per_successful_task_usd"]
            if ours is None or theirs is None or ours >= theirs:
                raise ValueError("No known measured cost benefit against paired baseline")
        else:
            raise ValueError("No predeclared supported operating objective")
        if summary["success_rate"] < baseline["success_rate"]:
            raise ValueError("Candidate quality is below the paired baseline")
        if request.allow_escalation:
            raise ValueError("Automatic escalation requires separately evaluated composite policy; unavailable in this alpha")
        return artifact
    except (ValueError, OSError, KeyError, TypeError, ZeroDivisionError) as error:
        raise ExecutionError("unqualified", str(error)) from error


def main():
    parser = argparse.ArgumentParser(description="Verify retained evaluation integrity and recompute grading; this never enables Auto")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, help="Write a new metadata-only integrity report")
    args = parser.parse_args()
    try:
        artifact, sha = import_evidence(args.artifact)
        report = {"schema_version": 1, "artifact_sha256": sha, "source_kind": artifact["source_kind"],
                  "dataset_id": artifact["dataset"]["dataset_id"], "dataset_sha256": artifact["dataset_sha256"],
                  "synthetic": artifact["dataset"]["synthetic"], "representative_claim": artifact["dataset"]["representative"],
                  "integrity_checked": True, "measurement_truth_reviewed": False, "auto_enabled": False,
                  "policy_sha256": artifact["policy_sha256"], "summary": artifact["summary"]}
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as destination:
                destination.write(serialized)
        print(serialized, end="")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
