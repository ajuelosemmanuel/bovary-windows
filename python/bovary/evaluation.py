"""Paired retained-case evaluation. Synthetic diagnostics never qualify Auto."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from .contract import Request
from .validators import VERSION, validate, validate_specification
from .contract import ExecutionError


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def check_dataset(dataset):
    if not isinstance(dataset, dict):
        raise ValueError("Dataset must be an object")
    if dataset.get("schema_version") != 1 or not dataset.get("dataset_id") or not dataset.get("rubric_id"):
        raise ValueError("Dataset needs versioned dataset/rubric identities")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must retain at least one case")
    ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Dataset cases must be objects")
        if not isinstance(case.get("id"), str) or case["id"] in ids or not case["id"]:
            raise ValueError("Case IDs must be unique")
        ids.add(case["id"])
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip() or not isinstance(case.get("validator"), dict):
            raise ValueError("Every case needs a prompt and independent validator")
        if case.get("task") not in ("extract", "classify", "format", "reason", "code", "general", "summarize", "translate"):
            raise ValueError("Every case needs a supported task")
        try:
            validate_specification(case["validator"])
        except ExecutionError as error:
            raise ValueError("Dataset contains an invalid validator specification") from error
    if type(dataset.get("synthetic")) is not bool or type(dataset.get("representative")) is not bool:
        raise ValueError("Dataset must explicitly label synthetic/representative status")


def _percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def _nonnegative(value, label, *, integer=False):
    if value is None:
        return
    try:
        valid = type(value) is int and value >= 0 if integer else type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"Invalid retained {label}")


def _check_usage(usage):
    if not isinstance(usage, dict) or any(not isinstance(key, str) for key in usage):
        raise ValueError("Retained usage must be an object")
    for name, value in usage.items():
        if name.endswith("_tokens"):
            _nonnegative(value, name, integer=True)
        elif name.endswith("_usd"):
            _nonnegative(value, name)
    input_count, output_count, total = (usage.get(k) for k in ("input_tokens", "output_tokens", "total_tokens"))
    if input_count is not None and output_count is not None and total is not None and total != input_count + output_count:
        raise ValueError("Retained total tokens disagree with input/output usage")
    for details_name, parent_name in (("input_tokens_details", "input_tokens"), ("output_tokens_details", "output_tokens")):
        details = usage.get(details_name)
        if details is None:
            continue
        if not isinstance(details, dict) or any(not isinstance(key, str) for key in details):
            raise ValueError("Retained token details must be objects")
        for name, value in details.items():
            if name.endswith("_tokens"):
                _nonnegative(value, name, integer=True)
                if value is not None and usage.get(parent_name) is not None and value > usage[parent_name]:
                    raise ValueError("Retained token detail exceeds its parent category")


def _check_result(result, *, attempt=False):
    if not isinstance(result, dict) or result.get("status") not in ("complete", "needs_review", "failed", "cancelled", "truncated", "suggested"):
        raise ValueError("Invalid retained result status")
    if not isinstance(result.get("text", ""), str):
        raise ValueError("Retained result text must be text")
    _nonnegative(result.get("cost_usd"), "cost")
    if "cost_status" in result and not isinstance(result["cost_status"], str):
        raise ValueError("Retained cost accounting status must be text")
    _check_usage(result.get("usage", {}))
    timing = result.get("timing", {})
    if not isinstance(timing, dict):
        raise ValueError("Retained timing must be an object")
    for key in ("end_to_end_s", "ttft_s", "startup_s", "model_load_s"):
        _nonnegative(timing.get(key), key)
    if timing.get("ttft_s") is not None and timing.get("end_to_end_s") is not None and timing["ttft_s"] > timing["end_to_end_s"] + 1e-6:
        raise ValueError("Retained TTFT exceeds end-to-end duration")
    for key in ("warm", "cold"):
        if key in timing and type(timing[key]) is not bool:
            raise ValueError("Retained cold/warm flags must be boolean")
    validation = result.get("validation", {})
    if not isinstance(validation, dict):
        raise ValueError("Retained validation must be an object")
    _nonnegative(validation.get("cost_usd"), "validation cost")
    if attempt and result.get("attempts"):
        raise ValueError("Backend attempt snapshots cannot contain nested attempts")


def _known_attempt_cost(attempt):
    status = attempt.get("cost_status", "unknown")
    if not isinstance(status, str) or status.startswith(("unknown", "unmeasured", "reserved", "not_a_model")):
        return None
    return attempt.get("cost_usd")


def summarize(dataset, records):
    """Regrade retained text; never trust a result's own validation or summary."""
    check_dataset(dataset)
    if not isinstance(records, list) or not records:
        raise ValueError("Evidence must retain nonempty evaluated scenarios and every case")
    cases = {x["id"]: x for x in dataset["cases"]}
    grouped = {}
    keys = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("result"), dict):
            raise ValueError("Malformed retained result")
        if record.get("scenario") not in ("local", "commercial", "routing", "escalation"):
            raise ValueError("Unknown retained evaluation scenario")
        elapsed = record.get("elapsed_s")
        if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid retained end-to-end latency")
        key = (record["scenario"], record["case_id"])
        if key in keys or record["case_id"] not in cases:
            raise ValueError("Duplicate or unknown evaluated case")
        keys.add(key)
        grouped.setdefault(record["scenario"], []).append(record)
    output = {}
    for scenario, rows in grouped.items():
        if {r["case_id"] for r in rows} != set(cases):
            raise ValueError("Every scenario must retain every case including failures")
        correct = false_accept = fallback = errors = 0
        latency, ttft, costs = [], [], []
        usage = []
        cold = []
        warm_observations = 0
        cold_observations = 0
        attempt_count = 0
        for row in rows:
            result = row["result"]
            _check_result(result)
            if result.get("timing", {}).get("end_to_end_s") is not None and result["timing"]["end_to_end_s"] > row["elapsed_s"] + 1e-6:
                raise ValueError("Retained result duration exceeds measured case duration")
            accepted = result.get("status") == "complete"
            judged = validate(result.get("text", ""), cases[row["case_id"]]["validator"]) if accepted else {"passed": False}
            correct += bool(accepted and judged["passed"])
            false_accept += bool(accepted and not judged["passed"])
            errors += result.get("status") not in ("complete", "needs_review")
            attempts = result.get("attempts", [])
            if not isinstance(attempts, list) or any(not isinstance(attempt, dict) for attempt in attempts):
                raise ValueError("Retained backend attempts must be explicit snapshots")
            if accepted and not attempts:
                raise ValueError("Completed execution requires an actual backend attempt snapshot")
            attempt_count += len(attempts)
            fallback += len(attempts) > 1
            latency.append(row["elapsed_s"])
            timing = result.get("timing", {})
            if timing.get("ttft_s") is not None:
                ttft.append(timing["ttft_s"])
            for attempt in attempts:
                _check_result(attempt, attempt=True)
                attempt_timing = attempt.get("timing", {})
                if attempt_timing.get("warm") is True:
                    warm_observations += 1
                if attempt_timing.get("warm") is False or attempt_timing.get("cold") is True:
                    cold_observations += 1
                    cold.append(attempt_timing.get("startup_s", attempt_timing.get("model_load_s")))
                usage.append(attempt.get("usage", {}))
            attempt_durations = [attempt.get("timing", {}).get("end_to_end_s") for attempt in attempts]
            if attempt_durations and all(duration is not None for duration in attempt_durations) and sum(attempt_durations) > row["elapsed_s"] + 1e-6:
                raise ValueError("Retained attempt durations exceed measured case duration")
            # Derive total task cost from each attempt and validation rather
            # than trusting an outer aggregate that can conceal unknown costs.
            attempt_costs = [_known_attempt_cost(attempt) for attempt in attempts]
            validation_costs = [attempt.get("validation", {}).get("cost_usd")
                                for attempt in attempts if attempt.get("validation", {}).get("status") in ("pass", "fail", "interrupted")]
            components = attempt_costs + validation_costs
            task_cost = sum(components) if components and all(value is not None for value in components) else None
            _nonnegative(task_cost, "total task cost")
            costs.append(task_cost)
        total_cost = sum(costs) if all(type(c) in (float, int) and math.isfinite(c) and c >= 0 for c in costs) else None
        _nonnegative(total_cost, "total scenario cost")
        token_totals = {}
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            token_totals[name] = sum(x[name] for x in usage) if all(type(x.get(name)) is int and x[name] >= 0 for x in usage) else None
        output[scenario] = {"cases": len(rows), "correct": correct, "success_rate": correct / len(rows),
                            "false_acceptances": false_accept, "errors": errors, "attempts": attempt_count,
                            "fallback_rate": fallback / len(rows), "median_latency_s": statistics.median(latency),
                            "p95_latency_s": _percentile(latency, .95), "p95_observations": len(latency),
                            "p95_caution": "Fewer than 100 observations; tail estimate is unstable" if len(latency) < 100 else "Descriptive held-out observations; no population guarantee",
                            "median_ttft_s": statistics.median(ttft) if ttft else None,
                            "ttft_observations": len(ttft), "cold_model_load_s": [x for x in cold if x is not None],
                            "cold_observations": cold_observations, "warm_observations": warm_observations,
                            "usage": token_totals, "billed_usage_details": usage,
                            "total_cost_usd": total_cost, "cost_per_successful_task_usd": total_cost / correct if total_cost is not None and correct else None}
    return output


def deterministic_baseline(case):
    """Author-declared ordinary-code baselines for bounded transformations."""
    baseline = case.get("baseline", {})
    if baseline.get("kind") == "extract_key_values":
        fields = {}
        for line in baseline["input"].splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip()] = value.strip()
        return json.dumps(fields, sort_keys=True, separators=(",", ":"))
    if baseline.get("kind") == "sort_lines":
        return "\n".join(sorted(baseline["input"].splitlines()))
    return None


def run_evaluation(workbench, dataset, request_template, output_dir, *,
                   scenarios=("local", "commercial", "routing", "escalation"), source_kind="measured", identities=None):
    """Runs serially. Caller must explicitly authorize all requested routes.

    A new directory is required. Raw prompts/answers are private by default and
    must be reviewed before publication. No representative claim is inferred.
    """
    check_dataset(dataset)
    if not scenarios or len(set(scenarios)) != len(scenarios) or any(x not in ("local", "commercial", "routing", "escalation") for x in scenarios):
        raise ValueError("Invalid paired scenarios")
    if source_kind not in ("measured", "fake"):
        raise ValueError("source_kind must be measured or fake")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        with (output_dir / "attempts.jsonl").open("x", encoding="utf-8") as log:
            # Same case, then scenarios: reduces time-order drift. Local worker
            # remains warm; load timing stays separate in its native metrics.
            for case in dataset["cases"]:
                for scenario in scenarios:
                    mode = scenario if scenario in ("local", "commercial") else "suggest" if scenario == "routing" else "local"
                    request = replace(request_template, prompt=case["prompt"], request_id=f"eval-{case['id']}-{scenario}",
                                      task=case["task"], workload=case.get("workload", dataset["dataset_id"]),
                                      mode=mode, validator=case["validator"], allow_escalation=scenario == "escalation")
                    start = time.monotonic()
                    if scenario == "routing":
                        suggestion = workbench.execute(request)
                        if suggestion.destination in ("local", "commercial"):
                            request = replace(request, mode=suggestion.destination)
                            result = workbench.execute(request)
                            result.reasons.insert(0, "Offline provisional routing evaluation; this does not enable Auto")
                        else:
                            result = suggestion
                    else:
                        result = workbench.execute(request)
                    record = {"scenario": scenario, "case_id": case["id"], "elapsed_s": time.monotonic() - start,
                              "prompt_sha256": digest(case["prompt"]), "validator_sha256": digest(case["validator"]),
                              "result": result.to_dict()}
                    records.append(record)
                    log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
            baselines = []
            for case in dataset["cases"]:
                start = time.monotonic()
                text = deterministic_baseline(case)
                if text is not None:
                    baselines.append({"case_id": case["id"], "text": text, "elapsed_s": time.monotonic() - start,
                                      "validation": validate(text, case["validator"])})
        artifact = {"schema_version": 1, "source_kind": source_kind, "dataset": dataset,
                    "dataset_sha256": digest(dataset), "validator_version": VERSION,
                    "policy_sha256": digest(workbench.policy), "identities": identities or {},
                    "records": records, "summary": summarize(dataset, records), "deterministic_baselines": baselines,
                    "auto_qualified": False}
        (output_dir / "evaluation.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return artifact
    finally:
        workbench.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", action="append", choices=("local", "commercial", "routing", "escalation"), default=None)
    parser.add_argument("--provider-config", type=Path)
    parser.add_argument("--allow-cloud", action="store_true")
    parser.add_argument("--data-class", default="")
    parser.add_argument("--max-cost-usd", type=float, default=0)
    parser.add_argument("--deadline-s", type=float, default=300)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    from .runtime import create_workbench
    workbench = create_workbench(provider_config=args.provider_config)
    request = Request("evaluation template", allowed_destinations=("local", "commercial") if args.allow_cloud else ("local",),
                      offline=not args.allow_cloud, private=not args.allow_cloud,
                      output_tokens=args.tokens, max_cost_usd=args.max_cost_usd,
                      deadline=time.monotonic() + args.deadline_s, identity={"data_class": args.data_class})
    try:
        result = run_evaluation(workbench, json.loads(args.dataset.read_text(encoding="utf-8-sig")), request, args.output,
                                scenarios=tuple(args.scenario or ("local",)))
        print(json.dumps(result["summary"], indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
