#!/usr/bin/env python3
"""Aggregate explicitly selected benchmark directories and render static figures.

No model workload is executed. Unknown metrics remain null/blank/unknown.
Project-local matplotlib can be installed with:
  python -m pip install --target work/deps/python matplotlib
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work/matplotlib-cache"))
sys.path.insert(0, str(ROOT/"work"/"deps"/"python"))
GIB = 1024**3


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def nested(source, *keys):
    for key in keys:
        if not isinstance(source, dict): return None
        source = source.get(key)
    return source if number(source) else None


def extract_metrics(run):
    engine = run.get("engine_metrics", {})
    memory = run.get("memory_summary", {})
    decode = run.get("decode_memory_summary", {})
    before = run.get("system_before_launch", {})
    after = run.get("system_after_exit", {})
    peak_ws = nested(memory, "process_peak_working_set_bytes", "max")
    if peak_ws is None: peak_ws = nested(memory, "process_working_set_bytes", "max")
    metrics = {
        "actual_generated_tokens": run.get("actual_generated_tokens"),
        "model_load_s": engine.get("native_model_load_s"),
        "prompt_tokens_per_s": engine.get("native_prompt_tokens_per_s"),
        "ttft_internal_s": engine.get("native_ttft_process_internal_s"),
        "decode_tokens_per_s": engine.get("native_decode_tokens_per_second"),
        "end_to_end_s": run.get("end_to_end_seconds"),
        "intertoken_p95_s": engine.get("native_intertoken_p95_seconds"),
        "intertoken_max_s": engine.get("native_intertoken_max_seconds"),
        "peak_ws_bytes": peak_ws,
        "decode_ws_bytes": nested(decode, "process_working_set_bytes", "median"),
        "peak_private_commit_bytes": nested(memory, "process_private_commit_bytes", "max"),
        "decode_private_commit_bytes": nested(decode, "process_private_commit_bytes", "median"),
        "peak_virtual_address_bytes": nested(memory, "process_virtual_address_bytes", "max"),
        "decode_virtual_address_bytes": nested(decode, "process_virtual_address_bytes", "median"),
        "peak_gpu_dedicated_bytes": nested(memory, "gpu_process_dedicated_bytes", "max"),
        "decode_gpu_dedicated_bytes": nested(decode, "gpu_process_dedicated_bytes", "median"),
        "peak_gpu_shared_bytes": nested(memory, "gpu_process_shared_bytes", "max"),
        "decode_gpu_shared_bytes": nested(decode, "gpu_process_shared_bytes", "median"),
        "peak_adapter_dedicated_bytes": nested(memory, "gpu_adapter_dedicated_bytes", "max"),
        "peak_adapter_shared_bytes": nested(memory, "gpu_adapter_shared_bytes", "max"),
        "system_available_before_bytes": before.get("system_available_physical_bytes"),
        "system_cache_before_bytes": before.get("system_cache_including_standby_bytes"),
        "system_available_after_bytes": after.get("system_available_physical_bytes"),
        "system_cache_after_bytes": after.get("system_cache_including_standby_bytes"),
        "system_cache_peak_bytes": nested(memory, "system_cache_including_standby_bytes", "max"),
        "system_commit_peak_bytes": nested(memory, "system_commit_bytes", "max"),
        "page_reads_per_s_peak": nested(memory, "system_page_reads_per_second", "max"),
        "stream_host_buffer_bytes": engine.get("stream_host_buffer_bytes"),
        "stream_explicit_pinned_bytes": engine.get("stream_explicit_pinned_bytes"),
        "load_host_buffer_bytes": engine.get("load_host_buffer_bytes"),
        "load_pinned_bytes": engine.get("load_pinned_bytes"),
        "load_buffer_slots": engine.get("load_buffer_slots"),
        "load_slot_requested_bytes": engine.get("load_slot_requested_bytes"),
        "load_direct_io": engine.get("load_direct_io"),
        "load_async_uploads": engine.get("load_async_uploads"),
        "load_pool_record_count": engine.get("load_pool_record_count"),
        "stream_read_api_bytes_per_token_including_prompt": engine.get("stream_read_api_bytes_per_generated_token_including_prompt"),
        "stream_upload_payload_bytes_per_token_including_prompt": engine.get("stream_upload_payload_bytes_per_generated_token_including_prompt"),
        "stream_mean_read_call_s": engine.get("stream_mean_read_call_seconds"),
        "expert_cache_hit_rate_percent": engine.get("expert_cache_hit_rate_percent"),
        "decode_io_bytes_per_generated_token_after_first": engine.get("decode_io_bytes_per_generated_token_after_first"),
        "decode_upload_payload_bytes_per_generated_token_after_first": engine.get("decode_requested_bytes_per_generated_token_after_first"),
        "decode_io_bytes_per_evaluation": engine.get("decode_io_bytes_per_evaluation"),
        "decode_upload_payload_bytes_per_evaluation": engine.get("decode_requested_bytes_per_evaluation"),
        "decode_evaluation_count": engine.get("stream_trace_decode_evaluation_count"),
        "decode_emitted_tokens_after_first": engine.get("stream_trace_emitted_tokens_after_first"),
        "decode_mean_read_call_s": engine.get("decode_mean_read_call_seconds"),
        "decode_read_fraction_of_native_decode_time": engine.get("decode_read_fraction_of_native_decode_time"),
        "decode_upload_wait_fraction_of_native_decode_time": engine.get("decode_upload_wait_fraction_of_native_decode_time"),
        "decode_read_fraction_of_read_plus_upload_wait": engine.get("decode_read_fraction_of_read_plus_upload_wait"),
        "decode_upload_wait_fraction_of_read_plus_upload_wait": engine.get("decode_upload_wait_fraction_of_read_plus_upload_wait"),
        "decode_read_api_bytes_per_read_second": engine.get("decode_read_api_bytes_per_read_second"),
        "decode_upload_payload_bytes_per_upload_wait_second": engine.get("decode_upload_payload_bytes_per_upload_wait_second"),
        "decode_expert_cache_hit_rate_percent": engine.get("decode_expert_cache_hit_rate_percent"),
        "logical_process_read_bytes_per_token_including_load": run.get("process_lifetime_read_transfer_bytes_per_generated_token_including_load"),
        "disk_free_before_bytes": nested(run, "disk_monitor", "free_before_launch_bytes"),
        "disk_free_after_bytes": nested(run, "disk_monitor", "free_after_exit_bytes"),
        "sampled_peak_net_volume_space_increase_bytes": nested(run, "disk_monitor", "sampled_peak_net_volume_space_increase_bytes"),
    }
    for name in ("system_used_physical_bytes", "system_cache_including_standby_bytes", "system_commit_bytes"):
        a, b = before.get(name), after.get(name)
        metrics[name+"_after_minus_before"] = b-a if number(a) and number(b) else None
    return {key: value if number(value) else None for key, value in metrics.items()}


def summarize(values):
    valid = [x for x in values if number(x)]
    return {"median": statistics.median(valid), "min": min(valid), "max": max(valid),
            "spread": max(valid)-min(valid), "count": len(valid)} if valid else None


def load_groups(inputs, model_bytes):
    groups, excluded, seen = {}, [], set()
    for item in inputs:
        path = Path(item).resolve()
        if path.is_dir(): path = path/"results.json"
        if not path.is_file(): raise FileNotFoundError(path)
        matrix_path = path.parent/"configuration.json"
        matrix = json.loads(matrix_path.read_text(encoding="utf-8-sig")) if matrix_path.exists() else {}
        configurations = {x["id"]: x for x in matrix.get("configurations", [])}
        runs = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(runs, list): raise ValueError(f"Expected run list: {path}")
        for run in runs:
            identity = (str(path), run.get("run_id"))
            if identity in seen: continue
            seen.add(identity)
            if run.get("warmup") or run.get("return_code") != 0 or run.get("timed_out"):
                excluded.append({"source": str(path), "run_id": run.get("run_id"),
                    "warmup": run.get("warmup"), "return_code": run.get("return_code"), "timed_out": run.get("timed_out")})
                continue
            cfg = configurations.get(run["configuration"], {})
            # Retain complete configuration identity. Per-run output paths in the
            # actual command are omitted in favour of the template, when present.
            command_identity = cfg.get("command") or [str(x).replace(str(path.parent), "{run_root}") for x in run.get("command", [])]
            signature = {"configuration": run["configuration"], "workload": run["workload"],
                "context": run.get("context"), "requested_generation_tokens": run.get("requested_generation_tokens"),
                "prompt_sha256": run.get("prompt_sha256"), "metadata": run.get("metadata", {}),
                "command_template": command_identity, "env": cfg.get("env", {}),
                "cache_condition": run.get("cache_condition", "unknown"),
                "sample_interval_seconds": matrix.get("sample_interval_seconds"), "pdh": matrix.get("pdh")}
            key = json.dumps(signature, sort_keys=True)
            group = groups.setdefault(key, {"signature": signature, "runs": []})
            group["runs"].append({"source": str(path), "run_id": run["run_id"], "metrics": extract_metrics(run),
                "early_stopping": run.get("early_stopping"), "measurement_errors": run.get("measurement_errors", {})})
    rows = []
    for key, group in sorted(groups.items()):
        runs, sig = group["runs"], group["signature"]
        names = sorted({metric for run in runs for metric in run["metrics"]})
        metrics = {name: summarize([run["metrics"].get(name) for run in runs]) for name in names}
        command = sig["command_template"]
        def flag_value(flag):
            try: return int(command[command.index(flag)+1])
            except (ValueError, IndexError, TypeError): return None
        def string_flag(flag, default=None):
            try: return str(command[command.index(flag)+1])
            except (ValueError, IndexError, TypeError): return default
        streaming = sig["env"].get("BOVARY_STREAM") == "1"
        try:
            host_buffer_mib = int(sig["env"].get("BOVARY_STREAM_BUFFER_MIB", "8")) if streaming else None
        except (ValueError, TypeError):
            host_buffer_mib = None
        load_mode = string_flag("--load-mode", "mmap")
        try:
            load_buffer_mib = int(sig["env"].get("BOVARY_STREAM_LOAD_BUFFER_MIB", "64" if load_mode == "direct" else "1"))
        except (ValueError, TypeError):
            load_buffer_mib = None
        declared_bytes = sig["metadata"].get("model_bytes")
        if declared_bytes is not None and (not isinstance(declared_bytes, int) or isinstance(declared_bytes, bool) or declared_bytes <= 0):
            raise ValueError(f"{sig['configuration']}: metadata.model_bytes must be a positive integer")
        rows.append({"configuration": sig["configuration"], "workload": sig["workload"],
            "group_id": hashlib.sha256(key.encode()).hexdigest()[:10], "context": sig["context"],
            "requested_generation_tokens": sig["requested_generation_tokens"], "measured_repetitions": len(runs),
            "status": "shortlisted (>=3 repetitions)" if len(runs) >= 3 else "preliminary (<3 repetitions)",
            "model_bytes": declared_bytes if declared_bytes is not None else model_bytes,
            "model_bytes_source": "configuration metadata" if declared_bytes is not None else "global --model-bytes/default Q4 checkpoint",
            "model_sha256": sig["metadata"].get("model_sha256"),
            "cache_mib": flag_value("--cache-mib"),
            "stream_io": sig["env"].get("BOVARY_STREAM_IO", "buffered") if streaming else None,
            "host_buffer_mib": host_buffer_mib,
            "load_buffer_mib": load_buffer_mib,
            "prompt_batch": flag_value("--prompt-batch") or 1,
            "gpu_embedding": "--gpu-embedding" in command,
            "load_mode": load_mode,
            "pool_mode": sig["env"].get("BOVARY_STREAM_POOL", "serial") if streaming else None,
            "quality_evidence": sig["metadata"].get("quality_evidence", "unknown; inspect separate correctness results"),
            "signature": sig, "metrics": metrics, "runs": runs})
    return rows, excluded


def val(row, metric, statistic="median"):
    return nested(row.get("metrics", {}), metric, statistic)


def pareto_front(rows):
    """Five dimensions; missing dimensions never become zero/inferred."""
    dimensions = {"peak_ws_bytes": "min", "peak_gpu_dedicated_bytes": "min",
                  "peak_gpu_shared_bytes": "min", "decode_tokens_per_s": "max"}
    accepted, rejected = [], []
    for row in rows:
        missing = [metric for metric in dimensions if val(row, metric) is None]
        missing += [metric+" (three valid repetitions)" for metric in dimensions
            if val(row, metric) is not None and nested(row["metrics"], metric, "count") < 3]
        if row["measured_repetitions"] < 3: missing.append("three measured repetitions")
        if missing:
            rejected.append({"group_id": row["group_id"], "reason": "missing " + ", ".join(missing)})
        else: accepted.append(row)
    front = []
    def cohort(row):
        metadata = row["signature"]["metadata"]
        model_identity = tuple((key, metadata.get(key)) for key in ("model_sha256", "model_revision", "quantization"))
        return (row["workload"], row["context"], row["requested_generation_tokens"],
            row["signature"]["prompt_sha256"], row["prompt_batch"], val(row, "actual_generated_tokens", "min"),
            val(row, "actual_generated_tokens", "max"), model_identity)
    for row in accepted:
        dominated_by = []
        for other in accepted:
            if row is other or cohort(row) != cohort(other): continue
            pairs = [(val(other, name), val(row, name), direction) for name, direction in dimensions.items()]
            pairs.append((other["model_bytes"], row["model_bytes"], "min"))
            no_worse = all(a <= b if d == "min" else a >= b for a, b, d in pairs)
            better = any(a < b if d == "min" else a > b for a, b, d in pairs)
            if no_worse and better: dominated_by.append(other["group_id"])
        if not dominated_by: front.append(row)
    return {"dimensions": {**dimensions, "model_bytes": "min"},
        "scope": "Observed five-dimension front: minimize median per-run peak process WS, dedicated GPU allocation, shared GPU allocation and model bytes; maximize native decode speed. Cohorts require identical workload, context, requested generation length, prompt SHA-256, prompt batch, actual generated-token min/max range, and supplied model SHA-256/revision/quantization identity. Every dimension requires at least three valid repetitions. Private commit, virtual address space, OS cache, system RAM, power/thermal state, TTFT and task quality are excluded dimensions. Model identity keeps lossy weights separate; it does not prove quality equivalence. This is not a total-memory or recommendation frontier.",
        "nondominated_group_ids": [row["group_id"] for row in front], "excluded_groups": rejected}, front


def display(value, scale=1, digits=3):
    return f"{value/scale:.{digits}f}" if number(value) else "unknown"


def model_label(row):
    description = str(row["signature"]["metadata"].get("quantization", ""))
    return "mixed Q3 gate/up" if "Q3_K" in description else description or "unlabelled model"


def plot_label(row):
    known = {
        "B0-partial8-idle": "B0 partial GPU",
        "B1-cpu-experts-idle": "B1 CPU experts",
        "B2-cache512-idle": "B2 host + GPU cache",
        "E2-pinned32-idle": "SSD 32 MiB",
        "E1-minhost512-idle": "Serial 1 MiB transfer",
        "E2-pinned512-idle": "SSD 512 MiB",
        "E2-pinned1024-idle": "SSD 1 GiB",
        "E2-pinned1536-idle": "SSD 1.5 GiB",
        "E3-mixed1024-idle": "Mixed 1 GiB*",
        "V3-minhost512-load1": "Minimum RAM",
        "V3-balanced1024-load4": "Balanced",
        "V3-mixed1024-load4": "Minimum disk*",
        "B2-reference-workloads": "B2 resident control",
        "E2-balanced-workloads": "SSD balanced",
        "B0-batched128-workloads": "B0 prompt batch 128",
    }
    return known.get(row["configuration"], row["configuration"])


def plot_colour(row):
    if model_label(row) == "mixed Q3 gate/up": return "#bc663d"
    return "#247c78" if row["stream_io"] is not None else "#315b7d"


def plot_point(axis, row, label=False, offset=(6, 6), alignment="left"):
    x, y = val(row, "peak_ws_bytes"), val(row, "decode_tokens_per_s")
    if x is None or y is None: return False
    xerr = [[(x-val(row, "peak_ws_bytes", "min"))/GIB], [(val(row, "peak_ws_bytes", "max")-x)/GIB]]
    yerr = [[y-val(row, "decode_tokens_per_s", "min")], [val(row, "decode_tokens_per_s", "max")-y]]
    recommended = row["configuration"].startswith("V3-")
    axis.errorbar(x/GIB, y, xerr=xerr, yerr=yerr, fmt="D" if recommended else "o",
        color=plot_colour(row), capsize=3, markersize=6 if recommended else 5, linewidth=1.1)
    if label:
        axis.annotate(plot_label(row), (x/GIB, y), xytext=offset, textcoords="offset points",
            ha=alignment, fontsize=8)
    return True


def markdown_table(rows):
    header = ["Configuration / group", "Workload", "n", "Context", "Actual tokens", "Decode tok/s", "Peak WS GiB", "Decode WS GiB", "Peak private GiB", "Peak dedicated GiB", "Peak shared GiB", "Disk GiB"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|"*len(header)]
    for row in rows:
        entries = [row["configuration"]+" / "+row["group_id"], row["workload"], str(row["measured_repetitions"]),
            str(row["context"]) if row["context"] is not None else "unknown",
            display(val(row, "actual_generated_tokens"), digits=0), display(val(row, "decode_tokens_per_s")),
            display(val(row, "peak_ws_bytes"), GIB), display(val(row, "decode_ws_bytes"), GIB),
            display(val(row, "peak_private_commit_bytes"), GIB), display(val(row, "peak_gpu_dedicated_bytes"), GIB),
            display(val(row, "peak_gpu_shared_bytes"), GIB), display(row["model_bytes"], GIB)]
        lines.append("| " + " | ".join(entries) + " |")
    return "\n".join(lines)


def timing_table(rows):
    lines = ["| Configuration / group | Workload | Prompt batch | n | Load s | Prompt tok/s | TTFT s | Decode steps/s | Inter-token p95 s |",
        "|---|---|---:|---:|---|---|---|---|---|"]
    def interval(row, metric):
        return (display(val(row, metric)) + " [" + display(val(row, metric, "min")) +
            ", " + display(val(row, metric, "max")) + "]")
    for row in rows:
        values = [row["configuration"]+" / "+row["group_id"], row["workload"], str(row["prompt_batch"]),
            str(row["measured_repetitions"])]
        values += [interval(row, metric) for metric in ("model_load_s", "prompt_tokens_per_s",
            "ttft_internal_s", "decode_tokens_per_s", "intertoken_p95_s")]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_tables(rows, excluded, output):
    payload = {"groups": rows, "excluded_runs": excluded,
        "limitations": ["All tables show medians of per-run metrics unless column names specify otherwise.",
            "Missing metrics are unknown. Fewer than three repetitions are preliminary.",
            "Process WS excludes OS file cache and cannot establish minimum system RAM.",
            "WDDM dedicated/shared counters are allocation accounting, not independently proven physical residency.",
            "Shared GPU memory overlaps host RAM; these columns must not be summed into total memory.",
            "Model bytes use explicit per-configuration metadata.model_bytes when present, otherwise the supplied global GGUF size; never inferred from a filename. They are not total retained installation footprint.",
            "Architecture quality equivalence requires separate logit/token evidence."]}
    (output/"measured-comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    flat = []
    for row in rows:
        item = {key: row[key] for key in ("configuration", "group_id", "workload", "context", "requested_generation_tokens", "measured_repetitions", "status", "model_bytes", "model_bytes_source", "model_sha256", "cache_mib", "stream_io", "host_buffer_mib", "load_buffer_mib", "prompt_batch", "gpu_embedding", "load_mode", "pool_mode", "quality_evidence")}
        for metric, stats in row["metrics"].items():
            for statistic in ("median", "min", "max", "spread", "count"):
                item[metric+"_"+statistic] = stats.get(statistic) if stats else None
        flat.append(item)
    columns = sorted({key for row in flat for key in row})
    with (output/"measured-comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader(); writer.writerows(flat)
    text = "# Measured comparison\n\n" + markdown_table(rows) + "\n\n"
    text += "All numbers in the compact table are medians of successful measured repetitions. CSV/JSON contain min/max/spread and per-metric valid sample counts. Unknown data are not zero. Groups with fewer than three repetitions are preliminary.\n\n"
    text += "Process working set is a residency counter for the launched process. It excludes system file-cache/standby history and does not establish minimum system RAM. Private commitment, dedicated GPU allocation, and shared GPU allocation are different dimensions and overlap; do not add these into a total. Model disk size is one GGUF; installation footprint is reported separately.\n\n"
    text += "Native decode throughput uses decode-step count divided by summed decode duration. TTFT includes native backend/model startup. Streaming API read bytes per generated token include prompt work; logical process read bytes also include load. These are not measured physical disk bytes. Native phase alignment to process samples is approximate.\n\n"
    for row in rows:
        text += f"- `{row['group_id']}`: {row['status']}; cache condition `{row['signature']['cache_condition']}`; quality evidence: {row['quality_evidence']}.\n"
    text += "\nNo balanced/minimum-RAM/minimum-disk recommendation is inferred automatically. A recommendation requires correctness evidence, output review, complete memory accounting, and a defined useful-speed target.\n"
    (output/"measured-comparison.md").write_text(text, encoding="utf-8")
    (output/"timing-comparison.md").write_text("# Measured timing comparison\n\n" +
        "Values are repetition medians [minimum, maximum]. Prompt batching is part of the configuration; the practical ordinary batch-128 baseline is separate from batch-1 storage controls. TTFT includes backend/model startup, context creation and prompt evaluation. P95 is computed within each run before aggregation; it is not a pooled percentile.\n\n" +
        timing_table(rows) + "\n", encoding="utf-8")
    pareto, front = pareto_front(rows)
    (output/"nondominated.json").write_text(json.dumps(pareto, indent=2), encoding="utf-8")
    (output/"nondominated.md").write_text("# Observed nondominated configurations\n\n" + pareto["scope"] + "\n\n" +
        (markdown_table(front) if front else "No group has all required dimensions and three measured repetitions.") +
        "\n\nThis is a measurement frontier, not a verified quality-equivalent recommendation. Full global-cache and minimum-system-RAM conclusions remain separate.\n", encoding="utf-8")


def wilson_interval(successes, count, z=1.959963984540054):
    if count < 1: return None
    proportion = successes/count
    denominator = 1+z*z/count
    centre = (proportion+z*z/(2*count))/denominator
    half = z*math.sqrt(proportion*(1-proportion)/count+z*z/(4*count*count))/denominator
    return max(0, centre-half), min(1, centre+half)


def prepare_quality_report(quality, output, source_path):
    """Keep measured lossy quality separate from architecture Pareto analysis."""
    variants = quality.get("variants", {})
    rows = []
    for label in ("reference", "candidate"):
        variant = variants.get(label)
        if not isinstance(variant, dict): raise ValueError(f"Missing quality variant: {label}")
        size = variant.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"{label}: quality model bytes must be a positive integer")
        cases = variant.get("cases", [])
        if not cases: raise ValueError(f"{label}: no measured quality cases")
        row = {"variant": label, "model_bytes": size, "model_sha256": variant.get("sha256"),
            "case_count": len(cases), "model_path": variant.get("path")}
        for kind, field in (("constrained_accuracy", "constrained_correct"),
                            ("greedy_accuracy", "greedy_correct"),
                            ("single_letter_compliance", "exact_single_letter")):
            if any(not isinstance(case.get(field), bool) for case in cases):
                raise ValueError(f"{label}: missing/nonboolean {field}")
            correct = sum(case[field] for case in cases)
            rate = correct/len(cases)
            if variant.get(kind) is not None and (not number(variant[kind]) or abs(variant[kind]-rate)>1e-12):
                raise ValueError(f"{label}: {kind} disagrees with raw cases")
            low, high = wilson_interval(correct, len(cases))
            row.update({kind: rate, kind+"_successes": correct, kind+"_wilson95_low": low, kind+"_wilson95_high": high})
        nlls = [case.get("correct_letter_full_vocab_nll") for case in cases]
        if not all(number(value) for value in nlls): raise ValueError(f"{label}: missing/nonfinite correct-letter NLL")
        mean_nll = statistics.mean(nlls)
        if variant.get("mean_correct_letter_nll") is not None and (not number(variant["mean_correct_letter_nll"]) or abs(variant["mean_correct_letter_nll"]-mean_nll)>1e-9):
            raise ValueError(f"{label}: mean NLL disagrees with cases")
        row.update(mean_correct_letter_nll=mean_nll, per_case_correct_letter_nll=nlls)
        rows.append(row)
    if [case.get("id") for case in variants["reference"]["cases"]] != [case.get("id") for case in variants["candidate"]["cases"]]:
        raise ValueError("Quality case IDs/order differ across variants")
    result = {"source": str(source_path), "variants": rows, "context": quality.get("context"),
        "cases_sha256": quality.get("cases_sha256"),
        "mean_conditional_first_decision_kl_nats": quality.get("mean_conditional_first_decision_kl_nats"),
        "limitations": quality.get("limitations", []) + [
            "Wilson intervals are descriptive binomial intervals on a small fixed authored set, not evidence of broad task-population performance.",
            "Constrained-choice ranking, greedy answer extraction, and strict single-letter compliance are different measurements.",
            "These lossy-quality observations are not added to the architecture Pareto analysis as quality-equivalent configurations."]}
    (output/"quality-comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    flat = [{key: value for key, value in row.items() if key != "per_case_correct_letter_nll"} for row in rows]
    with (output/"quality-comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(flat[0])); writer.writeheader(); writer.writerows(flat)
    lines = ["# Disk size and measured diagnostic quality", "",
        "The small fixed question set measures lossy requantization separately from exact streaming/cache changes. Intervals are 95% Wilson binomial descriptions of this authored set; they do not establish broad model quality.", "",
        "| Variant | Model GiB | Cases | Constrained accuracy [Wilson interval] | Greedy accuracy [Wilson interval] | Single-letter compliance | Mean correct-letter NLL |",
        "|---|---:|---:|---|---|---:|---:|"]
    for row in rows:
        acc = lambda key: f"{row[key]:.1%} [{row[key+'_wilson95_low']:.1%}, {row[key+'_wilson95_high']:.1%}]"
        lines.append(f"| {row['variant']} | {row['model_bytes']/GIB:.3f} | {row['case_count']} | {acc('constrained_accuracy')} | {acc('greedy_accuracy')} | {row['single_letter_compliance']:.1%} | {row['mean_correct_letter_nll']:.6f} |")
    lines += ["", "Mean full-vocabulary KL(reference || candidate), conditional on the same first-decision input prefix: " +
        display(result["mean_conditional_first_decision_kl_nats"], digits=8) + " nats. This is not perplexity.", ""]
    lines += ["- " + str(item) for item in result["limitations"]]
    (output/"quality-comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def make_plots(rows, compression, output, quality=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    plots = []
    workloads = sorted({row["workload"] for row in rows})
    short = [row for row in rows if row["workload"] == "short" and
        val(row, "peak_ws_bytes") is not None and val(row, "decode_tokens_per_s") is not None]
    if short:
        from matplotlib.lines import Line2D
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), gridspec_kw={"width_ratios": (1, 1.2)})
        streamed = [row for row in short if row["stream_io"] is not None]
        for row in short:
            plot_point(axes[0], row, label=row["stream_io"] is None,
                offset=(-7, 9), alignment="right")
        for row in streamed:
            left = val(row, "peak_ws_bytes")/GIB < .60
            plot_point(axes[1], row, label=True, offset=(7, 6) if left else (-7, 6),
                alignment="left" if left else "right")
        axes[0].set(title="All short-workload configurations", xlabel="Peak process working set (GiB)",
            ylabel="Native decode steps / second", xlim=(0, max(val(r, "peak_ws_bytes", "max")/GIB for r in short)*1.12))
        if streamed:
            low = min(val(r, "peak_ws_bytes", "min")/GIB for r in streamed)
            high = max(val(r, "peak_ws_bytes", "max")/GIB for r in streamed)
            padding = max((high-low)*.2, .03)
            axes[1].set_xlim(low-padding, high+padding)
        axes[1].set(title="Streaming configurations: residency zoom", xlabel="Peak process working set (GiB)",
            ylabel="Native decode steps / second")
        for axis in axes:
            axis.grid(alpha=.2)
            axis.set_ylim(bottom=0)
        handles = [Line2D([], [], marker="o", linestyle="None", color="#315b7d", label="Ordinary Q4 controls"),
            Line2D([], [], marker="o", linestyle="None", color="#247c78", label="SSD streaming, unchanged Q4 weights"),
            Line2D([], [], marker="o", linestyle="None", color="#bc663d", label="*Lossy mixed weights; separate quality evidence"),
            Line2D([], [], marker="D", linestyle="None", color="#333333", label="Current launcher profiles")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035), ncol=2, fontsize=8)
        counts = ", ".join(str(n) for n in sorted({r["measured_repetitions"] for r in short}))
        fig.suptitle(f"RAM versus speed: medians, n={counts}; bars show repetition min–max", fontsize=12)
        fig.text(.02, .008, "Working set excludes OS file cache/standby. Private commitment and dedicated/shared GPU allocations are separate; no minimum-system-RAM claim. Diamonds identify current launcher profiles.", fontsize=8)
        fig.tight_layout(rect=(0, .12, 1, .93))
        for suffix in ("png", "svg"): fig.savefig(output/("ram-versus-speed."+suffix), dpi=180, bbox_inches="tight")
        plots.append("ram-versus-speed"); plt.close(fig)
    other_workloads = [workload for workload in workloads if workload != "short"]
    if other_workloads:
        fig, axes = plt.subplots(1, len(other_workloads), figsize=(5.2*len(other_workloads), 5), squeeze=False)
        for axis, workload in zip(axes[0], other_workloads):
            for row in [r for r in rows if r["workload"] == workload]:
                plot_point(axis, row, label=True, offset=(-6, 6) if row["stream_io"] is None else (6, 6),
                    alignment="right" if row["stream_io"] is None else "left")
            axis.set(title=workload, xlabel="Peak process working set (GiB)", ylabel="Native decode steps / second")
            axis.set_xlim(left=0); axis.set_ylim(bottom=0); axis.grid(alpha=.2)
        fig.suptitle("Longer workloads: medians with repetition min–max; prompt batch is labelled", fontsize=12)
        fig.text(.02, .012, "Context is fixed at the recorded capacity. Resident-set measurements exclude OS file cache. Consult the timing table for prefill speed and TTFT; decode speed alone does not describe long-prompt latency.", fontsize=8)
        fig.tight_layout(rect=(0, .06, 1, .92))
        for suffix in ("png", "svg"): fig.savefig(output/("workload-ram-versus-speed."+suffix), dpi=180, bbox_inches="tight")
        plots.append("workload-ram-versus-speed"); plt.close(fig)
    streaming = [row for row in rows if row["stream_io"] is not None and row["cache_mib"] is not None]
    if any(row["workload"] == "short" for row in streaming):
        streaming = [row for row in streaming if row["workload"] == "short"]
    if streaming:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.3))
        has_decode_io = any(val(row, "decode_io_bytes_per_generated_token_after_first") is not None for row in streaming)
        io_metric = "decode_io_bytes_per_generated_token_after_first" if has_decode_io else "stream_read_api_bytes_per_token_including_prompt"
        def cache_cohort(row):
            return (row["workload"], row["stream_io"], row["context"], row["host_buffer_mib"],
                row["gpu_embedding"], row["load_mode"], row["pool_mode"], row["requested_generation_tokens"],
                row["load_buffer_mib"], row["prompt_batch"], row["model_sha256"], row["model_bytes"], model_label(row))
        cohorts = sorted({cache_cohort(row) for row in streaming}, key=str)
        for cohort in cohorts:
            workload, mode, context, host_buffer, gpu_embedding, load_mode, pool_mode, requested, load_buffer, prompt_batch, model_sha, model_bytes, model_name = cohort
            selected = sorted([row for row in streaming if cache_cohort(row) == cohort], key=lambda x: x["cache_mib"])
            for axis, metric, scale in ((axes[0], "decode_tokens_per_s", 1), (axes[1], io_metric, 1024**2)):
                valid = [row for row in selected if val(row, metric) is not None]
                if valid: axis.plot([row["cache_mib"] for row in valid], [val(row, metric)/scale for row in valid],
                    marker="D" if model_name == "mixed Q3 gate/up" else "o", linestyle="-" if len(valid)>1 else "None",
                    label=f"{model_name}; {workload}; {pool_mode}; slot={host_buffer}, load={load_buffer} MiB; batch={prompt_batch}")
        axes[0].set(xlabel="Requested GPU expert-cache budget (MiB)", ylabel="Native decode steps / second", title="Measured cache budget versus decode speed")
        axes[1].set(xlabel="Requested GPU expert-cache budget (MiB)",
            ylabel="ReadFile API MiB / emitted token after first" if has_decode_io else "ReadFile API MiB / generated token (includes prompt)",
            title="Decode trace only; physical SSD bytes unknown" if has_decode_io else "Aligned/buffered API bytes; physical SSD bytes unknown")
        for axis in axes: axis.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles: fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .01), ncol=2, fontsize=7.5)
        fig.suptitle("Cache trade-offs; distinct checkpoints and buffer settings are never joined", fontsize=12)
        fig.tight_layout(rect=(0, .23, 1, .93))
        for suffix in ("png", "svg"): fig.savefig(output/("cache-tradeoff."+suffix), dpi=180, bbox_inches="tight")
        plots.append("cache-tradeoff"); plt.close(fig)
    if compression:
        summaries = compression.get("summaries", [])
        model_bytes = compression.get("model_bytes")
        labels, sizes, colors = [], [], []
        if number(model_bytes):
            labels.append("Q4_0 GGUF\nverified retained checkpoint")
            sizes.append(model_bytes/GIB); colors.append("#315b7d")
        for row in summaries:
            size = row.get("estimated_full_model_bytes")
            if number(size):
                samples = [sample for sample in compression.get("raw_samples", []) if sample.get("zstd_level") == row["level"]]
                evidence = "sampled byte-exact" if samples and all(sample.get("byte_exact") is True for sample in samples) else "roundtrip evidence incomplete"
                labels.append(f"zstd level {row['level']}\nwhole-model estimate; {evidence}")
                sizes.append(size/GIB); colors.append("#67a18b")
        if sizes:
            fig, axis = plt.subplots(figsize=(10, 5))
            bars = axis.barh(labels, sizes, color=colors)
            for i, (bar, size) in enumerate(zip(bars, sizes)):
                if i: bar.set_hatch("///")
                axis.text(size+.06, bar.get_y()+bar.get_height()/2, f"{size:.3f} GiB", va="center")
            axis.set_xlim(0, max(sizes)*1.22)
            axis.set(xlabel="Model-weight storage (GiB); hatched bars are estimates, not retained compressed files",
                ylabel="Representation evidence; no task-quality score measured",
                title="Disk size versus numerical-representation evidence")
            axis.invert_yaxis(); axis.grid(axis="x", alpha=.2)
            fig.text(.02, .01, "Lossless samples restored identical bytes. These compressed sizes are extrapolations; compression is not integrated into inference. Lossy quality is reported separately.", fontsize=8)
            fig.tight_layout(rect=(0, .04, 1, 1))
            for suffix in ("png", "svg"): fig.savefig(output/("disk-size-versus-quality-evidence."+suffix), dpi=180, bbox_inches="tight")
            plots.append("disk-size-versus-quality-evidence"); plt.close(fig)
    if quality:
        variants = quality["variants"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        for row, color in zip(variants, ("#315b7d", "#bc663d")):
            x = row["model_bytes"]/GIB
            for metric, marker, style in (("constrained_accuracy", "o", "constrained"), ("greedy_accuracy", "D", "greedy")):
                y = row[metric]
                axes[0].errorbar(x, y, yerr=[[max(0, y-row[metric+"_wilson95_low"])], [max(0, row[metric+"_wilson95_high"]-y)]],
                    fmt=marker, color=color, mfc=color if style == "constrained" else "white", capsize=4,
                    label=f"{row['variant']}: {style}, n={row['case_count']}")
            nlls = row["per_case_correct_letter_nll"]
            axes[1].scatter([x]*len(nlls), nlls, color=color, alpha=.25, s=20)
            axes[1].scatter([x], [row["mean_correct_letter_nll"]], color=color, marker="D", s=65,
                label=f"{row['variant']}: mean (faint points: cases)")
        axes[0].set(xlabel="Verified model-file size (GiB)", ylabel="Accuracy on fixed authored questions",
            ylim=(-.03, 1.06), title="Diagnostic accuracy; descriptive 95% Wilson intervals")
        axes[1].set(xlabel="Verified model-file size (GiB)", ylabel="Full-vocabulary correct-letter NLL (nats; lower is better)",
            title="Same-prefix first answer decision; not perplexity")
        for axis in axes:
            axis.grid(alpha=.2); axis.legend(fontsize=8)
            xs = [row["model_bytes"]/GIB for row in variants]
            padding = max((max(xs)-min(xs))*.25, .15)
            axis.set_xlim(min(xs)-padding, max(xs)+padding)
        fig.suptitle("Disk size versus measured lossy-quality diagnostics", fontsize=12)
        kl = display(quality.get("mean_conditional_first_decision_kl_nats"), digits=7)
        fig.text(.02, .015, f"Small fixed set; no broad quality-equivalence claim. Mean conditional KL(reference || candidate) = {kl} nats; context={quality.get('context')}.", fontsize=8)
        fig.tight_layout(rect=(0, .05, 1, .95))
        for suffix in ("png", "svg"): fig.savefig(output/("disk-size-versus-quality."+suffix), dpi=180, bbox_inches="tight")
        plots.append("disk-size-versus-quality"); plt.close(fig)
    return plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", nargs="+", required=True, help="Explicit run directories or results.json files; repeatable")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs")
    parser.add_argument("--compression", type=Path, default=ROOT/"outputs"/"compression.json")
    parser.add_argument("--model-bytes", type=int, default=14618145824)
    parser.add_argument("--quality", type=Path, help="Completed quality_eval.py quality-results.json; kept separate from architecture Pareto analysis")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.model_bytes <= 0: raise ValueError("model-bytes must be positive")
    inputs = [item for group in args.input for item in group]
    rows, excluded = load_groups(inputs, args.model_bytes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tables(rows, excluded, args.output_dir)
    compression = json.loads(args.compression.read_text(encoding="utf-8-sig")) if args.compression.exists() else None
    quality = prepare_quality_report(json.loads(args.quality.read_text(encoding="utf-8-sig")), args.output_dir, args.quality) if args.quality else None
    plots = [] if args.no_plots else make_plots(rows, compression, args.output_dir, quality)
    (args.output_dir/"report-generation.json").write_text(json.dumps({"inputs": inputs,
        "groups": len(rows), "excluded_runs": len(excluded), "plots": plots,
        "model_bytes": args.model_bytes, "compression": str(args.compression) if compression else None,
        "quality_source": str(args.quality) if args.quality else None}, indent=2), encoding="utf-8")
    print(json.dumps({"groups": len(rows), "excluded_runs": len(excluded), "plots": plots}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
