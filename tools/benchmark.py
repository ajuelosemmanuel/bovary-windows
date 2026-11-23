#!/usr/bin/env python3
"""Serial subprocess benchmark runner; see outputs/measurement-methods.md."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

try:
    from .windows_metrics import Sampler, power_snapshot
except ImportError:
    from windows_metrics import Sampler, power_snapshot


PROMPTS = {
    "short": "Explain why the sky looks blue in three complete sentences.",
    "medium": "A small library lends books, laptops, and musical instruments. Design a practical borrowing policy that balances access, maintenance, and fairness. Explain how the policy handles late returns, damaged items, scarce items, and people who cannot pay a deposit. Include concrete examples and discuss trade-offs.",
    "long": """Review this disaster recovery scenario and produce an operational plan with a timeline, explicit dependencies, and decision points.

A river flood has isolated a town of 18,000 people. The bridge to the regional warehouse is closed pending a structural inspection; a mountain road remains passable to small vehicles in daylight. The weather service expects another storm in 36 hours, but the rainfall forecast is uncertain.

The hospital has 48 occupied beds, including six patients who need continuous powered equipment. Its generator has fuel for 20 hours at current load. A nearby clinic has a smaller generator and refrigeration for medicine, but its roof leaks. Moving patients requires medical staff, vehicles, and reliable information about road conditions.

The water plant is operational but one intake sensor is unreliable. Stored treated water will last one day at normal demand. A repair team can inspect the sensor after access to its electrical cabinet is confirmed safe. The sanitation team reports two damaged sewer pumps; residents need clear advice before any contamination is confirmed.

Two schools can shelter displaced residents. The north school has accessible entrances and a working kitchen but limited toilets. The south school has more space and toilets, but its access road floods at high tide. One hundred shelter residents have limited English proficiency, and several families need charging for mobility devices.

Mobile coverage is intermittent. The fire station has radios, a paper street map, and two trained dispatchers. Volunteers are arriving without assignments. The supermarket offers food and three vans, while a local contractor offers pumps whose fuel requirements have not been checked. Only four drivers have experience on the mountain road.

The municipal emergency budget can buy either an immediate fuel shipment at a premium or additional road inspection equipment; an outside grant may arrive tomorrow. Discuss what can proceed now, what needs verification, and which actions must wait. Explain how to allocate supplies fairly, track commitments, avoid conflicting messages, protect vulnerable residents, and revise the plan if the bridge or hospital generator remains unavailable. Include a first-six-hours checklist and a handover procedure for the next shift.""",
    "sustained": "Write a detailed practical guide to restoring an abandoned community garden over one year. Cover soil assessment, water, paths, accessibility, seasonal planting, compost, tools, budgets, volunteer coordination, pests, record keeping, and evaluating success. Use varied concrete examples and continue developing new details rather than repeating phrases.",
}


def parse_timings(text):
    """Parse only explicitly reported engine metrics, never requested token counts."""
    metrics = {}
    for name, pattern in {
        "engine_load_ms": r"load time\s*=\s*([\d.]+)\s*ms",
        "engine_prompt_ms": r"prompt eval time\s*=\s*([\d.]+)\s*ms",
        "engine_prompt_tokens": r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens?",
        "engine_prompt_tokens_per_second": r"prompt eval time[^\n]*?([\d.]+)\s*tokens per second",
        "engine_decode_ms": r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms",
        "engine_decode_evaluations": r"(?<!prompt )eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*runs?",
        "engine_decode_tokens_per_second": r"(?<!prompt )eval time[^\n]*?([\d.]+)\s*tokens per second",
        "engine_total_ms": r"total time\s*=\s*([\d.]+)\s*ms",
        "engine_generated_tokens": r"(?:generated|predicted) tokens\s*[:=]\s*(\d+)",
    }.items():
        found = re.findall(pattern, text, flags=re.I)
        if found:
            metrics[name] = float(found[-1])
    # llama-bench JSON reports pp/tg independently. Preserve source fields.
    try:
        data = json.loads(text)
        if isinstance(data, list):
            metrics["llama_bench_rows"] = data
    except (ValueError, TypeError):
        pass
    stream = re.findall(r"bovary_stream: (?=[^\n]*\bread_calls=)([^\n]*)", text)
    if stream:
        counters = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", stream[-1])}
        field_map = {"read_calls": "stream_read_calls", "requested_bytes": "stream_upload_payload_bytes",
            "io_bytes": "stream_read_api_bytes", "read_ns": "stream_read_nanoseconds",
            "upload_wait_ns": "stream_upload_wait_nanoseconds", "host_buffer_bytes": "stream_host_buffer_bytes",
            "pinned_bytes": "stream_explicit_pinned_bytes"}
        metrics.update({target: counters[source] for source, target in field_map.items() if source in counters})
        pool = re.search(r"(?:^|\s)pool=(\w+)\b", stream[-1])
        if pool: metrics["stream_pool_mode"] = pool.group(1)
        if counters.get("read_calls") and "read_ns" in counters:
            metrics["stream_mean_read_call_seconds"] = counters["read_ns"]/counters["read_calls"]/1e9
        if counters.get("read_ns") and "io_bytes" in counters:
            metrics["stream_read_api_bytes_per_second"] = counters["io_bytes"]/(counters["read_ns"]/1e9)
    cache = re.findall(r"llama_moe_cache: ubatch\s*<=\s*8: hits = (\d+), misses = (\d+) \((\d+) on the CPU\), hit rate = ([\d.]+)%, uploaded = ([\d.]+) MiB", text)
    if cache:
        hits, misses, cpu, hit_rate, uploaded = cache[-1]
        metrics.update(expert_cache_hits=int(hits), expert_cache_misses=int(misses),
            expert_cache_cpu_misses=int(cpu), expert_cache_hit_rate_percent=float(hit_rate),
            expert_cache_logged_uploaded_mib=float(uploaded))
    load_records = []
    load_fields = {"host_buffer_slots": "load_buffer_slots", "host_buffer_bytes": "load_host_buffer_bytes",
        "pinned_bytes": "load_pinned_bytes", "slot_requested_bytes": "load_slot_requested_bytes",
        "direct_io": "load_direct_io", "async_uploads": "load_async_uploads"}
    # A separate prefix prevents temporary model-loader pools from being mistaken
    # for the expert-stream pool. Multiple contexts log separate sequential pools.
    for line in re.findall(r"\bbovary_load:\s*([^\r\n]+)", text):
        values = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", line)}
        load_records.append({target: values[source] for source, target in load_fields.items() if source in values})
    if load_records:
        metrics["load_pool_records"] = load_records
        metrics["load_pool_record_count"] = len(load_records)
        metrics["load_pool_metric_aggregation"] = "Maximum individual logged temporary loader pool; not a sum or a concurrent total"
        for target in load_fields.values():
            values = [row[target] for row in load_records if target in row]
            if values: metrics[target] = max(values)
    return metrics


def summarize_samples(samples):
    result = {}
    keys = sorted({key for sample in samples for key in sample if key != "elapsed_seconds"})
    for key in keys:
        values = [s[key] for s in samples if isinstance(s.get(key), (int, float))]
        if values:
            tail = values[len(values)//2:]
            result[key] = {"first": values[0], "last": values[-1], "min": min(values),
                "max": max(values), "median": statistics.median(values),
                "second_half_median": statistics.median(tail), "sample_count": len(values)}
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered: return None
    index = (len(ordered)-1) * fraction
    lower = int(index)
    upper = min(lower+1, len(ordered)-1)
    return ordered[lower] + (ordered[upper]-ordered[lower]) * (index-lower)


def native_metrics_summary(native):
    """Keep native token counts/timestamps distinct from parsed perf output."""
    metrics = {}
    for key in ("model_load_s", "context_ready_s", "prompt_tokens", "prompt_s", "prompt_tokens_per_s",
                "ttft_process_internal_s", "generation_s", "total_s", "actual_generated_tokens"):
        if isinstance(native.get(key), (int, float)):
            metrics["native_"+key] = native[key]
    steps = native.get("decode_step_s", [])
    if steps and all(isinstance(x, (int, float)) and x >= 0 for x in steps):
        metrics["native_decode_step_count"] = len(steps)
        metrics["native_decode_tokens_per_second"] = len(steps)/sum(steps) if sum(steps) else None
        metrics["native_decode_step_p50_seconds"] = percentile(steps, .5)
        metrics["native_decode_step_p95_seconds"] = percentile(steps, .95)
        metrics["native_decode_step_max_seconds"] = max(steps)
    elapsed = native.get("token_elapsed_s", [])
    if len(elapsed) > 1 and all(isinstance(x, (int, float)) for x in elapsed):
        intervals = [b-a for a, b in zip(elapsed, elapsed[1:])]
        if min(intervals) >= 0:
            metrics["native_intertoken_p50_seconds"] = percentile(intervals, .5)
            metrics["native_intertoken_p95_seconds"] = percentile(intervals, .95)
            metrics["native_intertoken_max_seconds"] = max(intervals)
    return metrics


def analyze_stream_trace(path, native, footer_metrics=None, pool_mode=None):
    """Analyse engine trace deltas using its own monotonic-clock epoch."""
    if not path.is_file(): return {"available": False, "reason": "No stream trace file"}, {}
    fields = ("hits", "misses", "requested_bytes", "io_bytes", "read_calls", "read_ns", "upload_wait_ns")
    rows, errors = [], []
    try:
        with path.open("rb") as raw_file:
            raw_file.seek(0, os.SEEK_END)
            if not raw_file.tell(): errors.append({"error": "Empty stream trace"})
            else:
                raw_file.seek(-1, os.SEEK_END)
                if raw_file.read(1) != b"\n": errors.append({"error": "Trace lacks terminal newline; may be truncated"})
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not set(("clock_ns", *fields)).issubset(reader.fieldnames or []):
                errors.append({"error": "Missing required trace columns"})
            for line, raw in enumerate(reader, start=2):
                try:
                    if None in raw: raise ValueError("Trace column count mismatch")
                    row = {key: int(raw[key]) for key in ("clock_ns", *fields)}
                    if any(value < 0 for value in row.values()): raise ValueError("negative trace counter")
                    rows.append(row)
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append({"line": line, "error": str(exc)})
    except (OSError, csv.Error) as exc:
        errors.append({"error": str(exc)})
    analysis = {"available": True, "rows": len(rows), "errors": errors,
        "clock_method": "Engine integer steady_clock epoch; compare clock_ns-clock_start_ns to native token elapsed seconds. No outer-process clock alignment.",
        "denominators": "Per emitted token after first uses max(actual_generated_tokens-1,0). Per decode evaluation uses len(decode_step_s); EOG can make these differ.",
        "scope": "Trace sums are API-level reader counters and upload payload, not physical SSD/PCIe bus counters. Native decode timing includes compute outside traced read/upload stages."}
    footer_pool = footer_metrics.get("stream_pool_mode") if footer_metrics else None
    selected_pool = pool_mode or footer_pool or "serial"
    analysis.update(pool_mode=selected_pool,
        pool_mode_source="resolved_environment" if pool_mode else "reader_footer" if footer_pool else "legacy_serial_default",
        gpu_transfer_elapsed_seconds=None,
        upload_wait_semantics="Enqueue/event overhead plus residual completion wait; does not include the full DMA interval" if selected_pool == "pinned2" else "Enqueue plus immediate backend synchronization; not isolated DMA duration")
    if pool_mode and footer_pool and pool_mode != footer_pool:
        errors.append({"error": "Resolved pool mode differs from reader footer"})
    if selected_pool == "pinned2" and footer_metrics is not None and footer_pool is None:
        errors.append({"error": "pinned2 requested but reader footer does not verify pool mode"})
    if not rows: errors.append({"error": "No complete trace rows"})
    if errors:
        analysis["valid"] = False
        return analysis, {}
    metrics = {}
    totals = {key: sum(row[key] for row in rows) for key in fields}
    metrics.update({"stream_trace_total_"+key: value for key, value in totals.items()})
    analysis["totals"] = totals
    footer_fields = {"requested_bytes": "stream_upload_payload_bytes", "io_bytes": "stream_read_api_bytes",
        "read_calls": "stream_read_calls", "read_ns": "stream_read_nanoseconds", "upload_wait_ns": "stream_upload_wait_nanoseconds"}
    if footer_metrics is not None:
        missing = [name for name, metric in footer_fields.items() if metric not in footer_metrics]
        mismatch = {name: {"trace": totals[name], "footer": footer_metrics[metric]}
            for name, metric in footer_fields.items() if metric in footer_metrics and totals[name] != footer_metrics[metric]}
        analysis["footer_validation"] = {"missing": missing, "mismatched": mismatch,
            "passed": not missing and not mismatch}
        if missing or mismatch:
            errors.append({"error": "Trace/footer totals missing or mismatched; completeness not established"})
            analysis["valid"] = False
            return analysis, {}
    else:
        analysis["footer_validation"] = {"passed": None, "reason": "Footer validation not requested"}
    analysis["valid"] = True
    start_ns = native.get("clock_start_ns")
    token_times = native.get("token_elapsed_s", [])
    if not isinstance(start_ns, int) or not token_times or not isinstance(token_times[0], (int, float)):
        analysis["decode_unavailable_reason"] = "Missing integer native clock_start_ns or first token timestamp"
        return analysis, metrics
    first_offset_ns = token_times[0]*1e9
    end_offset_ns = native["total_s"]*1e9 if isinstance(native.get("total_s"), (int, float)) else float("inf")
    decoded = [row for row in rows if first_offset_ns < row["clock_ns"]-start_ns <= end_offset_ns]
    decode_totals = {key: sum(row[key] for row in decoded) for key in fields}
    analysis.update(decode_rows=len(decoded), decode_totals=decode_totals,
        decode_boundary="Strictly after first emitted token through native total_s; includes a final EOG evaluation if present.")
    metrics.update({"stream_trace_decode_"+key: value for key, value in decode_totals.items()})
    evaluations = len(native.get("decode_step_s", []))
    emitted_after_first = max(native.get("actual_generated_tokens", 0)-1, 0)
    metrics["stream_trace_decode_evaluation_count"] = evaluations
    metrics["stream_trace_emitted_tokens_after_first"] = emitted_after_first
    decode_s = sum(native.get("decode_step_s", []))
    for key in ("io_bytes", "requested_bytes"):
        if emitted_after_first:
            metrics["decode_"+key+"_per_generated_token_after_first"] = decode_totals[key]/emitted_after_first
        if evaluations:
            metrics["decode_"+key+"_per_evaluation"] = decode_totals[key]/evaluations
        if decode_s:
            metrics["decode_"+key+"_per_native_decode_second"] = decode_totals[key]/decode_s
    calls, read_ns, wait_ns = (decode_totals[key] for key in ("read_calls", "read_ns", "upload_wait_ns"))
    if calls: metrics["decode_mean_read_call_seconds"] = read_ns/calls/1e9
    if read_ns: metrics["decode_read_api_bytes_per_read_second"] = decode_totals["io_bytes"]/(read_ns/1e9)
    if wait_ns and selected_pool == "serial":
        metrics["decode_upload_payload_bytes_per_upload_wait_second"] = decode_totals["requested_bytes"]/(wait_ns/1e9)
    elif selected_pool != "serial":
        analysis["upload_wait_throughput_unavailable_reason"] = "Residual wait is not full transfer duration; dividing payload by it would invent bandwidth"
    if read_ns+wait_ns:
        metrics["decode_read_fraction_of_read_plus_upload_wait"] = read_ns/(read_ns+wait_ns)
        metrics["decode_upload_wait_fraction_of_read_plus_upload_wait"] = wait_ns/(read_ns+wait_ns)
    if decode_s:
        metrics["decode_read_fraction_of_native_decode_time"] = read_ns/1e9/decode_s
        metrics["decode_upload_wait_fraction_of_native_decode_time"] = wait_ns/1e9/decode_s
    hits, misses = decode_totals["hits"], decode_totals["misses"]
    if hits+misses: metrics["decode_expert_cache_hit_rate_percent"] = 100*hits/(hits+misses)
    return analysis, metrics


def aggregate_runs(runs):
    groups = {}
    for run in runs:
        if run["warmup"] or run["return_code"] != 0 or run["timed_out"]:
            continue
        group = groups.setdefault((run["configuration"], run["workload"]), [])
        group.append(run)
    output = []
    for (configuration, workload), group in sorted(groups.items()):
        numeric = {}
        for run in group:
            flat = {"end_to_end_seconds": run["end_to_end_seconds"],
                    "launch_to_first_generation_stdout_seconds": run.get("launch_to_first_generation_stdout_seconds")}
            flat.update({k: v for k, v in run["engine_metrics"].items() if isinstance(v, (int, float))})
            for key, summary in run["memory_summary"].items():
                flat[key + "_peak"] = summary["max"]
                flat[key + "_second_half_median"] = summary["second_half_median"]
            for key, summary in run.get("decode_memory_summary", {}).items():
                flat[key + "_decode_median"] = summary["median"]
                flat[key + "_decode_peak"] = summary["max"]
            flat["process_lifetime_read_transfer_bytes_per_generated_token_including_load"] = run.get("process_lifetime_read_transfer_bytes_per_generated_token_including_load")
            for key, value in flat.items():
                if isinstance(value, (int, float)):
                    numeric.setdefault(key, []).append(value)
        output.append({"configuration": configuration, "workload": workload, "measured_repetitions": len(group),
            "metrics": {k: {"median": statistics.median(v), "min": min(v), "max": max(v),
                "spread": max(v)-min(v), "count": len(v)} for k, v in numeric.items()}})
    return output


class SerialLock:
    def __init__(self, path): self.path, self.file = Path(path), None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Windows permits locking a byte beyond EOF. The lock needs no contents.
        self.file = self.path.open("a+b")
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("Another benchmark holds the serial lock") from None
        return self
    def __exit__(self, *_):
        if os.name == "nt":
            import msvcrt
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()


def read_pipe(pipe, target, timestamps, start, generation=False):
    with target.open("wb") as output:
        while True:
            chunk = os.read(pipe.fileno(), 4096)
            if not chunk: break
            now = time.perf_counter()-start
            output.write(chunk)
            output.flush()
            timestamps.append({"elapsed_seconds": now, "bytes": len(chunk)})
            if generation and chunk.strip() and "first_generation" not in timestamps.__dict__:
                timestamps.__dict__["first_generation"] = now
    pipe.close()


class TimestampList(list): pass


class GpuTelemetry:
    """Whole-device NVML telemetry through one nvidia-smi subprocess per run."""
    def __init__(self, directory, enabled=True):
        self.process, self.handles, self.error, self.command = None, [], None, None
        self.enabled, self.stopped_by_runner = enabled, False
        self.stderr_path = directory/"gpu-device.stderr.txt"
        if not enabled: return
        executable = shutil.which("nvidia-smi")
        if not executable:
            self.error = "nvidia-smi not found on PATH"
            return
        self.command = [executable,
            "--query-gpu=timestamp,index,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate,clocks_event_reasons.active,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_power_cap",
            "--format=csv", "--loop-ms=1000"]
        try:
            output = (directory/"gpu-device.csv").open("wb")
            self.handles.append(output)
            errors = self.stderr_path.open("wb")
            self.handles.append(errors)
            self.process = subprocess.Popen(self.command, stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as exc:
            self.error = repr(exc)
            for handle in self.handles: handle.close()
            self.handles = []

    def stop(self):
        if self.process:
            if self.process.poll() is None:
                self.stopped_by_runner = True
                self.process.terminate()
                try: self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            elif self.process.returncode:
                self.error = f"nvidia-smi exited with code {self.process.returncode}"
        for handle in self.handles: handle.close()
        self.handles = []

    def metadata(self):
        return {"enabled": self.enabled, "command": self.command, "error": self.error,
            "return_code": self.process.returncode if self.process else None,
            "stopped_by_runner": self.stopped_by_runner,
            "stderr": self.stderr_path.read_text(encoding="utf-8", errors="replace") if self.stderr_path.exists() else None,
            "scope": "Whole-device NVML/nvidia-smi telemetry including display/driver/other processes; not per-process physical residency. WDDM shared memory is measured separately by PDH. One-second sampling can miss peaks; telemetry has overhead."}


class DiskMonitor:
    """Optional whole-volume free-space samples; never an inferred file size."""
    def __init__(self, path=None):
        if path is not None and not isinstance(path, str):
            raise ValueError("monitor_disk_path must be a string")
        self.path, self.error = path, None

    def sample(self):
        if self.path is None: return {}
        try: return {"disk_free_bytes": shutil.disk_usage(self.path).free}
        except (OSError, ValueError) as exc:
            self.error = repr(exc)
            return {"disk_free_bytes": None}

    def summary(self, before, after, samples):
        if self.path is None: return {"enabled": False}
        values = [sample["disk_free_bytes"] for sample in samples if isinstance(sample.get("disk_free_bytes"), int)]
        values += [value for value in (before, after) if isinstance(value, int)]
        minimum = min(values) if values else None
        return {"enabled": True, "path": self.path, "resolved_path": os.path.abspath(self.path), "error": self.error,
            "free_before_launch_bytes": before, "free_after_exit_bytes": after,
            "minimum_sampled_free_bytes": minimum, "valid_sample_count": len(values),
            "sampled_peak_net_volume_space_increase_bytes": max(0, before-minimum) if before is not None and minimum is not None else None,
            "net_volume_space_increase_after_exit_bytes": before-after if before is not None and after is not None else None,
            "limitations": "Whole-volume net free-space change includes unrelated writes/deletes; not a per-file allocation counter. Sampling can miss transient installation peaks. No disk queries occur unless explicitly enabled."}


def run_name(configuration, workload, repetition, warmup):
    run_id = f"{configuration['id']}-{workload}-{'warmup' if warmup else 'rep'}{repetition}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("Configuration/workload identifiers must be filesystem-safe")
    return run_id


def run_one(config, configuration, workload, repetition, warmup, output_dir):
    run_id = run_name(configuration, workload, repetition, warmup)
    directory = output_dir / run_id
    directory.mkdir(parents=True, exist_ok=False)
    prompt = PROMPTS[workload]
    prompt_file = directory / "prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    n_predict = config.get("generation_tokens", {}).get(workload, config.get("n_predict", 128))
    context = configuration.get("context", config.get("context", 4096))
    variables = {"prompt_file": str(prompt_file.resolve()), "prompt": prompt,
                 "n_predict": str(n_predict), "ctx": str(context),
                 "metrics_file": str((directory/"native-metrics.json").resolve()),
                 "logits_file": str((directory/"logits.f32").resolve()),
                 "stream_trace_file": str((directory/"stream-layers.csv").resolve())}
    # Replace only documented placeholders. Windows sandbox temp paths contain
    # literal {GUID} components; arbitrary braces and shell-free code stay intact.
    placeholder = re.compile(r"\{(" + "|".join(map(re.escape, variables)) + r")\}")
    command = [placeholder.sub(lambda match: variables[match.group(1)], arg) for arg in configuration["command"]]
    environment_overrides = {key: placeholder.sub(lambda match: variables[match.group(1)], value)
        if isinstance(value, str) else value for key, value in configuration.get("env", {}).items()}
    disk_monitor = DiskMonitor(configuration.get("monitor_disk_path", config.get("monitor_disk_path")))
    baseline_sampler = Sampler(os.getpid(), enable_pdh=False)
    try:
        system_before_launch = {key: value for key, value in baseline_sampler.sample().items() if key.startswith("system_")}
    finally:
        baseline_sampler.close()
    power_before_launch = power_snapshot()
    disk_free_before = disk_monitor.sample().get("disk_free_bytes")
    telemetry = GpuTelemetry(directory, enabled=configuration.get("gpu_telemetry", config.get("gpu_telemetry", True)))
    start_wall = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=configuration.get("cwd"), env={**os.environ, **environment_overrides},
            stdin=subprocess.DEVNULL)
    except BaseException:
        telemetry.stop()
        raise
    stdout_events, stderr_events = TimestampList(), TimestampList()
    readers = [threading.Thread(target=read_pipe, args=(process.stdout, directory/"stdout.bin", stdout_events, start,
                    configuration.get("stdout_is_generation", False)), daemon=True),
               threading.Thread(target=read_pipe, args=(process.stderr, directory/"stderr.bin", stderr_events, start), daemon=True)]
    for thread in readers: thread.start()
    sampler_start = time.perf_counter()
    try:
        sampler = Sampler(process.pid, enable_pdh=config.get("pdh", True))
    except BaseException:
        if process.poll() is None: process.kill()
        process.wait(timeout=30)
        for thread in readers: thread.join(timeout=30)
        telemetry.stop()
        raise
    sampler_init_seconds = time.perf_counter()-sampler_start
    samples, timed_out = [], False
    system_after_exit = {}
    disk_free_after = None
    interval = config.get("sample_interval_seconds", 0.1)
    timeout = config.get("timeout_seconds", 1800)
    try:
        while True:
            elapsed = time.perf_counter()-start
            samples.append({"elapsed_seconds": elapsed, **sampler.sample(), **disk_monitor.sample()})
            if process.poll() is not None: break
            if elapsed >= timeout:
                timed_out = True
                process.kill()
                break
            time.sleep(interval)
        process.wait(timeout=30)
        for thread in readers: thread.join(timeout=30)
        system_after_exit = {key: value for key, value in sampler.sample().items() if key.startswith("system_")}
        disk_free_after = disk_monitor.sample().get("disk_free_bytes")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        sampler.close()
        telemetry.stop()
    elapsed = time.perf_counter()-start
    # Preserve per-stream events rather than falsely treating pipe chunks as tokens.
    (directory/"output-events.json").write_text(json.dumps({"stdout": stdout_events, "stderr": stderr_events}, indent=2), encoding="utf-8")
    keys = ["elapsed_seconds"] + sorted({key for sample in samples for key in sample if key != "elapsed_seconds"})
    with (directory/"samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        writer.writerows(samples)
    text = (directory/"stderr.bin").read_bytes().decode("utf-8", errors="replace") + "\n" + (directory/"stdout.bin").read_bytes().decode("utf-8", errors="replace")
    metrics = parse_timings(text)
    native = {}
    native_error = None
    if (directory/"native-metrics.json").exists():
        try:
            native = json.loads((directory/"native-metrics.json").read_text(encoding="utf-8"))
            if not isinstance(native, dict): raise ValueError("Native metrics must be an object")
        except (ValueError, OSError) as exc:
            native_error = str(exc)
            native = {}
    metrics.update(native_metrics_summary(native))
    trace_analysis, trace_metrics = analyze_stream_trace(directory/"stream-layers.csv", native, metrics,
        pool_mode=environment_overrides.get("BOVARY_STREAM_POOL"))
    metrics.update(trace_metrics)
    decode_start = (native.get("token_elapsed_s") or [None])[0]
    decode_end = (native.get("token_elapsed_s") or [None])[-1]
    decode_samples = [sample for sample in samples if isinstance(decode_start, (float, int))
        and decode_start <= sample["elapsed_seconds"] <= decode_end]
    ready_start = native.get("context_ready_s")
    ready_end = native.get("total_s", float("inf"))
    ready_samples = [sample for sample in samples if isinstance(ready_start, (float, int))
        and ready_start <= sample["elapsed_seconds"] <= ready_end]
    result = {"schema_version": 1, "run_id": run_id, "configuration": configuration["id"],
        "workload": workload, "warmup": warmup, "repetition": repetition, "start_utc": start_wall,
        "command": command, "cwd": configuration.get("cwd"), "pid": process.pid,
        "environment_overrides": environment_overrides,
        "context": context, "requested_generation_tokens": n_predict,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_utf8_bytes": len(prompt.encode()),
        "return_code": process.returncode, "timed_out": timed_out, "end_to_end_seconds": elapsed,
        "launch_to_first_generation_stdout_seconds": stdout_events.__dict__.get("first_generation"),
        "ttft_boundary": "process launch to first non-whitespace stdout pipe chunk; requires stdout_is_generation assertion; includes load; affected by buffering" if configuration.get("stdout_is_generation") else "unavailable",
        "actual_generated_tokens": native.get("actual_generated_tokens", metrics.get("engine_generated_tokens")),
        "early_stopping": native.get("early_stop"),
        "engine_metrics": metrics, "memory_summary": summarize_samples(samples), "measurement_errors": sampler.errors,
        "decode_memory_summary": summarize_samples(decode_samples), "ready_memory_summary": summarize_samples(ready_samples),
        "system_before_launch": system_before_launch, "system_after_exit": system_after_exit,
        "power_before_launch": power_before_launch, "gpu_device_telemetry": telemetry.metadata(),
        "disk_monitor": disk_monitor.summary(disk_free_before, disk_free_after, samples),
        "phase_alignment": "Approximate Popen launch plus native internal elapsed time; native clock starts before backend init, a few milliseconds after process entry. Decode window is first through last emitted token; ready window ends at native total_s. Short phases can have no samples.",
        "sampler_initialization_seconds": sampler_init_seconds, "native_metrics_error": native_error,
        "stream_trace_analysis": trace_analysis,
        "metadata": configuration.get("metadata", {}), "cache_condition": configuration.get("cache_condition", "unknown"),
        "limitations": ["Process restart does not flush file cache.",
            "Decode evaluations are not guaranteed to equal emitted tokens; actual_generated_tokens stays null unless explicitly reported.",
            "Pipe chunk events are not token timings; native token_elapsed_s/decode_step_s provide token latency only when emitted by the instrumented runner.",
            "Second-half sample median is a time-window proxy, not an engine-labelled decode phase.",
            "Native process IO transfer counts include non-disk reads and buffered cache hits.",
            "WDDM PDH figures are OS-accounted adapter/process allocations, not independently verified physical residency.",
            "No per-allocation pinned host, H2D transfer, expert cache-hit, or KV counters without engine instrumentation."]}
    if result["actual_generated_tokens"] is not None:
        if result["early_stopping"] is None:
            result["early_stopping"] = result["actual_generated_tokens"] < n_predict
        reads = [sample["process_io_read_transfer_bytes_total"] for sample in samples if sample.get("process_io_read_transfer_bytes_total") is not None]
        result["process_lifetime_read_transfer_bytes_per_generated_token_including_load"] = max(reads)/result["actual_generated_tokens"] if reads and result["actual_generated_tokens"] > 0 else None
        if result["actual_generated_tokens"] > 0:
            for key in ("stream_read_api_bytes", "stream_upload_payload_bytes"):
                if key in metrics:
                    metrics[key+"_per_generated_token_including_prompt"] = metrics[key]/result["actual_generated_tokens"]
    (directory/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_schedule(config):
    """Opt-in interleaving balances order; the original default order is retained."""
    configurations = config["configurations"]
    workloads = config.get("workloads", list(PROMPTS))
    warmups, repetitions = config.get("warmups", 1), config.get("repetitions", 3)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (warmups, repetitions)):
        raise ValueError("warmups and repetitions must be nonnegative integers")
    if config.get("interleave_repetitions", False):
        for warmup, count in ((True, warmups), (False, repetitions)):
            for repetition in range(1, count+1):
                for workload in workloads:
                    for configuration in configurations:
                        yield configuration, workload, repetition, warmup
    else:
        for configuration in configurations:
            for workload in workloads:
                for warmup, count in ((True, warmups), (False, repetitions)):
                    for repetition in range(1, count+1):
                        yield configuration, workload, repetition, warmup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    for name, default in (("sample_interval_seconds", .1), ("timeout_seconds", 1800)):
        value = config.get(name, default)
        if type(value) not in (int, float) or not 0 < value < float("inf"):
            parser.error(f"{name} must be a positive finite number")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    workloads = config.get("workloads", list(PROMPTS))
    if not set(workloads).issubset(PROMPTS): raise ValueError("Unknown workload")
    lock = Path(config.get("lock_file", Path(__file__).resolve().parents[1] / "work" / "benchmark.lock")).resolve()
    directories = [output]
    lock_entries = set()
    if lock == output:
        parser.error("Benchmark lock conflicts with the output directory")
    if lock.is_relative_to(output):
        lock_top = output / lock.relative_to(output).parts[0]
        artifacts = {"configuration.json", "host.json", "results.json", "summary.json"}
        artifacts.update(run_name(*run) for run in run_schedule(config))
        if lock_top in {output / name for name in artifacts}:
            parser.error("Benchmark lock conflicts with an output artifact")
        lock_entries = {lock}
        for directory in lock.parents:
            if directory == output:
                break
            directories.append(directory)
            lock_entries.add(directory)
    with SerialLock(lock):
        if any(entry not in lock_entries for directory in directories for entry in directory.iterdir()):
            parser.error("Benchmark output directory must be empty apart from its lock")
        try:
            with (output/"configuration.json").open("x", encoding="utf-8") as file:
                json.dump(config, file, indent=2)
        except FileExistsError:
            parser.error("Benchmark output directory already has a configuration")
        (output/"host.json").write_text(json.dumps({"platform": platform.platform(), "python": sys.version,
            "processor": platform.processor(), "machine": platform.machine(), "cpu_count": os.cpu_count()}, indent=2), encoding="utf-8")
        results = []
        for configuration, workload, repetition, warmup in run_schedule(config):
            print(f"Running {configuration['id']} {workload} {'warmup' if warmup else 'measured'} {repetition}", flush=True)
            result = run_one(config, configuration, workload, repetition, warmup, output)
            results.append(result)
            (output/"results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            (output/"summary.json").write_text(json.dumps(aggregate_runs(results), indent=2), encoding="utf-8")
            if result["return_code"] != 0 or result["timed_out"]:
                print(f"Run failed: {result['run_id']}; inspect raw output", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
