import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark import SerialLock, aggregate_runs, parse_timings, summarize_samples, native_metrics_summary, analyze_stream_trace, run_schedule
from tools.windows_metrics import Sampler, process_instance_value
from tools.compare_logits import compare


class MeasurementTests(unittest.TestCase):
    def test_runner_preserves_existing_output_and_rejects_unbounded_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_file, output = root / "config.json", root / "output"
            output.mkdir()
            original = output / "configuration.json"
            original.write_text("prior configuration", encoding="utf-8")
            command = [sys.executable, "tools/benchmark.py", "--config", str(config_file),
                       "--output-dir", str(output)]
            config = {"lock_file": str(root / "serial.lock"), "configurations": []}
            for overrides, message in (
                ({"timeout_seconds": float("nan")}, "timeout_seconds must be a positive finite number"),
                ({"sample_interval_seconds": True}, "sample_interval_seconds must be a positive finite number"),
                ({}, "output directory must be empty"),
                ({"lock_file": str(original)}, "lock conflicts with an output artifact"),
                ({"lock_file": str(output / "host.json")}, "lock conflicts with an output artifact"),
                ({"lock_file": str(output / "A-short-rep1" / "serial.lock"),
                  "configurations": [{"id": "A"}]}, "lock conflicts with an output artifact"),
            ):
                with self.subTest(overrides=overrides):
                    config_file.write_text(json.dumps({**config, **overrides}), encoding="utf-8")
                    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(message, result.stderr)
                    self.assertEqual(original.read_text(encoding="utf-8"), "prior configuration")
                    self.assertEqual(list(output.iterdir()), [original])

    def test_interleaved_subprocess_order_and_default_order(self):
        configs = [{"id": name, "command": [sys.executable, "-c", "print('measurement order fixture')"]} for name in ("A", "B")]
        config = {"configurations": configs, "workloads": ["short"], "warmups": 0, "repetitions": 2,
            "gpu_telemetry": False, "pdh": False, "sample_interval_seconds": .01, "interleave_repetitions": True}
        default = {**config, "interleave_repetitions": False}
        self.assertEqual([(c["id"], n) for c, _, n, _ in run_schedule(default)], [("A", 1), ("A", 2), ("B", 1), ("B", 2)])
        warmup_config = {**config, "warmups": 1}
        self.assertEqual([(c["id"], n, warmup) for c, _, n, warmup in run_schedule(warmup_config)],
            [("A", 1, True), ("B", 1, True), ("A", 1, False), ("B", 1, False), ("A", 2, False), ("B", 2, False)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config["lock_file"] = str(root/"output"/"locks"/"serial.lock")
            (root/"config.json").write_text(json.dumps(config), encoding="utf-8")
            result = subprocess.run([sys.executable, "tools/benchmark.py", "--config", str(root/"config.json"),
                "--output-dir", str(root/"output")], cwd=Path(__file__).resolve().parents[1], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            runs = json.loads((root/"output"/"results.json").read_text())
            self.assertEqual([(r["configuration"], r["repetition"]) for r in runs], [("A", 1), ("B", 1), ("A", 2), ("B", 2)])
    def test_stream_trace_clock_window_and_eog_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"stream-layers.csv"
            path.write_text("clock_ns,layer,selectedids,hits,misses,requested_bytes,io_bytes,read_calls,read_ns,upload_wait_ns\n1000000000000,0,8,0,8,100,120,1,1000000,2000000\n1000200000000,0,8,2,6,200,240,2,2000000,4000000\n1000300000000,1,8,4,4,300,360,3,3000000,6000000\n", encoding="utf-8")
            native = {"clock_start_ns": 1000000000000, "token_elapsed_s": [.1, .2],
                "total_s": .4, "actual_generated_tokens": 2, "decode_step_s": [.1,.1]}
            analysis, metrics = analyze_stream_trace(path, native)
            self.assertEqual(analysis["decode_rows"], 2)
            self.assertEqual(metrics["stream_trace_total_io_bytes"], 720)
            self.assertEqual(metrics["stream_trace_decode_io_bytes"], 600)
            self.assertEqual(metrics["decode_io_bytes_per_generated_token_after_first"], 600)
            self.assertEqual(metrics["decode_io_bytes_per_evaluation"], 300)
            self.assertAlmostEqual(metrics["decode_read_fraction_of_read_plus_upload_wait"], 1/3)
            self.assertAlmostEqual(metrics["decode_expert_cache_hit_rate_percent"], 37.5)
            footer = {"stream_upload_payload_bytes": 600, "stream_read_api_bytes": 720,
                "stream_read_calls": 6, "stream_read_nanoseconds": 6000000, "stream_upload_wait_nanoseconds": 12000000}
            verified, _ = analyze_stream_trace(path, native, footer)
            self.assertTrue(verified["footer_validation"]["passed"])
            footer["stream_pool_mode"] = "pinned2"
            pinned, pinned_metrics = analyze_stream_trace(path, native, footer, pool_mode="pinned2")
            self.assertTrue(pinned["valid"])
            self.assertIsNone(pinned["gpu_transfer_elapsed_seconds"])
            self.assertNotIn("decode_upload_payload_bytes_per_upload_wait_second", pinned_metrics)
            wrong_pool, wrong_metrics = analyze_stream_trace(path, native, footer, pool_mode="serial")
            self.assertFalse(wrong_pool["valid"])
            self.assertEqual(wrong_metrics, {})
            footer["stream_read_api_bytes"] += 1
            invalid, unavailable = analyze_stream_trace(path, native, footer)
            self.assertFalse(invalid["valid"])
            self.assertEqual(unavailable, {})
    def test_pdh_virtual_bytes_join_uses_pid_and_full_instance(self):
        self.assertEqual(process_instance_value(42, {"engine": 40, "engine#1": 42},
            {"engine": 1000, "engine#1": 9000}), 9000)
        self.assertIsNone(process_instance_value(43, {"engine": 42}, {"engine": 1000}))
        self.assertIsNone(process_instance_value(42, {"engine": 42, "engine#1": 42}, {"engine": 1000, "engine#1": 9000}))
        self.assertIsNone(process_instance_value(42, {"engine": 42}, {"other": 1000}))
    def test_engine_counts_are_not_requested_counts(self):
        metrics = parse_timings("""
llama_perf_context_print: load time = 200.12 ms
llama_perf_context_print: prompt eval time = 90.00 ms / 42 tokens (2.14 ms per token, 466.67 tokens per second)
llama_perf_context_print: eval time = 900.00 ms / 9 runs (100.00 ms per token, 10.00 tokens per second)
llama_perf_context_print: total time = 1090.00 ms / 51 tokens
""")
        self.assertEqual(metrics["engine_prompt_tokens"], 42)
        self.assertEqual(metrics["engine_decode_evaluations"], 9)
        self.assertEqual(metrics["engine_decode_tokens_per_second"], 10)
        self.assertNotIn("engine_generated_tokens", metrics)

    def test_stream_counters_keep_payload_and_read_alignment_distinct(self):
        result = parse_timings("bovary_stream: read_calls=2 requested_bytes=1000 io_bytes=8192 read_ns=2000000 host_buffer_bytes=8388608 pinned_bytes=0\nllama_moe_cache: ubatch <= 8: hits = 7, misses = 3 (0 on the CPU), hit rate = 70.00%, uploaded = 0.01 MiB")
        self.assertEqual(result["stream_upload_payload_bytes"], 1000)
        self.assertEqual(result["stream_read_api_bytes"], 8192)
        self.assertEqual(result["stream_mean_read_call_seconds"], .001)
        self.assertEqual(result["expert_cache_hits"], 7)
        pinned = parse_timings("bovary_stream: read_calls=2 requested_bytes=1000 io_bytes=8192 read_ns=2000000 upload_wait_ns=10 pool=pinned2 host_buffer_bytes=16777216 pinned_bytes=16777216")
        self.assertEqual(pinned["stream_pool_mode"], "pinned2")
        self.assertEqual(pinned["stream_host_buffer_bytes"], 16777216)
        self.assertEqual(pinned["stream_explicit_pinned_bytes"], 16777216)

    def test_sparse_samples_preserve_unknown(self):
        result = summarize_samples([{"elapsed_seconds": 0, "x": None, "y": 4},
                                    {"elapsed_seconds": 1, "x": None, "y": 8}])
        self.assertNotIn("x", result)
        self.assertEqual(result["y"]["median"], 6)
        self.assertEqual(result["y"]["second_half_median"], 8)

    def test_native_latencies_and_actual_count(self):
        result = native_metrics_summary({"actual_generated_tokens": 4,
            "decode_step_s": [.1, .2, .3], "token_elapsed_s": [2, 2.1, 2.3, 2.6]})
        self.assertAlmostEqual(result["native_decode_tokens_per_second"], 5)
        self.assertAlmostEqual(result["native_decode_step_p95_seconds"], .29)
        self.assertEqual(result["native_actual_generated_tokens"], 4)
        self.assertAlmostEqual(result["native_intertoken_max_seconds"], .3)

    def test_logit_comparison_detects_perturbation_and_eog_row(self):
        from array import array
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            for directory in (a, b):
                directory.mkdir()
                (directory/"prompt.txt").write_text("same prompt", encoding="utf-8")
                (directory/"native-metrics.json").write_text(json.dumps({"context": 4096, "vocab_size": 3,
                    "prompt_tokens": 10, "requested_generated_tokens": 2, "actual_generated_tokens": 1,
                    "early_stop": True, "tokens": [1]}), encoding="utf-8")
                (directory/"logits.f32").write_bytes(array("f", [1, 3, 2, 3, 2, 1]).tobytes())
            same = compare(a, b)
            self.assertTrue(same["passed"])
            self.assertEqual(same["byte_exact_rows"], 2)
            (b/"logits.f32").write_bytes(array("f", [1, 3.01, 2, 3, 2, 1]).tobytes())
            altered = compare(a, b)
            self.assertFalse(altered["passed"])
            self.assertEqual(altered["tolerance_violations"], 1)

    def test_failures_and_warmups_excluded(self):
        def row(seconds, warmup=False, code=0):
            return {"configuration": "a", "workload": "short", "warmup": warmup,
                "return_code": code, "timed_out": False, "end_to_end_seconds": seconds,
                "engine_metrics": {}, "memory_summary": {}}
        summary = aggregate_runs([row(100, True), row(20, code=1), row(1), row(3), row(2)])
        self.assertEqual(summary[0]["measured_repetitions"], 3)
        self.assertEqual(summary[0]["metrics"]["end_to_end_seconds"]["median"], 2)
        self.assertEqual(summary[0]["metrics"]["end_to_end_seconds"]["spread"], 2)

    def test_lock_blocks_second_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"bench.lock"
            with SerialLock(path):
                script = "from tools.benchmark import SerialLock; import sys;\nwith SerialLock(sys.argv[1]): pass"
                result = subprocess.run([sys.executable, "-c", script, str(path)],
                    cwd=Path(__file__).resolve().parents[1], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
            with SerialLock(path):
                pass
            self.assertEqual(path.stat().st_size, 0)

    def test_end_to_end_native_metric_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root/"child.py"
            child.write_text("""import json,sys,time,os
from pathlib import Path
allocation=bytearray(8*1024*1024)
print('hello', flush=True)
time.sleep(.4)
json.dump({'actual_generated_tokens':2,'requested_generated_tokens':2,'early_stop':False,'context_ready_s':.01,'token_elapsed_s':[.02,.12],'decode_step_s':[.1],'clock_start_ns':1000000000000,'total_s':.4},open(sys.argv[1],'w'))
Path(os.environ['BOVARY_STREAM_TRACE']).write_text('clock_ns,layer,selectedids,hits,misses,requested_bytes,io_bytes,read_calls,read_ns,upload_wait_ns\\n1000100000000,0,8,0,8,4000,4096,1,1000,2000\\n')
print('bovary_stream: read_calls=1 requested_bytes=4000 io_bytes=4096 read_ns=1000 upload_wait_ns=2000 host_buffer_bytes=8388608 pinned_bytes=0',file=sys.stderr)
""", encoding="utf-8")
            config = {"workloads": ["short"], "warmups": 0, "repetitions": 1, "n_predict": 2,
                "pdh": False, "gpu_telemetry": False, "sample_interval_seconds": .03, "lock_file": str(root/"serial.lock"),
                "configurations": [{"id": "smoke", "stdout_is_generation": True,
                    "env": {"BOVARY_STREAM_TRACE": "{stream_trace_file}"},
                    "command": [sys.executable, str(child), "{metrics_file}"]}]}
            (root/"config.json").write_text(json.dumps(config), encoding="utf-8")
            result = subprocess.run([sys.executable, "tools/benchmark.py", "--config", str(root/"config.json"),
                "--output-dir", str(root/"output")], cwd=Path(__file__).resolve().parents[1], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            measured = json.loads((root/"output"/"results.json").read_text())[0]
            self.assertEqual(measured["actual_generated_tokens"], 2)
            self.assertFalse(measured["early_stopping"])
            self.assertEqual(measured["engine_metrics"]["native_decode_tokens_per_second"], 10)
            self.assertEqual(measured["engine_metrics"]["decode_io_bytes_per_generated_token_after_first"], 4096)
            self.assertTrue(measured["stream_trace_analysis"]["footer_validation"]["passed"])
            self.assertNotIn("{stream_trace_file}", measured["environment_overrides"]["BOVARY_STREAM_TRACE"])
            self.assertIsNotNone(measured["launch_to_first_generation_stdout_seconds"])
            self.assertTrue((root/"output"/"smoke-short-rep1"/"samples.csv").exists())

    @unittest.skipUnless(os.name == "nt", "Windows-native counters")
    def test_native_self_process_memory(self):
        sampler = Sampler(os.getpid(), enable_pdh=False)
        try:
            sample = sampler.sample()
            self.assertGreater(sample["process_working_set_bytes"], 0)
            self.assertGreater(sample["process_private_commit_bytes"], 0)
            self.assertGreater(sample["system_total_physical_bytes"], sample["system_available_physical_bytes"])
            self.assertGreater(sample["system_commit_limit_bytes"], sample["system_commit_bytes"])
        finally:
            sampler.close()


if __name__ == "__main__": unittest.main()
