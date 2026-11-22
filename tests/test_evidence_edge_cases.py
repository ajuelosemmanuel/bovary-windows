"""Configuration errors must never become paid escalation triggers."""
from dataclasses import replace
import copy
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bovary.contract import ExecutionError, Request, Result
from bovary.routing import Workbench
from bovary.evaluation import summarize
from bovary.provider import OpenAIAdapter, ProviderConfig
from bovary.validators import validate_specification


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def execute(self, request, **kwargs):
        self.calls.append(request)
        return Result(request.request_id, request.mode, text="answer")

    def unload(self):
        pass


class ValidatorSpecificationTests(unittest.TestCase):
    def test_valid_specs_include_nested_json_and_known_code_fixture(self):
        for specification in (None, {"kind": "exact", "expected": ""},
                              {"kind": "json_fields", "expected": {"record": {"amount": 1, "items": [True, None, 1.5]}}},
                              {"kind": "classification", "labels": ["yes", "no"], "expected": "yes"},
                              {"kind": "code_fixture", "fixture": "sum_two-v1"}):
            with self.subTest(specification=specification):
                self.assertIsNone(validate_specification(specification))

    def test_invalid_specs_are_configuration_errors(self):
        invalid = [[], "exact", {}, {"kind": []}, {"kind": "unknown"},
                   {"kind": "exact"}, {"kind": "exact", "expected": 1},
                   {"kind": "exact", "expected": "yes", "unexpected": True},
                   {"kind": "json_fields", "expected": {}},
                   {"kind": "json_fields", "expected": {"amount": 1}, "exact_fields": 1},
                   {"kind": "json_fields", "expected": {"record": {"amount": float("nan")}}},
                   {"kind": "json_fields", "expected": {"record": {1: "non-text-key"}}},
                   {"kind": "json_fields", "expected": {"record": (1, 2)}},
                   {"kind": "classification", "labels": [], "expected": "yes"},
                   {"kind": "classification", "labels": "yes", "expected": "yes"},
                   {"kind": "classification", "labels": ["yes", "yes"], "expected": "yes"},
                   {"kind": "classification", "labels": [None], "expected": "yes"},
                   {"kind": "classification", "labels": ["yes"], "expected": "no"},
                   {"kind": "code_fixture", "fixture": "unknown"},
                   {"kind": "code_fixture", "fixture": {}}]
        for specification in invalid:
            with self.subTest(specification=specification), self.assertRaises(ExecutionError) as raised:
                validate_specification(specification)
            self.assertEqual(raised.exception.code, "invalid_validator")

    def test_invalid_spec_never_runs_local_or_escalates(self):
        local, cloud = FakeAdapter(), FakeAdapter()
        workbench = Workbench(local, cloud)
        base = Request("Extract", task="extract", allowed_destinations=("local", "commercial"),
                       offline=False, private=False, allow_escalation=True,
                       validator={"kind": "classification", "labels": ["yes"], "expected": "no"})
        for mode in ("local", "commercial", "suggest", "auto"):
            with self.subTest(mode=mode):
                result = workbench.execute(replace(base, mode=mode))
                self.assertEqual(result.error["code"], "invalid_validator")
        self.assertEqual((local.calls, cloud.calls), ([], []))


class EvidenceAccountingTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {"schema_version": 1, "dataset_id": "edge-diagnostic-v1", "rubric_id": "exact-v1",
                        "synthetic": True, "representative": False, "cases": [
                            {"id": "case", "task": "format", "prompt": "Output answer", "validator": {"kind": "exact", "expected": "answer"}}]}
        attempt = Result("edge", "local", text="answer", usage={"input_tokens": 2, "output_tokens": 1},
                         timing={"end_to_end_s": .1, "ttft_s": .05},
                         validation={"status": "pass", "passed": True}, cost_status="unmeasured_local_compute").to_dict()
        result = Result("edge", "local", text="answer", cost_usd=0, attempts=[attempt]).to_dict()
        self.record = {"scenario": "local", "case_id": "case", "elapsed_s": .2, "result": result}

    def test_outer_zero_cannot_hide_unknown_attempt_or_validation_cost(self):
        summary = summarize(self.dataset, [self.record])["local"]
        self.assertIsNone(summary["total_cost_usd"])
        attempt = self.record["result"]["attempts"][0]
        attempt.update(cost_usd=.1, cost_status="measured")
        self.assertIsNone(summarize(self.dataset, [self.record])["local"]["total_cost_usd"])
        attempt["validation"]["cost_usd"] = .02
        self.assertAlmostEqual(summarize(self.dataset, [self.record])["local"]["total_cost_usd"], .12)

    def test_malformed_usage_cost_and_timing_fail_closed(self):
        for field, value in (("cost_usd", -1), ("cost_usd", float("nan")), ("usage", {"output_tokens": -1}),
                             ("usage", {"input_tokens": 1, "output_tokens": 2, "total_tokens": 1}),
                             ("usage", {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 2}}),
                             ("usage", {"output_tokens": True}), ("timing", {"ttft_s": float("inf")}),
                             ("timing", {"end_to_end_s": .1, "ttft_s": .2}), ("timing", {"warm": "yes"})):
            record = copy.deepcopy(self.record)
            record["result"]["attempts"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                summarize(self.dataset, [record])
        record = copy.deepcopy(self.record)
        record["result"]["timing"] = {"end_to_end_s": 1}
        with self.assertRaises(ValueError):
            summarize(self.dataset, [record])
        record = copy.deepcopy(self.record)
        record["result"]["attempts"][0]["timing"] = {"end_to_end_s": 1}
        with self.assertRaises(ValueError):
            summarize(self.dataset, [record])

    def test_actual_failed_execution_retains_attempt_while_preflight_does_not(self):
        class CrashAdapter(FakeAdapter):
            def execute(self, request, **kwargs):
                snapshot = Result(request.request_id, "local", status="failed", error={"code": "worker_crash"},
                                  usage={"input_tokens": None, "output_tokens": None},
                                  cost_status="unmeasured_local_compute").to_dict()
                raise ExecutionError("worker_crash", "Fixture worker started then crashed", attempt=snapshot)
        request = Request("Output answer", task="format")
        result = Workbench(CrashAdapter()).execute(request)
        self.assertEqual(len(result.attempts), 1)
        self.record["result"] = result.to_dict()
        summary = summarize(self.dataset, [self.record])["local"]
        self.assertEqual((summary["attempts"], summary["errors"]), (1, 1))
        self.assertIsNone(summary["usage"]["output_tokens"])
        preflight = Workbench().execute(request)
        self.assertEqual(preflight.attempts, [])

    def test_commercial_cancel_does_not_unload_another_local_request(self):
        class BusyLocal(FakeAdapter):
            unload_calls = 0
            def unload(self):
                self.unload_calls += 1
                raise ExecutionError("busy", "Another fixture request is active")
        class CancelCloud(FakeAdapter):
            def execute(self, request, **kwargs):
                raise ExecutionError("cancelled", "Cloud fixture cancelled")
        local = BusyLocal()
        result = Workbench(local, CancelCloud()).execute(Request("text", mode="commercial", offline=False,
                           private=False, allowed_destinations=("commercial",)))
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(local.unload_calls, 0)

    def test_busy_unload_after_own_validation_cannot_replace_cancel_result(self):
        class BusyLocal(FakeAdapter):
            def unload(self):
                raise ExecutionError("busy", "Another fixture request is now active")
        def cancelled_validation(*args, **kwargs):
            raise ExecutionError("cancelled", "Validation fixture cancelled")
        with patch("bovary.routing.validate", side_effect=cancelled_validation):
            result = Workbench(BusyLocal()).execute(Request("text"))
        self.assertEqual(result.error["code"], "cancelled")
        self.assertEqual(len(result.attempts), 1)
        self.assertTrue(any("cleanup deferred" in reason for reason in result.reasons))


class ProviderConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.config = ProviderConfig(enabled=True, account="fake", session_cap_usd=1,
                                     input_usd_per_million=1, output_usd_per_million=2, pricing_version="fixture")

    def request(self):
        return Request("text", mode="commercial", allowed_destinations=("commercial",), offline=False,
                       private=False, deadline=time.monotonic()+30, max_cost_usd=.1, identity={"data_class": "public"})

    def test_pending_reservation_cannot_use_stale_authorization_after_disable(self):
        started, release, b_waiting, allow_b = (threading.Event() for _ in range(4))
        class Guard:
            def __init__(self):
                self.lock = threading.Lock()
            def __enter__(self):
                if threading.current_thread().name == "fixture-B":
                    b_waiting.set()
                    if not allow_b.wait(10):
                        raise RuntimeError("Fixture synchronization timed out")
                self.lock.acquire()
                return self
            def __exit__(self, *_):
                self.lock.release()
        class Transport:
            calls = []
            def events(self, payload, **kwargs):
                self.calls.append(threading.current_thread().name)
                started.set()
                if not release.wait(10):
                    raise RuntimeError("Fixture synchronization timed out")
                yield {"type": "response.completed", "response": {"usage": {"input_tokens": 99999, "output_tokens": 1}}}
        transport = Transport()
        adapter = OpenAIAdapter(self.config, transport)
        adapter._lock = Guard()
        results = {}
        def execute(name):
            try:
                results[name] = adapter.execute(self.request()).status
            except ExecutionError as error:
                results[name] = error.code
        a = threading.Thread(target=execute, args=("A",), name="fixture-A")
        b = threading.Thread(target=execute, args=("B",), name="fixture-B")
        with patch.object(os, 'environ', {**os.environ, "OPENAI_API_KEY": "fake"}):
            try:
                a.start()
                self.assertTrue(started.wait(10))
                b.start()
                self.assertTrue(b_waiting.wait(10))
                release.set()
                a.join(10)
                self.assertFalse(a.is_alive())
                allow_b.set()
                b.join(10)
                self.assertFalse(b.is_alive())
            finally:
                release.set()
                allow_b.set()
                a.join(10)
                if b.ident is not None:
                    b.join(10)
        self.assertEqual(results, {"A": "needs_review", "B": "live_disabled"})
        self.assertEqual(transport.calls, ["fixture-A"])

    def test_total_streamed_output_is_bounded_and_reservation_retained(self):
        class Transport:
            def events(self, payload, **kwargs):
                for _ in range(20):
                    yield {"type": "response.output_text.delta", "delta": "x" * 4096}
        adapter = OpenAIAdapter(self.config, Transport())
        with patch.object(os, 'environ', {**os.environ, "OPENAI_API_KEY": "fake"}):
            result = adapter.execute(self.request())
            self.assertEqual(result.error["code"], "provider_output_limit")
            self.assertEqual(result.cost_status, "unknown_reserved")
            self.assertLessEqual(len(result.text), 128 * 256 + 4096)
            self.assertGreater(adapter.committed_usd, 0)
            with self.assertRaises(ExecutionError) as raised:
                adapter.execute(self.request())
            self.assertEqual(raised.exception.code, "live_disabled")


if __name__ == "__main__":
    unittest.main()
