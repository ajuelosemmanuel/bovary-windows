import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bovary.contract import ExecutionError, Request, Result
from bovary.evaluation import digest, run_evaluation, summarize
from bovary.evidence import import_evidence, verify_qualification
from bovary.routing import Workbench
from bovary.provider import OpenAIAdapter, ProviderConfig
from bovary.validators import validate


class FakeAdapter:
    def __init__(self, text="yes", status="complete"):
        self.text, self.status, self.calls, self.unloaded = text, status, [], False
    def execute(self, request, on_event=None, cancel=None):
        self.calls.append(request)
        return Result(request.request_id, request.mode, text=self.text, status=self.status,
                      usage={"output_tokens": 1}, timing={"cold": len(self.calls) == 1})
    def unload(self):
        self.unloaded = True


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.local, self.cloud = FakeAdapter("bad"), FakeAdapter("yes")
        self.workbench = Workbench(self.local, self.cloud)
        self.request = Request("Classify", task="classify", validator={"kind": "classification", "labels": ["yes", "no"], "expected": "yes"})

    def test_suggest_never_executes_and_auto_requires_more_than_policy_pass(self):
        result = self.workbench.execute(replace(self.request, mode="suggest"))
        self.assertEqual(result.status, "suggested")
        result = self.workbench.execute(replace(self.request, mode="auto"))
        self.assertEqual(result.error["code"], "auto_disabled")
        self.assertEqual((self.local.calls, self.cloud.calls), ([], []))

    def test_truncated_answer_is_not_validated_or_escalated(self):
        self.local.text, self.local.status = 'yes', 'truncated'
        request = replace(self.request, output_tokens=None, deadline=time.monotonic()+30)
        result = self.workbench.execute(request)
        self.assertEqual(result.status, 'truncated')
        self.assertEqual(result.validation, {'status':'not_checked', 'passed':None})
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(self.cloud.calls, [])

    def test_interrupted_local_answer_keeps_partial_text_without_accepting_it(self):
        for code in ('deadline', 'cancelled'):
            with self.subTest(code=code):
                partial = Result(self.request.request_id, 'local', text='A useful partial answer', status='failed')
                with patch.object(self.local, 'execute', side_effect=ExecutionError(code, 'interrupted', attempt=partial.to_dict())):
                    result = self.workbench.execute(self.request)
                self.assertEqual(result.text, partial.text)
                self.assertEqual(result.status, 'cancelled' if code == 'cancelled' else 'failed')
                self.assertEqual(result.validation.get('passed'), None)
                self.assertEqual(result.attempts[0]['validation']['status'], 'interrupted')
                self.assertEqual(self.cloud.calls, [])

    def test_evaluation_does_not_grade_or_accept_truncated_answers(self):
        self.local.text, self.local.status = 'yes', 'truncated'
        dataset = {'schema_version':1, 'dataset_id':'truncation-fixture', 'rubric_id':'exact-v1',
                   'synthetic':True, 'representative':False, 'cases':[{'id':'one', 'task':'classify',
                   'prompt':'Classify', 'validator':self.request.validator}]}
        with tempfile.TemporaryDirectory() as directory, patch('bovary.evaluation.validate') as grader:
            artifact = run_evaluation(self.workbench, dataset, self.request, Path(directory)/'evaluation', scenarios=('local',), source_kind='fake')
            grader.assert_not_called()
        self.assertEqual(artifact['summary']['local']['correct'], 0)
        self.assertEqual(artifact['summary']['local']['false_acceptances'], 0)
        self.assertEqual(artifact['summary']['local']['errors'], 1)

    def test_caller_claimed_qualification_cannot_enable_auto(self):
        self.workbench.qualification = "forged-summary.json"
        self.workbench.trusted_evidence_sha256 = ("forged-passing-digest",)
        request = replace(self.request, mode="auto", identity={"model_sha256": "claimed", "representative": True, "correct": 100})
        result = self.workbench.execute(request)
        self.assertEqual(result.error["code"], "auto_disabled")
        self.assertEqual((self.local.calls, self.cloud.calls), ([], []))

    def test_offline_and_private_dominate_cloud_flags(self):
        for private, offline in ((True, False), (False, True)):
            result = self.workbench.execute(replace(self.request, mode="commercial", allowed_destinations=("local", "commercial"), private=private, offline=offline))
            self.assertEqual(result.error["code"], "privacy")
        self.assertFalse(self.cloud.calls)

    def test_failed_local_validator_without_escalation_needs_review(self):
        result = self.workbench.execute(self.request)
        self.assertEqual(result.status, "needs_review")
        self.assertEqual(result.attempts[0]["status"], "needs_review")
        self.assertFalse(self.cloud.calls)

    def test_escalation_is_once_and_rechecks_privacy(self):
        result = self.workbench.execute(replace(self.request, allow_escalation=True))
        self.assertEqual(result.status, "needs_review")
        self.assertFalse(self.cloud.calls)
        self.cloud.text = "no"
        result = self.workbench.execute(replace(self.request, allow_escalation=True, private=False, offline=False,
                                                allowed_destinations=("local", "commercial")))
        self.assertEqual(result.status, "needs_review")
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(len(self.cloud.calls), 1)
        self.assertIsNone(result.cost_usd)
        self.assertEqual(result.cost_status, "unknown_total")

    def test_no_gpu_switch_from_cpu_permission(self):
        result = self.workbench.execute(replace(self.request, local_profile="balanced"))
        self.assertEqual(result.error["code"], "gpu_forbidden")
        self.assertFalse(self.local.calls)

    def test_escalation_rechecks_budget_before_any_provider_transport(self):
        class ZeroCallTransport:
            calls = 0
            def events(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("Provider must remain unopened")
        transport = ZeroCallTransport()
        self.workbench.commercial = OpenAIAdapter(ProviderConfig(enabled=True, account="fake",
            session_cap_usd=1, input_usd_per_million=1, output_usd_per_million=2,
            pricing_version="fixture-only"), transport)
        request = replace(self.request, allow_escalation=True, private=False, offline=False,
                          allowed_destinations=("local", "commercial"), deadline=time.monotonic()+20,
                          identity={"data_class": "public"}, max_cost_usd=0)
        with patch.object(os, 'environ', {**os.environ, "OPENAI_API_KEY": "fake"}):
            result = self.workbench.execute(request)
        self.assertEqual(result.error["code"], "budget")
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.attempts[0]["validation"]["status"], "fail")
        self.assertEqual(transport.calls, 0)

    def test_deadline_expired_after_local_attempt_blocks_escalation(self):
        request = replace(self.request, allow_escalation=True, private=False, offline=False,
                          allowed_destinations=("local", "commercial"), deadline=time.monotonic()+20)
        original = self.local.execute
        def expire(local_request, **kwargs):
            result = original(local_request, **kwargs)
            request.deadline = time.monotonic()-1
            return result
        self.local.execute = expire
        result = self.workbench.execute(request)
        self.assertEqual(result.error["code"], "deadline")
        self.assertEqual(len(result.attempts), 1)
        self.assertFalse(self.cloud.calls)

    def test_expired_deadline_after_validator_cannot_return_complete(self):
        self.local.text = "yes"
        request = replace(self.request, deadline=time.monotonic()+20)
        def slow_validator(*args, **kwargs):
            request.deadline = time.monotonic()-1
            return {"status": "pass", "passed": True}
        with patch("bovary.routing.validate", side_effect=slow_validator):
            result = self.workbench.execute(request)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error["code"], "deadline")
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.attempts[0]["text"], "yes")
        self.assertEqual(result.attempts[0]["validation"]["status"], "interrupted")
        self.assertTrue(self.local.unloaded)

    def test_code_validator_subprocess_receives_remaining_deadline(self):
        from types import SimpleNamespace
        with patch("bovary.validators.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="[5,3,0,-1]")) as run:
            result = validate("def solve(a,b): return a+b", {"kind": "code_fixture", "fixture": "sum_two-v1"}, deadline=time.monotonic()+.5)
        self.assertTrue(result["passed"])
        self.assertGreater(run.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(run.call_args.kwargs["timeout"], .5)

    def test_cancelled_after_validator_cannot_return_complete(self):
        self.local.text = "yes"
        cancel = threading.Event()
        def cancelling_validator(*args, **kwargs):
            cancel.set()
            return {"status": "pass", "passed": True}
        with patch("bovary.routing.validate", side_effect=cancelling_validator):
            result = self.workbench.execute(self.request, cancel=cancel)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.error["code"], "cancelled")
        self.assertEqual(len(result.attempts), 1)
        self.assertTrue(self.local.unloaded)

    def test_unavailable_manual_backend_no_fallback(self):
        self.workbench.local = None
        result = self.workbench.execute(self.request)
        self.assertEqual(result.error["code"], "backend_unavailable")
        self.assertFalse(self.cloud.calls)

    def test_task_specific_validation_checks_semantics_and_code_is_controlled(self):
        self.assertFalse(validate('{"amount":1}', {"kind": "json_fields", "expected": {"amount": 2}})["passed"])
        self.assertFalse(validate('{"amount":true}', {"kind": "json_fields", "expected": {"amount": 1}})["passed"])
        self.assertTrue(validate('def solve(a,b):\n return a+b', {"kind": "code_fixture", "fixture": "sum_two-v1"})["passed"])
        for text in ('import os\ndef solve(a,b): return a+b', 'def solve(a,b): return __import__("os").system("echo bad")',
                     'def solve(a,b): return a**999999', 'def solve(a,b): return open("private")', 'def solve(a,b): return'):
            self.assertFalse(validate(text, {"kind": "code_fixture", "fixture": "sum_two-v1"})["passed"])

    def test_nested_json_semantics_require_recursive_exact_types(self):
        specification = {"kind": "json_fields", "expected": {"record": {"amount": 1, "items": [True, {"count": 0}]}}}
        self.assertTrue(validate('{"record":{"amount":1,"items":[true,{"count":0}]}}', specification)["passed"])
        for value in (
            {"record": {"amount": True, "items": [True, {"count": 0}]}},
            {"record": {"amount": 1.0, "items": [True, {"count": 0}]}},
            {"record": {"amount": 1, "items": [1, {"count": 0}]}},
            {"record": {"amount": 1, "items": [True, {"count": False}]}},
            {"record": {"amount": 1, "items": [True, {"count": 0, "extra": 1}]}},
            {"record": {"amount": 1, "items": [True]}}
        ):
            with self.subTest(value=value):
                self.assertFalse(validate(json.dumps(value), specification)["passed"])

    def test_preflight_blocked_cases_count_zero_backend_attempts(self):
        dataset = {"schema_version": 1, "dataset_id": "synthetic-v1", "rubric_id": "v1", "synthetic": True,
                   "representative": False, "cases": [{"id": "one", "task": "format", "prompt": "Output yes",
                   "validator": {"kind": "exact", "expected": "yes"}}]}
        blocked = self.workbench.execute(replace(self.request, mode="commercial"))
        self.assertEqual(blocked.attempts, [])
        summary = summarize(dataset, [{"scenario": "commercial", "case_id": "one", "elapsed_s": .01, "result": blocked.to_dict()}])["commercial"]
        self.assertEqual((summary["cases"], summary["errors"], summary["attempts"]), (1, 1, 0))
        self.assertIsNone(summary["total_cost_usd"])
        self.assertIsNone(summary["cost_per_successful_task_usd"])
        executed = self.workbench.execute(self.request)
        summary = summarize(dataset, [{"scenario": "local", "case_id": "one", "elapsed_s": .01, "result": executed.to_dict()}])["local"]
        self.assertEqual(summary["attempts"], 1)

    def test_retained_evaluation_regrades_and_rejects_summary_tampering(self):
        dataset = {"schema_version": 1, "dataset_id": "synthetic-v1", "rubric_id": "v1", "synthetic": True,
                   "representative": False, "cases": [{"id": "one", "task": "classify", "prompt": "Classify yes",
                   "validator": {"kind": "classification", "labels": ["yes", "no"], "expected": "yes"}}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new"
            artifact = run_evaluation(self.workbench, dataset, self.request, path,
                                      scenarios=("local", "commercial", "routing", "escalation"), source_kind="fake")
            self.assertEqual(artifact["summary"]["local"]["correct"], 0)
            self.assertEqual(artifact["summary"]["commercial"]["correct"], 0)  # privacy blocked, still counted
            self.assertIsNone(artifact["summary"]["local"]["cost_per_successful_task_usd"])
            self.assertEqual(len(artifact["records"]), 4)
            retained = path / "evaluation.json"
            verified, sha = import_evidence(retained)
            self.assertEqual(verified, artifact)
            with self.assertRaises(Exception):
                verify_qualification(retained, (sha,), self.request, "local", self.workbench.policy)
            artifact["summary"]["local"]["correct"] = 100
            retained.write_text(json.dumps(artifact))
            with self.assertRaises(ValueError):
                import_evidence(retained)
            artifact["records"] = []
            artifact["summary"] = {}
            retained.write_text(json.dumps(artifact))
            with self.assertRaises(ValueError):
                import_evidence(retained)
            self.assertTrue(self.local.unloaded)

    def test_evidence_checks_stale_prompt_dataset_and_validator(self):
        dataset = {"schema_version": 1, "dataset_id": "synthetic-v1", "rubric_id": "v1", "synthetic": True,
                   "representative": False, "cases": [{"id": "one", "task": "format", "prompt": "Output yes",
                   "validator": {"kind": "exact", "expected": "yes"}}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new"
            original = run_evaluation(self.workbench, dataset, self.request, path, scenarios=("local",), source_kind="fake")
            retained = path / "evaluation.json"
            for change in ("prompt", "dataset", "validator"):
                artifact = copy.deepcopy(original)
                if change == "prompt":
                    artifact["records"][0]["prompt_sha256"] = "tampered"
                elif change == "dataset":
                    artifact["dataset"]["cases"][0]["prompt"] = "changed"
                else:
                    artifact["validator_version"] = "old"
                retained.write_text(json.dumps(artifact))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    import_evidence(retained)


if __name__ == "__main__":
    unittest.main()
