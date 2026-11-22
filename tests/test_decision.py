import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bovary.policy import decide, wilson_lower

ROOT = Path(__file__).resolve().parents[1]


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / "benchmarks/decision_policy.json").read_text())

    def evidence(self, name, complexity="low", cases=100, correct=100, **overrides):
        model = self.policy["models"][name]
        model["evaluations"].append({
            "workload": "default", "complexity": complexity,
            "model": model["model"], "configuration": model["configuration"],
            "dataset_id": "held-out-set-v1-and-rubric-v1", "source": "fixture.json",
            "representative": True, "cases": cases, "correct": correct,
            "p95_latency_s": 5, **overrides,
        })

    def test_complexity_defaults_and_explicit_override(self):
        local = decide(self.policy, task="extract")
        self.assertEqual((local["recommendation"], local["status"]), ("local", "provisional"))
        self.assertEqual(decide(self.policy, task="code", allow_cloud=True)["recommendation"], "commercial")
        self.assertEqual(decide(self.policy, task="extract", multistep=True, allow_cloud=True)["complexity"], "high")
        self.assertEqual(decide(self.policy, task="code", complexity="low")["recommendation"], "local")

    def test_cloud_constraints_never_fall_back_to_unqualified_local(self):
        for flags in ({}, {"allow_cloud": True, "offline": True}):
            result = decide(self.policy, task="reason", **flags)
            self.assertEqual(result["status"], "blocked")
            self.assertIsNone(result["recommendation"])

    def test_failed_local_validation_escalates_or_blocks(self):
        self.assertEqual(decide(self.policy, task="extract", validation_failed=True,
                                allow_cloud=True)["recommendation"], "commercial")
        self.assertIsNone(decide(self.policy, task="extract", validation_failed=True)["recommendation"])

    def test_quality_evidence_can_expand_local_complexity(self):
        self.evidence("local", complexity="high")
        result = decide(self.policy, task="reason", input_tokens=200, allow_cloud=True)
        self.assertEqual((result["recommendation"], result["status"]), ("local", "supported"))
        self.assertEqual(decide(self.policy, task="reason")["status"], "provisional")

    def test_ten_question_probe_does_not_qualify_high_complexity(self):
        self.evidence("local", complexity="high", cases=10, correct=10)
        result = decide(self.policy, task="reason", allow_cloud=True)
        self.assertEqual(result["recommendation"], "commercial")
        self.assertEqual(result["candidates"]["local"]["quality"], "unknown")
        self.assertAlmostEqual(wilson_lower(10, 10), .7224672, places=6)

    def test_mismatched_or_unrepresentative_evidence_is_not_reused(self):
        for override in ({"model": "different"}, {"configuration": "mixed"},
                         {"workload": "other"}, {"representative": False}):
            self.policy["models"]["local"]["evaluations"] = []
            self.evidence("local", complexity="high", **override)
            self.assertEqual(decide(self.policy, task="reason", allow_cloud=True)["recommendation"], "commercial")

    def test_observed_quality_failure_excludes_even_simple_local(self):
        self.evidence("local", correct=50)
        result = decide(self.policy, task="format", allow_cloud=True)
        self.assertEqual(result["recommendation"], "commercial")
        self.assertEqual(result["candidates"]["local"]["quality"], "fail")

    def test_context_overflow_escalates(self):
        result = decide(self.policy, task="extract", input_tokens=4000, output_tokens=128, allow_cloud=True)
        self.assertEqual(result["recommendation"], "commercial")
        self.assertIsNone(decide(self.policy, task="extract", input_tokens=2000000, allow_cloud=True)["recommendation"])

    def test_budget_and_latency_require_evidence(self):
        result = decide(self.policy, task="extract", max_monthly_cost_usd=20)
        self.assertEqual(result["status"], "benchmark_required")
        self.assertIsNone(result["candidates"]["local"]["estimated_monthly_cost_usd"])
        self.assertEqual(decide(self.policy, task="extract", max_latency_s=20)["status"], "benchmark_required")
        self.evidence("local")
        self.assertEqual(decide(self.policy, task="extract", max_latency_s=2)["status"], "blocked")

    def test_compare_total_cost_only_after_matched_quality_passes(self):
        for name, variable, fixed in (("local", .001, 20), ("commercial", .01, 0)):
            self.evidence(name)
            self.policy["models"][name].update(cost_per_request_usd=variable, monthly_fixed_cost_usd=fixed)
        low_volume = decide(self.policy, task="extract", input_tokens=100, allow_cloud=True, monthly_requests=1000)
        self.assertEqual(low_volume["recommendation"], "commercial")
        self.assertEqual(low_volume["candidates"]["local"]["estimated_monthly_cost_usd"], 21)
        self.assertEqual(decide(self.policy, task="extract", input_tokens=100, allow_cloud=True,
                                monthly_requests=10000)["recommendation"], "local")
        self.policy["models"]["commercial"]["evaluations"][0]["dataset_id"] = "different-set"
        self.assertEqual(decide(self.policy, task="extract", allow_cloud=True)["status"], "benchmark_required")

    def test_invalid_values_and_duplicates_fail_closed(self):
        for value in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(ValueError):
                decide(self.policy, max_monthly_cost_usd=value)
        self.evidence("local")
        self.policy["models"]["local"]["evaluations"] *= 2
        with self.assertRaises(ValueError):
            decide(self.policy)

    def test_model_order_does_not_change_preference(self):
        reordered = copy.deepcopy(self.policy)
        reordered["models"] = dict(reversed(list(reordered["models"].items())))
        self.assertEqual(decide(reordered, task="extract", allow_cloud=True)["recommendation"], "local")

    def test_cli_audit_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            command = [sys.executable, "-m", "bovary.policy", "--task", "extract", "--output", str(path)]
            first = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "python")})
            self.assertEqual(first.returncode, 0, first.stderr)
            result = json.loads(path.read_text())
            self.assertFalse(result["executes_inference"])
            self.assertEqual(len(result["policy_sha256"]), 64)
            second = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "python")})
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(json.loads(path.read_text()), result)


if __name__ == "__main__":
    unittest.main()
