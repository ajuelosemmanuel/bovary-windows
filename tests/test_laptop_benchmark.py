import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bovary.contract import ExecutionError
from bovary import laptop_benchmark as benchmark


class LaptopBenchmarkTests(unittest.TestCase):
    def test_loaded_summary_uses_actual_worker_reuse_and_redacts_text(self):
        worker = Mock(loaded=True)
        worker.__enter__ = Mock(return_value=worker)
        worker.__exit__ = Mock()
        worker.execute.side_effect = [
            ExecutionError("model_unavailable", "C:/private/model.gguf"),
            Mock(timing={"warm": False, "ttft_s": 7.0, "end_to_end_s": 8.0},
                 text="Thursday.", status="complete", usage={"input_tokens": 67, "output_tokens": 2}),
            Mock(timing={"warm": True, "ttft_s": 2.0, "end_to_end_s": 3.0},
                 text="secret reply", status="complete", usage={"input_tokens": 158, "output_tokens": 4}),
        ]
        with patch.object(benchmark, "MemorySampler", autospec=True) as sampler:
            sampler.return_value.__enter__.return_value = Mock(hardware={}, metrics={})
            report = benchmark.run_bovary("cpu-only", worker_factory=lambda: worker)
        self.assertEqual(report["correct"], 1)
        self.assertIsNone(report["request_first_text_s"])
        self.assertEqual(report["loaded_median_first_text_s"], 2)
        self.assertEqual(report["cases"][1]["state"], "startup")
        serialized = json.dumps(report)
        for hidden in ("secret reply", "C:/private", "Thursday", benchmark.CASES[0][1]):
            self.assertNotIn(hidden, serialized)
        self.assertEqual([call.args[0].allow_gpu for call in worker.execute.call_args_list], [False] * 3)

    def test_incomplete_candidate_is_not_scored_correct(self):
        result = benchmark.row(benchmark.CASES[0], {"ttft_s": 1}, "45", state="startup",
                               failure="incomplete")
        self.assertFalse(result["correct"])
        self.assertEqual(result["ttft_s"], 1)


if __name__ == "__main__":
    unittest.main()
