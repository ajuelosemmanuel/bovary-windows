import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bovary import cli
from bovary.contract import ExecutionError, Result
from bovary.mini import MiniAgent, replace_expression


class Worker:
    def __init__(self):
        self.opens = self.closes = 0
        self.requests = []
        self.status = "complete"

    def __enter__(self):
        self.opens += 1
        return self

    def __exit__(self, *exc):
        self.closes += 1

    def execute(self, request, on_event=None):
        self.requests.append(request)
        if self.status == "failed":
            raise ExecutionError("deadline", "Deadline expired")
        return Result(request.request_id, "local", status=self.status, text="answer", usage={"output_tokens": 1})


class MiniTests(unittest.TestCase):
    def setUp(self):
        self.worker = Worker()
        self.agent = MiniAgent(worker_factory=lambda: self.worker)

    def test_exact_hits_skip_model_and_source_changes_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.txt"
            path.write_text("v1", encoding="utf-8")
            first = self.agent.run("explain", source=path)
            emitted = []
            hit = self.agent.run("explain", source=path, on_text=emitted.append)
            self.assertEqual(emitted, [first["text"]])
            self.assertEqual((hit["model_calls"], hit["generated_tokens"]), (0, 0))
            path.write_text("v2", encoding="utf-8")
            self.assertFalse(self.agent.run("explain", source=path)["cache_hit"])
            self.assertIn("v2", self.worker.requests[-1].prompt)
            self.assertFalse(self.agent.run("explain", source=path, tokens=24)["cache_hit"])
        self.assertEqual(self.worker.opens, self.worker.closes)
        self.assertTrue(all(r.private and r.offline and not r.allow_gpu for r in self.worker.requests))

    def test_failure_and_truncation_release_the_worker_and_do_not_cache(self):
        self.worker.status = "failed"
        with self.assertRaises(ExecutionError):
            self.agent.run("task")
        self.worker.status = "truncated"
        for _ in range(2):
            self.assertFalse(self.agent.run("task")["cache_hit"])
        self.assertEqual(self.worker.closes, 3)

    def test_bounds_and_eviction(self):
        with self.assertRaises(ValueError):
            self.agent.run("x" * 8193)
        with self.assertRaises(ValueError):
            self.agent.run("go", tokens=257)
        for i in range(17):
            self.agent.run(str(i))
        self.assertFalse(self.agent.run("0")["cache_hit"])
        self.assertTrue(self.agent.run("0")["cache_hit"])
        self.agent.clear()
        self.assertFalse(self.agent.run("0")["cache_hit"])

    def test_batch_shares_one_worker_and_deadline_then_unloads(self):
        rows = self.agent.run_batch(["one", "two", "one"])
        self.assertEqual([r["model_calls"] for r in rows], [1, 1, 0])
        self.assertEqual((self.worker.opens, self.worker.closes), (1, 1))
        self.assertEqual(self.worker.requests[0].deadline, self.worker.requests[1].deadline)
        self.agent.run_batch(["one", "two"])
        self.assertEqual(self.worker.opens, 1)
        with self.assertRaises(ValueError):
            self.agent.run_batch(["one"] * 9)

    def test_cli_session(self):
        out = io.StringIO()
        with patch("bovary.mini.MiniAgent", return_value=self.agent), \
                patch.object(sys, "stdin", io.StringIO("hi\nhi\n/clear\nhi\n/quit\n")), redirect_stdout(out):
            self.assertEqual(cli.main(["mini", "--json"]), 0)
        self.assertEqual([json.loads(line)["model_calls"] for line in out.getvalue().splitlines()], [1, 0, 1])

    def test_cli_single_failure_preserves_the_attempt(self):
        attempt = Result("first", "local", status="failed", text="partial",
                         usage={"emitted_tokens": 1}, timing={"warm": False}).to_dict()
        error = ExecutionError("deadline", "Deadline expired", attempt=attempt)
        out = io.StringIO()
        with patch("bovary.mini.MiniAgent", return_value=self.agent), \
                patch.object(self.worker, "execute", side_effect=error), redirect_stdout(out):
            self.assertEqual(cli.main(["mini", "--json", "--prompt", "one"]), 2)
        self.assertEqual(json.loads(out.getvalue()), {
            "status": "failed", "error": "deadline", "message": "Deadline expired", "attempt": attempt,
        })

    def test_cli_batch_retains_cancelled_attempt_and_later_result(self):
        attempt = Result("first", "local", status="cancelled", text="partial",
                         usage={"emitted_tokens": 1}, timing={"warm": False}).to_dict()
        error = ExecutionError("cancelled", "Request cancelled", attempt=attempt)
        completed = Result("second", "local", status="complete", text="answer")
        out = io.StringIO()
        with patch("bovary.mini.MiniAgent", return_value=self.agent), \
                patch.object(self.worker, "execute", side_effect=[error, completed]), redirect_stdout(out):
            self.assertEqual(cli.main(["mini", "--json", "--prompt", "one", "--prompt", "two"]), 2)
        first, second = map(json.loads, out.getvalue().splitlines())
        self.assertEqual((first["status"], first["error"], first["attempt"]), ("cancelled", "cancelled", attempt))
        self.assertGreaterEqual(first["elapsed_s"], 0)
        self.assertEqual((second["status"], second["text"]), ("complete", "answer"))

    def test_plain_batch_prints_partial_attempt_text_and_reports_failure_separately(self):
        error = ExecutionError("deadline", "Deadline expired", attempt={"text": "partial"})
        completed = Result("second", "local", status="complete", text="answer")
        out, err = io.StringIO(), io.StringIO()
        with patch("bovary.mini.MiniAgent", return_value=self.agent), \
                patch.object(self.worker, "execute", side_effect=[error, completed]), \
                redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(cli.main(["mini", "--prompt", "one", "--prompt", "two"]), 2)
        self.assertEqual(out.getvalue(), "partial\nanswer\n")
        self.assertEqual(err.getvalue(), "Deadline expired\n")

    def test_plain_single_failure_does_not_repeat_streamed_partial_text(self):
        def stream_then_fail(request, on_event):
            on_event(Mock(text="partial"))
            raise ExecutionError("deadline", "Deadline expired", attempt={"text": "partial"})

        out, err = io.StringIO(), io.StringIO()
        with patch("bovary.mini.MiniAgent", return_value=self.agent), \
                patch.object(self.worker, "execute", side_effect=stream_then_fail), \
                redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(cli.main(["mini", "--prompt", "one"]), 2)
        self.assertEqual(out.getvalue(), "partial")
        self.assertEqual(json.loads(err.getvalue()), {
            "status": "failed", "error": "deadline", "message": "Deadline expired",
        })

    def test_gpu_prefill_mini_preset_uses_local_gpu(self):
        agent = MiniAgent(preset="gpu-prefill-2g", worker_factory=lambda: self.worker)
        agent.run("answer")
        request, = self.worker.requests
        self.assertEqual(request.local_profile, "gpu-prefill-2g")
        self.assertTrue(request.allow_gpu and request.offline and request.private)

    def test_cli_passes_host_cache_budget_with_streaming_cuda(self):
        args = ["mini", "--preset", "gpu-prefill-2g", "--prompt", "answer", "--json",
                "--performance-cores", "--host-cache-mib", "4096"]
        with patch("bovary.mini.windows_performance_core_mask", return_value=0xFFF), \
                patch("bovary.mini.PinnedGemma", return_value=self.worker), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args), 0)
        request, = self.worker.requests
        self.assertEqual(request.host_cache_mib, 4096)
        self.assertTrue(request.allow_gpu)

    def test_cli_rejects_incompatible_host_cache_budget(self):
        for preset, size in (("cpu-only", "1"), ("gpu-fast", "1"), ("balanced", "4097"), ("balanced", "-1")):
            with self.subTest(preset=preset, size=size), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main(["mini", "--preset", preset, "--host-cache-mib", size])
            self.assertEqual(error.exception.code, 2)

    def test_cli_performance_cores_passes_detected_mask_to_worker_factory(self):
        args = ["mini", "--preset", "gpu-prefill-2g", "--prompt", "answer", "--json"]
        with patch("bovary.mini.windows_performance_core_mask", return_value=0xFFF) as detect, \
                patch("bovary.mini.PinnedGemma", return_value=self.worker) as worker, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main([*args, "--performance-cores"]), 0)
        detect.assert_called_once_with()
        worker.assert_called_once_with(worker_affinity=0xFFF)
        self.assertEqual(self.worker.requests[0].local_profile, "gpu-prefill-2g")

    def test_cli_performance_cores_reports_unsupported_topology_before_startup(self):
        output = io.StringIO()
        with patch("bovary.mini.windows_performance_core_mask", side_effect=ValueError("unsupported topology")), \
                patch("bovary.mini.PinnedGemma") as worker, redirect_stderr(output):
            with self.assertRaises(SystemExit) as error:
                cli.main(["mini", "--performance-cores", "--prompt", "answer"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("unsupported topology", output.getvalue())
        worker.assert_not_called()

    def test_expression_edit_preserves_precedence_encoding_and_original(self):
        agent = Mock()
        agent.run.return_value = {"text": "`a + b`", "status": "complete"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.py"
            original = b"\xef\xbb\xbfvalue = 3 * old  # comment\r\n"
            path.write_bytes(original)
            row = replace_expression(agent, "Use the sum", path, "old", 32)
            self.assertEqual(row["text"].encode("utf-8"), original.replace(b"old", b"(a + b)"))
            self.assertEqual(path.read_bytes(), original)
            agent.run.return_value = {"text": "a +", "status": "truncated"}
            self.assertEqual(replace_expression(agent, "Use the sum", path, "old", 32)["text"], "")
            agent.run.return_value = {"text": "a; b", "status": "complete"}
            with self.assertRaises(ValueError):
                replace_expression(agent, "Use the sum", path, "old", 32)

    def test_expression_edit_requires_unique_ast_selection_before_loading(self):
        agent = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.py"
            path.write_text("answer = old + old  # label\n", encoding="utf-8")
            for selection in ("old", "missing", "label"):
                with self.subTest(selection=selection), self.assertRaises(ValueError):
                    replace_expression(agent, "Fix it", path, selection, 32)
            agent.run.assert_not_called()

    def test_expression_edit_rejects_ast_that_cannot_compile_in_context(self):
        agent = Mock()
        agent.run.return_value = {"text": "await fetch()", "status": "complete"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.py"
            path.write_text("value = old\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not fit"):
                replace_expression(agent, "Await the result", path, "old", 32)
            path.write_bytes(b"async def load():\n    value = old\n")
            row = replace_expression(agent, "Await the result", path, "old", 32)
        self.assertEqual(row["text"], "async def load():\n    value = await fetch()\n")
