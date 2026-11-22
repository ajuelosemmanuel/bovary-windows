"""Run bounded, independent tasks with the local Gemma worker."""
import argparse
import ast
from collections import OrderedDict
from contextlib import ExitStack
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

from .checkpoint import PinnedGemma
from .cli import configure_input
from .contract import ExecutionError, MAX_HOST_CACHE_MIB, Request, STREAMING_GPU_PROFILES
from .local import windows_performance_core_mask


CONTEXT_TOKENS = 2048
INPUT_BYTES = 8192
CACHE_ENTRIES = 16
PRESETS = ("cpu-only", "balanced", "gpu-fast", "gpu-cache-2g", "gpu-prefill-2g")


def task_prompt(task, source):
    if not isinstance(task, str) or not task.strip() or len(task.encode("utf-8")) > INPUT_BYTES:
        raise ValueError("Enter a nonempty task of at most 8192 UTF-8 bytes")
    if source is None:
        return task
    path = Path(source)
    if not path.is_file():
        raise ValueError("Context must be an existing regular file")
    with path.open("rb") as f:
        content = f.read(INPUT_BYTES + 1)
    if len(content) > INPUT_BYTES:
        raise ValueError("Context file exceeds 8192 bytes")
    return json.dumps({"task": task, "source_file": path.name, "source_text": content.decode("utf-8-sig")},
                      ensure_ascii=False)


def replace_expression(agent, task, path, old, tokens):
    """Generate one expression and reconstruct the source locally. Never run the code."""
    if not old.strip() or len(old.splitlines()) != 1 or old != old.strip():
        raise ValueError("Select one nonempty Python expression without surrounding whitespace")
    if not path.is_file():
        raise ValueError("Context must be an existing regular file")
    with path.open("rb") as source:
        data = source.read(INPUT_BYTES + 1)
    if len(data) > INPUT_BYTES:
        raise ValueError("Context file exceeds 8192 bytes")
    original = data.decode("utf-8-sig")
    if original.count(old) != 1:
        raise ValueError("The selected expression must occur exactly once in the source")
    try:
        tree = ast.parse(original)
        selected = next((node for node in ast.walk(tree) if isinstance(node, ast.expr)
                         and ast.get_source_segment(original, node) == old), None)
        if selected is None:
            raise ValueError("Selection must match a complete expression in the Python source")
    except SyntaxError as exc:
        raise ValueError("Context file must contain valid Python") from exc
    prompt = f"{task} Preserve everything else exactly.\n{original}\nReturn only the replacement for the expression `{old}`."
    result = agent.run(prompt, tokens=tokens)
    if result["status"] != "complete":
        return {**result, "text": ""}
    expr = result["text"].strip()
    if expr.startswith("```") and expr.endswith("```"):
        expr = expr.partition("\n")[2][:-3].strip()
    elif expr.startswith("`") and expr.endswith("`"):
        expr = expr[1:-1]
    if len(expr.splitlines()) != 1:
        raise ValueError("Model did not return one expression")
    try:
        replacement = ast.parse(expr, mode="eval").body
    except SyntaxError as exc:
        raise ValueError("Model returned invalid Python for the selected expression") from exc

    class Substitute(ast.NodeTransformer):
        def visit(self, node):
            return replacement if node is selected else super().visit(node)

    expected_tree = ast.dump(Substitute().visit(tree))
    for insertion in (expr, f"({expr})"):
        updated = original.replace(old, insertion)
        try:
            if ast.dump(ast.parse(updated)) == expected_tree:
                if data.startswith(b"\xef\xbb\xbf"):
                    updated = "\ufeff" + updated
                compile(updated.encode("utf-8"), str(path), "exec", dont_inherit=True)
                return {**result, "replacement": insertion, "text": updated}
        except SyntaxError:
            continue
    raise ValueError("Replacement does not fit the selected expression in this source")


def _failure_row(error):
    code = getattr(error, "code", "input_error")
    row = dict(status="cancelled" if code == "cancelled" else "failed",
               error=code, message=str(error))
    if isinstance(error, ExecutionError) and error.attempt is not None:
        row["attempt"] = error.attempt
    return row


def _print_failure(error, json_output):
    row = _failure_row(error)
    if not json_output:
        row.pop("attempt", None)
    print(json.dumps(row), file=sys.stdout if json_output else sys.stderr, flush=True)


class MiniAgent:
    """One short-lived worker per task or explicit batch. Answers stay in this session."""

    def __init__(self, preset="cpu-only", worker_factory=PinnedGemma, *, host_cache_mib=0):
        if preset not in PRESETS:
            raise ValueError("Unknown mini preset")
        if (type(host_cache_mib) is not int or not 0 <= host_cache_mib <= MAX_HOST_CACHE_MIB
                or host_cache_mib and preset not in STREAMING_GPU_PROFILES):
            raise ValueError("Host cache requires a streaming CUDA preset and a budget from 0 to 4096 MiB")
        self.preset = preset
        self.host_cache_mib = host_cache_mib
        self._worker_factory = worker_factory
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def clear(self):
        with self._lock:
            self._cache.clear()

    def run(self, task, *, source=None, tokens=96, on_text=None):
        return self._dispatch([task], source, tokens, on_text, capture_errors=False)[0]

    def run_batch(self, tasks, *, source=None, tokens=96):
        """Process up to eight independent tasks with one load and one shared deadline."""
        if not isinstance(tasks, (list, tuple)) or not 1 <= len(tasks) <= 8:
            raise ValueError("A mini batch accepts one to eight tasks")
        return self._dispatch(tasks, source, tokens, None, capture_errors=True)

    def _dispatch(self, tasks, source, tokens, on_text, capture_errors):
        if type(tokens) is not int or not 1 <= tokens <= 256:
            raise ValueError("Choose an output limit from 1 to 256 tokens")
        if not self._lock.acquire(blocking=False):
            raise ExecutionError("busy", "A mini task is already active")
        try:
            deadline = time.monotonic() + 180
            rows = []
            with ExitStack() as stack:
                worker = None

                def get_worker():
                    nonlocal worker
                    if worker is None:
                        worker = stack.enter_context(self._worker_factory())
                    return worker

                for task in tasks:
                    started = time.monotonic()
                    try:
                        row = self._answer(task, source, tokens, deadline, get_worker, on_text)
                    except (ValueError, OSError, ExecutionError) as exc:
                        if not capture_errors:
                            raise
                        row = _failure_row(exc)
                    row["elapsed_s"] = time.monotonic() - started
                    rows.append(row)
                cleanup_start = time.monotonic()
            rows[-1]["elapsed_s"] += time.monotonic() - cleanup_start
            return rows
        finally:
            self._lock.release()

    def _answer(self, task, source, tokens, deadline, get_worker, on_text):
        prompt = task_prompt(task, source)
        key = (source is not None, hashlib.sha256(prompt.encode("utf-8")).digest(), self.preset, self.host_cache_mib, tokens)
        if key in self._cache:
            answer = self._cache[key]
            self._cache.move_to_end(key)
            if on_text:
                on_text(answer)
            return dict(status="complete", text=answer, cache_hit=True, model_calls=0, generated_tokens=0)
        request = Request(prompt, local_profile=self.preset, allow_gpu=self.preset != "cpu-only", host_cache_mib=self.host_cache_mib,
                          threads=2, context_tokens=CONTEXT_TOKENS, output_tokens=tokens, deadline=deadline)
        result = get_worker().execute(request, on_event=(lambda event: on_text(event.text)) if on_text else None)
        if result.status == "complete":
            self._cache[key] = result.text
            if len(self._cache) > CACHE_ENTRIES:
                self._cache.popitem(last=False)
        return dict(status=result.status, text=result.text, cache_hit=False, model_calls=1,
                    warm=result.timing.get("warm"), generated_tokens=result.usage.get("output_tokens"))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bovary mini", description=__doc__)
    parser.add_argument("--prompt", action="append", help="One task. Repeat for a series, or omit for interactive input.")
    parser.add_argument("--context-file", type=Path, help="Read this small UTF-8 file afresh for each task")
    parser.add_argument("--replace-expr", metavar="EXPR", help="Generate just this expression's replacement and print the reconstructed Python source")
    parser.add_argument("--preset", choices=PRESETS, default="cpu-only")
    parser.add_argument("--performance-cores", action="store_true",
                        help="Pin the worker to performance cores on hybrid Windows")
    parser.add_argument("--host-cache-mib", type=int, default=0, metavar="MIB",
                        help="Cache up to this many MiB of streamed expert weights in host RAM (Windows CUDA, 0–4096)")
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--json", action="store_true", help="Write one result per line")
    args = parser.parse_args(argv)
    if args.replace_expr is not None and (args.context_file is None or len(args.prompt or []) != 1):
        parser.error("--replace-expr requires --context-file and exactly one --prompt")
    if not 1 <= args.tokens <= 256:
        parser.error("--tokens must be between 1 and 256")
    if args.prompt and len(args.prompt) > 8:
        parser.error("Use at most eight prompts per batch")
    if not 0 <= args.host_cache_mib <= MAX_HOST_CACHE_MIB or (
            args.host_cache_mib and args.preset not in STREAMING_GPU_PROFILES):
        parser.error("Host cache requires a streaming CUDA preset and a budget from 0 to 4096 MiB")
    if args.performance_cores:
        try:
            affinity = windows_performance_core_mask()
        except (ValueError, OSError, AttributeError) as error:
            parser.error(f"Could not detect Windows performance cores: {error}")
        agent = MiniAgent(args.preset, worker_factory=partial(PinnedGemma, worker_affinity=affinity),
                          host_cache_mib=args.host_cache_mib)
    else:
        agent = MiniAgent(args.preset, host_cache_mib=args.host_cache_mib)
    interactive = args.prompt is None and sys.stdin.isatty()
    if interactive:
        print("Mini session. Enter a task, /clear to forget cached answers, or /quit.", file=sys.stderr)
    code = 0
    try:
        if args.prompt is None:
            configure_input()
        if args.prompt and len(args.prompt) > 1:
            rows = agent.run_batch(args.prompt, source=args.context_file, tokens=args.tokens)
            for row in rows:
                if args.json:
                    print(json.dumps(row), flush=True)
                else:
                    print(row.get("text", row.get("attempt", {}).get("text", "")), flush=True)
                    if row["status"] != "complete":
                        print(row.get("message", row["status"]), file=sys.stderr)
            return 2 if any(row["status"] != "complete" for row in rows) else 0
        for line in args.prompt if args.prompt is not None else sys.stdin:
            task = line.rstrip("\r\n") if args.prompt is None else line
            if args.prompt is None and task == "/quit":
                break
            if args.prompt is None and task == "/clear":
                agent.clear()
                continue
            if args.prompt is None and not task.strip():
                continue
            try:
                if args.replace_expr is None:
                    result = agent.run(task, source=args.context_file, tokens=args.tokens,
                                       on_text=None if args.json else lambda text: print(text, end="", flush=True))
                else:
                    result = replace_expression(agent, task, args.context_file, args.replace_expr, args.tokens)
                if args.json:
                    print(json.dumps(result), flush=True)
                else:
                    if args.replace_expr is None:
                        print(flush=True)
                    elif sys.stdout is sys.__stdout__ and not sys.stdout.isatty() and hasattr(sys.stdout, "buffer"):
                        sys.stdout.buffer.write(result["text"].encode("utf-8"))
                        sys.stdout.buffer.flush()
                    else:
                        sys.stdout.write(result["text"])
                        sys.stdout.flush()
                    if result["status"] != "complete":
                        print(result["status"], file=sys.stderr)
                if result["status"] != "complete":
                    code = 2
            except (ExecutionError, OSError, ValueError) as error:
                _print_failure(error, args.json)
                code = 2
    except KeyboardInterrupt:
        return 130
    except ExecutionError as error:
        _print_failure(error, args.json)
        return 2
    except (OSError, UnicodeError) as error:
        print(f"Mini session input/output error: {error}", file=sys.stderr)
        return 2
    return code
