"""Answer one short question with the registered Windows CUDA preview."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

from run import configuration, registered_inputs


def ask(prompt, tokens=96, *, timeout=180):
    if not prompt.strip() or len(prompt.encode("utf-8")) > 8192:
        raise ValueError("Enter a nonempty prompt of at most 8192 UTF-8 bytes.")
    if type(tokens) is not int or not 1 <= tokens <= 256:
        raise ValueError("Choose 1 to 256 output tokens.")
    directory = os.environ.get("BOVARY_INKLING_BIN")
    receipt = os.environ.get("BOVARY_INKLING_RECEIPT")
    if not directory or not receipt:
        raise ValueError("Inkling requires a CUDA preview build and registered model. Set BOVARY_INKLING_BIN and BOVARY_INKLING_RECEIPT.")
    inputs = registered_inputs(Path(receipt))
    with tempfile.TemporaryDirectory(prefix="bovary-inkling-") as scratch:
        question = Path(scratch) / "prompt.txt"
        metrics = Path(scratch) / "metrics.json"
        question.write_text(prompt, encoding="utf-8")
        args = SimpleNamespace(bin=Path(directory).resolve(), prompt=question, metrics=metrics, tokens=tokens, **inputs)
        command, env = configuration(args)
        try:
            answer = subprocess.run(command, cwd=args.bin, env=env, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Inkling exceeded {timeout} seconds. Try a shorter question or choose Gemma.") from error
        if answer.returncode:
            detail = answer.stderr.decode("utf-8", errors="replace")[-1500:].strip()
            raise RuntimeError(f"Inkling worker exited with {answer.returncode}: {detail}")
        reason = json.loads(metrics.read_bytes())["finish_reason"]
        if reason not in {"eos", "end_message"}:
            raise ValueError(f"Inkling stopped at {reason}. Shorten the prompt or increase max_tokens.")
        return answer.stdout.decode("utf-8").strip()
