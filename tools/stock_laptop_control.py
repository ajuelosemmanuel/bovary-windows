"""Same prompt framing and greedy decoding in upstream llama.cpp (local comparison)."""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, ProxyHandler, build_opener

from bovary.checkpoint import PinnedGemma
from bovary.laptop_benchmark import CASES, MemorySampler, OUTPUT_TOKENS, THREADS, row, summarize
from bovary.mini import CONTEXT_TOKENS


def compare(server, model, manifest, log_path):
    model, _ = PinnedGemma(model=model, manifest=manifest)._checkpoint()
    version = subprocess.run([str(server), "--version"], cwd=server.parent, check=True, timeout=10,
                             capture_output=True)
    if b"build 12600, commit 29b52d398" not in version.stdout + version.stderr:
        raise ValueError("Expected upstream llama.cpp b12600, commit 29b52d398")
    # Match the worker's exact template and its add_special/parse_special options.
    api_key = secrets.token_urlsafe(24)
    opener = build_opener(ProxyHandler({}))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def request(path, data=None, timeout=5):
        encoded = None if data is None else json.dumps(data).encode()
        url = f"http://127.0.0.1:{port}{path}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        return opener.open(Request(url, data=encoded, headers=headers), timeout=timeout)

    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("LLAMA_ARG_", "BOVARY_STREAM")) and key != "GGML_BACKEND_PATH"}
    command = [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
               "--api-key", api_key, "--no-webui", "--no-warmup", "-np", "1", "-c", str(CONTEXT_TOKENS),
               "-t", str(THREADS), "-tb", str(THREADS), "--device", "CUDA0",
               "--gpu-layers", "auto", "--fit", "on", "--verbose"]
    rows = []
    process = None
    with log_path.open("x", encoding="utf8") as log, MemorySampler(lambda: process) as memory:
        try:
            started = time.monotonic()
            process = subprocess.Popen(command, cwd=server.parent, env=env, stdout=log, stderr=log)
            deadline = started + 180
            while True:
                try:
                    with request("/health", timeout=1) as response:
                        if json.load(response)["status"] == "ok":
                            break
                except (URLError, TimeoutError):
                    pass
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Stock server did not become ready")
                time.sleep(.1)
            for index, case in enumerate(CASES):
                if index:
                    started = time.monotonic()
                prompt = "<|turn>user\n" + case[1] + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
                with request("/tokenize", {"content": prompt, "add_special": True, "parse_special": True}) as response:
                    tokens = json.load(response)["tokens"]
                fragments = []
                first_text = None
                payload = {"prompt": tokens, "stream": True, "cache_prompt": False,
                           "n_predict": OUTPUT_TOKENS, "temperature": 0, "seed": 0, "return_tokens": True,
                           "repeat_penalty": 1, "frequency_penalty": 0, "presence_penalty": 0}
                final = None
                with request("/completion", payload, timeout=180) as response:
                    for line in response:
                        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                            continue
                        item = json.loads(line[6:])
                        if "error" in item:
                            raise RuntimeError("Stock completion failed")
                        if item.get("stop"):
                            final = item
                        elif item.get("content"):
                            fragments.append(item["content"])
                            if first_text is None:
                                first_text = time.monotonic() - started
                if final is None:
                    raise RuntimeError("Stock stream did not finish")
                elapsed = time.monotonic() - started
                timings = dict(ttft_s=first_text, end_to_end_s=elapsed)
                rows.append(row(case, timings, "".join(fragments),
                                state="loaded" if index else "startup",
                                input_tokens=len(tokens), output_tokens=final.get("tokens_predicted"),
                                failure=None if final.get("stop_type") == "eos" else "incomplete"))
        finally:
            if process is not None:
                process.terminate()
                process.wait(timeout=10)
    return summarize("stock llama.cpp b12600 CUDA", rows, memory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("log", type=Path, help="New private server log file")
    args = parser.parse_args()
    report = compare(args.server.resolve(), args.model.resolve(), args.manifest.resolve(), args.log)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["correct"] == report["total"] else 2)
