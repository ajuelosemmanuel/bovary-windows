"""Protocol fixture only. It does not load a model or provide quality evidence."""
import os
import sys
import time

if "--fake-old-brand" in sys.argv and "--protocol-version" in sys.argv:
    print("lowcost-worker 1", flush=True)
    sys.exit(0)
if "--fake-old-protocol" in sys.argv and "--protocol-version" in sys.argv:
    print("bovary-worker 1", flush=True)
    sys.exit(0)
for arg in sys.argv[1:]:
    if arg == "--require-cpu-only" and "--fake-cuda-only" in sys.argv:
        sys.exit(1)
    if arg == "--protocol-version":
        if "--fake-delay-version" in sys.argv:
            time.sleep(30)
        if "--fake-hold-version-pipe" in sys.argv:
            import json
            import subprocess

            # Record/own the actual holder, rather than a Windows venv redirector.
            child = subprocess.Popen([sys._base_executable, "-S", "-c", "import time; time.sleep(4)"],
                stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            with open(os.environ["BOVARY_FAKE_PROBE_RECORD"], "w", encoding="utf-8") as record:
                json.dump({"parent": os.getpid(), "child": child.pid}, record)
        if "--fake-oversized-version" in sys.argv:
            sys.stdout.buffer.write(b"bovary-worker 2\n" + b"x" * (1024 * 1024))
            sys.stdout.buffer.flush()
            time.sleep(4)
            sys.exit(0)
        print("bovary-worker 3" if "--fake-wrong-version" in sys.argv else "bovary-worker 2", flush=True)
        sys.exit(0)

import json

if "BOVARY_FAKE_WORKER_STARTED" in os.environ:
    with open(os.environ["BOVARY_FAKE_WORKER_STARTED"], "w", encoding="utf-8") as record:
        record.write(str(os.getpid()))

if os.name == "nt":
    import msvcrt
    msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)


def frame(kind, value):
    if isinstance(value, dict):
        value = json.dumps(value).encode()
    sys.stdout.buffer.write(f"{kind} {len(value)}\n".encode() + value)
    sys.stdout.buffer.flush()


if "--fake-delay-ready" in sys.argv:
    time.sleep(30)
frame("R", {"protocol": 1 if "--fake-old-ready" in sys.argv else 2, "model_load_s": .01})
context = int(sys.argv[sys.argv.index("--context") + 1])
for line in sys.stdin.buffer:
    if line == b"QUIT\n":
        break
    _, count, length = line.split()
    requested = int(count)
    prompt = sys.stdin.buffer.read(int(length)).decode("utf-8")
    if prompt == "CRASH":
        sys.exit(9)
    if prompt == "SILENT":
        time.sleep(30)
    if prompt == "OVERFLOW":
        frame("E", {"code": "context_overflow", "message": "prompt/generation exceeds context"})
        continue
    if prompt == "BADFRAME":
        sys.stdout.buffer.write(b"T 999999999\n")
        sys.stdout.buffer.flush()
        continue
    prompt_tokens = context if prompt == "NO_CONTEXT" else context - 4 if prompt == "CONTEXT_LIMIT" else len(prompt)
    generation_limit = requested or context - prompt_tokens
    if generation_limit < 1 or prompt_tokens + generation_limit > context:
        frame("E", {"code": "context_overflow", "message": "prompt/generation exceeds context"})
        continue
    early_stop = prompt not in ("TOKEN_LIMIT", "CONTEXT_LIMIT") and generation_limit > 2
    if prompt == "EOG_EMPTY":
        generated = 0
        early_stop = True
    elif not early_stop:
        generated = generation_limit
        for _ in range(generated):
            frame("T", b"x")
    else:
        generated = 2
        data = ("reply:" + prompt + " 世界").encode("utf-8")
        # Split a UTF-8 character across frames as native token pieces may do.
        frame("T", data[:-2])
        frame("T", data[-2:])
    frame("D", {"prompt_tokens": prompt_tokens, "actual_generated_tokens": generated,
                "requested_generated_tokens": requested, "generation_limit": generation_limit,
                "finish_reason": "eos" if early_stop else "context_limit" if requested == 0 else "token_limit",
                "early_stop": early_stop, "tokens": list(range(generated)), "pid": os.getpid()})
