"""Recreate the final matched experiments from verified local artifacts."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def set_value(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("mixed-smoke", "idle", "backend-check", "workloads", "load-smoke", "load-matrix", "context-probe"), required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--build-manifest", type=Path, required=True)
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="bovary-template-") as temporary:
        template = Path(temporary)/"template.json"
        subprocess.run([sys.executable, str(ROOT/"tools/make_config.py"), "--mode", "matrix",
                        "--output", str(template), "--build-manifest", str(args.build_manifest)], check=True)
        config = json.loads(template.read_text(encoding="utf-8"))
    original = {x["id"]: x for x in config["configurations"]}
    verification_path = ROOT / "outputs/mixed-verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else None
    for c in original.values():
        c["metadata"]["background_condition"] = "Uncontrolled; no claim about other applications, clock locking or contention"
    def stream(name, budget, pool="pinned2", buffer_mib=8, mixed=False):
        c = copy.deepcopy(original["E2-direct512"])
        c["id"] = name
        set_value(c["command"], "--cache-mib", budget)
        c["env"]["BOVARY_STREAM_POOL"] = pool
        c["env"]["BOVARY_STREAM_BUFFER_MIB"] = str(buffer_mib)
        if mixed:
            if not verification or not verification.get("verified"):
                raise RuntimeError("Run verified mixed conversion before this experiment")
            candidate = verification["candidate"]
            path = ROOT / "work/models" / Path(candidate["path"]).name
            if path.stat().st_size != candidate["bytes"]:
                raise RuntimeError("Mixed checkpoint size no longer matches verification")
            set_value(c["command"], "--model", path)
            c["env"]["BOVARY_STREAM_MODEL"] = str(path)
            c["metadata"].update(model_sha256=candidate["sha256"], model_bytes=candidate["bytes"],
                                  quantization="custom gate_up Q3_K; remaining 628 tensor payloads source-exact",
                                  numerical_quality="lossy requantization experiment; separate quality evaluation required")
        return c
    chosen = []
    if args.mode == "context-probe":
        chosen = []
        for size in (4096, 8192):
            c = stream(f"context{size}-cache512", 512)
            c["env"]["BOVARY_STREAM_LOAD_BUFFER_MIB"] = "4"
            c["context"] = size
            set_value(c["command"], "--context", size)
            c["command"].extend(["--logits", "{logits_file}"])
            c["metadata"].update(context_override=size, purpose="allocation/short-prompt context probe; not a full 8K-token history quality test")
            chosen.append(c)
        config.update(n_predict=32, repetitions=3, interleave_repetitions=True)
    elif args.mode in ("load-smoke", "load-matrix"):
        chosen = [stream("V3-default1024", 1024),
                  stream("V3-minhost512-load1", 512, pool="serial", buffer_mib=1),
                  stream("V3-balanced1024-load4", 1024)]
        chosen[1]["env"]["BOVARY_STREAM_LOAD_BUFFER_MIB"] = "1"
        chosen[2]["env"]["BOVARY_STREAM_LOAD_BUFFER_MIB"] = "4"
        if args.mode == "load-smoke":
            config.update(n_predict=8, repetitions=1)
            for c in chosen:
                c["command"].extend(["--logits", "{logits_file}"])
        else:
            chosen = chosen[1:] + [stream("V3-mixed1024-load4", 1024, mixed=True)]
            chosen[-1]["env"]["BOVARY_STREAM_LOAD_BUFFER_MIB"] = "4"
            config.update(n_predict=32, repetitions=3, interleave_repetitions=True)
    elif args.mode == "mixed-smoke":
        control = stream("mixed-reference1024", 1024, pool="serial", mixed=True)
        control["env"]["BOVARY_STREAM"] = "0"
        set_value(control["command"], "--load-mode", "mmap")
        chosen = [control, stream("mixed-pinned1024", 1024, mixed=True)]
        for c in chosen:
            c["command"].extend(["--logits", "{logits_file}", "--choices", "ABCD"])
        config.update(n_predict=8, repetitions=1)
    elif args.mode == "backend-check":
        for source in ("B0-partial8", "B2-cache512"):
            c = copy.deepcopy(original[source])
            c["id"] = "diagnostic-" + source
            c["command"].extend(["--logits", "{logits_file}"])
            chosen.append(c)
        batched = copy.deepcopy(original["B0-partial8"])
        batched["id"] = "diagnostic-B0-batched128"
        batched["command"].extend(["--prompt-batch", "128", "--logits", "{logits_file}"])
        batched["metadata"]["purpose"] = "practical batched-prefill smoke, not a storage-only equivalence control"
        batched["metadata"].update(prompt_batch=128, ubatch=128)
        chosen.append(batched)
        config.update(n_predict=32, repetitions=1)
    elif args.mode == "idle":
        for source in ("B0-partial8", "B1-cpu-experts", "B2-cache512"):
            c = copy.deepcopy(original[source])
            c["id"] += "-idle"
            chosen.append(c)
        chosen.extend([stream("E2-pinned32-idle", 32),
                       stream("E1-minhost512-idle", 512, pool="serial", buffer_mib=1),
                       stream("E2-pinned512-idle", 512), stream("E2-pinned1024-idle", 1024),
                       stream("E2-pinned1536-idle", 1536), stream("E3-mixed1024-idle", 1024, mixed=True)])
        config.update(n_predict=32, repetitions=3, interleave_repetitions=True)
    else:
        control = copy.deepcopy(original["B2-cache512"])
        control["id"] = "B2-reference-workloads"
        practical = copy.deepcopy(original["B0-partial8"])
        practical["id"] = "B0-batched128-workloads"
        set_value(practical["command"], "--prompt-batch", 128)
        practical["metadata"]["prompt_batch"] = 128
        practical["metadata"]["ubatch"] = 128
        practical["metadata"]["purpose"] = "practical ordinary batched-prefill baseline; separate from batch=1 architecture controls"
        chosen = [control, stream("E2-balanced-workloads", 1024), practical]
        chosen[1]["env"]["BOVARY_STREAM_LOAD_BUFFER_MIB"] = "4"
        config.update(workloads=["medium", "long", "sustained"], n_predict=32,
                      generation_tokens={"sustained": 128}, repetitions=3, interleave_repetitions=True)
        for c in chosen:
            c["command"].extend(["--logits", "{logits_file}"])
            c["metadata"]["full_logits_dump"] = "enabled equally in paired workload comparisons; affects measured end-to-end/TTFT slightly"
    config["configurations"] = chosen
    args.output.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(args.output)

if __name__ == "__main__":
    main()
