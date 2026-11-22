"""Generate the small initial serial experiment matrix with absolute paths."""
import argparse
import hashlib
import json
from build_provenance import load_build, verified_model
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="work/initial-config.json")
    p.add_argument("--mode", choices=("smoke", "matrix", "workloads"), default="smoke")
    p.add_argument("--build-manifest", required=True, type=Path)
    a = p.parse_args()
    executable, build_metadata = load_build(a.build_manifest, "cuda", ROOT)
    model_path, model_metadata = verified_model(ROOT)
    model, exe = str(model_path), str(executable)
    toolkit = ROOT / "work/deps/cuda-13.3"
    import os
    path = str(toolkit / "bin") + os.pathsep + str(toolkit / "bin/x64") + os.pathsep + os.environ.get("PATH", "")
    base = [exe, "--model", model, "--prompt-file", "{prompt_file}",
            "--metrics", "{metrics_file}", "--predict", "{n_predict}",
            "--context", "{ctx}", "--threads", "8"]
    configs = []
    source_metadata = {**build_metadata, **model_metadata,
                       "sampling": "greedy", "ubatch": 1,
                       "declared_settings": "Greedy sampling and one-token prefill; hardware state is not inferred.",
                       "optional_components": "omitted", "extra_cpu_weight_repacking": False}
    def add(name, extra, env=None):
        chosen_env = {"PATH": path, "BOVARY_STREAM": "0", **(env or {})}
        if chosen_env.get("BOVARY_STREAM") == "1":
            chosen_env["BOVARY_STREAM_TRACE"] = "{stream_trace_file}"
        configs.append({"id": name, "command": base + extra,
                        "env": chosen_env,
                        "stdout_is_generation": True, "cache_condition": "unknown; process restart only",
                        "metadata": source_metadata.copy()})
    if a.mode == "smoke":
        add("E1-direct512", ["--gpu-layers", "99", "--cpu-experts", "--cache-mib", "512", "--logits", "{logits_file}"],
            {"BOVARY_STREAM": "1", "BOVARY_STREAM_MODEL": model, "BOVARY_STREAM_IO": "unbuffered"})
        configs[-1]["command"].extend(["--load-mode", "direct"])
        add("E1-mmap512", ["--gpu-layers", "99", "--cpu-experts", "--cache-mib", "512", "--logits", "{logits_file}"],
            {"BOVARY_STREAM": "1", "BOVARY_STREAM_MODEL": model, "BOVARY_STREAM_IO": "unbuffered"})
        add("E1-direct512-gpuembed", ["--gpu-layers", "99", "--cpu-experts", "--cache-mib", "512", "--gpu-embedding", "--load-mode", "direct", "--logits", "{logits_file}"],
            {"BOVARY_STREAM": "1", "BOVARY_STREAM_MODEL": model, "BOVARY_STREAM_IO": "unbuffered"})
        add("B2-cache512", ["--gpu-layers", "99", "--cpu-experts", "--cache-mib", "512", "--logits", "{logits_file}"])
    elif a.mode == "matrix":
        add("B0-partial8", ["--gpu-layers", "8"])
        add("B1-cpu-experts", ["--gpu-layers", "99", "--cpu-experts"])
        add("B2-cache512", ["--gpu-layers", "99", "--cpu-experts", "--cache-mib", "512"])
        for cache in (32, 512, 1536):
            add(f"E2-direct{cache}", ["--gpu-layers", "99", "--cpu-experts", "--cache-mib", str(cache)],
                {"BOVARY_STREAM": "1", "BOVARY_STREAM_MODEL": model, "BOVARY_STREAM_IO": "unbuffered"})
            configs[-1]["command"].extend(["--load-mode", "direct"])
        add("E2-direct512-buffer1", ["--gpu-layers", "99", "--cpu-experts", "--cache-mib", "512", "--load-mode", "direct"],
            {"BOVARY_STREAM": "1", "BOVARY_STREAM_MODEL": model, "BOVARY_STREAM_IO": "unbuffered", "BOVARY_STREAM_BUFFER_MIB": "1"})
        add("E1-buffered512", ["--gpu-layers", "99", "--cpu-experts", "--cache-mib", "512"],
            {"BOVARY_STREAM": "1", "BOVARY_STREAM_MODEL": model, "BOVARY_STREAM_IO": "buffered"})
        configs[-1]["command"].extend(["--load-mode", "direct"])
    else:
        add("E1-direct512", ["--gpu-layers", "99", "--cpu-experts", "--cache-mib", "512"],
            {"BOVARY_STREAM": "1", "BOVARY_STREAM_MODEL": model, "BOVARY_STREAM_IO": "unbuffered"})
        configs[-1]["command"].extend(["--load-mode", "direct"])
    if a.mode != "smoke":
        for configuration in configs:
            if configuration["env"].get("BOVARY_STREAM") == "1":
                configuration["command"].append("--gpu-embedding")
    config = {"context": 4096, "n_predict": 8 if a.mode == "smoke" else 32,
              "workloads": ["short"] if a.mode != "workloads" else ["medium", "long", "sustained"],
              "generation_tokens": {"sustained": 128}, "repetitions": 1 if a.mode == "smoke" else 3,
              "warmups": 0, "sample_interval_seconds": .2, "timeout_seconds": 3600,
              "pdh": True, "configurations": configs}
    Path(a.output).write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(a.output)

if __name__ == "__main__":
    main()
