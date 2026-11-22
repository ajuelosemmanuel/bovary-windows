"""Make a monitored one-shot conversion experiment (no inference throughput)."""
import argparse
import hashlib
import json
from build_provenance import load_build, verified_model
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIXED = "gemma-4-26B-A4B-it-gateup-Q3_K-rest-Q4_0.gguf"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "work/quant-config.json")
    p.add_argument("--build-manifest", type=Path, required=True)
    args = p.parse_args()
    source, model_metadata = verified_model(ROOT)
    candidate = ROOT / "work/models" / MIXED
    exe, build_metadata = load_build(args.build_manifest, "cuda", ROOT, "bovary-quantize.exe")
    cuda = ROOT / "work/deps/cuda-13.3"
    env = {"BOVARY_STREAM": "0", "PATH": str(cuda / "bin/x64") + os.pathsep + str(cuda / "bin") + os.pathsep + os.environ.get("PATH", "")}
    command = [str(exe), "--input", str(source), "--output", str(candidate), "--threads", "4"]
    metadata = {**build_metadata, **model_metadata, "purpose": "lossy mixed quantization conversion, not generated-token benchmark",
                "source_model_sha256": model_metadata["model_sha256"],
                "recipe": "only 30 gate_up Q4_0 tensors become Q3_K; all 628 other tensor payloads preserved",
                "row_scratch_budget_bytes": 64 * 1024**2,
                "quantizer_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
                "source_model_bytes": source.stat().st_size, "candidate": str(candidate)}
    config = {"configurations": [{"id": "mixed-quantization", "command": command, "env": env,
                                  "stdout_is_generation": False, "metadata": metadata,
                                  "cache_condition": "unknown; conversion reads mmap source"}],
              "workloads": ["short"], "repetitions": 1, "warmups": 0,
              "context": 4096, "n_predict": 0, "timeout_seconds": 7200,
              "sample_interval_seconds": .2, "pdh": True,
              "monitor_disk_path": str(ROOT)}
    args.output.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(args.output)

if __name__ == "__main__":
    main()
