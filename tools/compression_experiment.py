"""Independent expert-slice lossless compression; no second checkpoint is retained."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "work/deps/python").exists():
    sys.path.insert(0, str(ROOT / "work/deps/python"))
sys.path.insert(0, str(ROOT / "work/engine/bovary-llama.cpp/gguf-py"))
from gguf import GGUFReader
import zstandard as zstd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="outputs/compression.json")
    parser.add_argument("--experts-per-tensor", type=int, default=4)
    args = parser.parse_args()
    model = Path(args.model)
    reader = GGUFReader(model)
    descriptors = []
    expert_total = 0
    for tensor in reader.tensors:
        if "_exps.weight" not in tensor.name:
            continue
        count = int(tensor.shape[2])
        size = tensor.n_bytes // count
        assert tensor.n_bytes % count == 0
        expert_total += tensor.n_bytes
        for expert in sorted({i * (count - 1) // max(1, args.experts_per_tensor - 1)
                              for i in range(args.experts_per_tensor)}):
            descriptors.append((tensor.name, expert, int(tensor.data_offset) + expert * size, size))
    del reader
    records = []
    with model.open("rb", buffering=0) as source:
        for name, expert, offset, size in descriptors:
            source.seek(offset)
            raw = source.read(size)
            assert len(raw) == size
            digest = hashlib.sha256(raw).hexdigest()
            for level in (1, 5):
                compressor = zstd.ZstdCompressor(level=level, write_checksum=True)
                start = time.perf_counter_ns()
                packed = compressor.compress(raw)
                compress_ns = time.perf_counter_ns() - start
                times = []
                for _ in range(3):
                    start = time.perf_counter_ns()
                    restored = zstd.ZstdDecompressor().decompress(packed, max_output_size=size)
                    times.append(time.perf_counter_ns() - start)
                    assert restored == raw
                records.append({"tensor": name, "expert": expert, "offset": offset,
                                "raw_bytes": size, "compressed_bytes": len(packed),
                                "zstd_level": level, "raw_sha256": digest,
                                "compress_ns": compress_ns, "decompress_ns": times,
                                "byte_exact": True})
    summaries = []
    for level in (1, 5):
        subset = [r for r in records if r["zstd_level"] == level]
        raw_bytes = sum(r["raw_bytes"] for r in subset)
        packed_bytes = sum(r["compressed_bytes"] for r in subset)
        ratio = packed_bytes / raw_bytes
        summaries.append({"level": level, "sample_raw_bytes": raw_bytes,
                          "sample_compressed_bytes": packed_bytes, "ratio": ratio,
                          "estimated_full_model_bytes": model.stat().st_size - expert_total + expert_total * ratio,
                          "median_decompress_ms": statistics.median(statistics.median(r["decompress_ns"]) for r in subset) / 1e6,
                          "quality": "byte-exact roundtrip for every sampled expert slice",
                          "estimate_limit": "stratified 4 expert IDs/tensor; no full model compressed or inference integration"})
    result = {"model": str(model.resolve()), "model_bytes": model.stat().st_size,
              "expert_total_bytes": expert_total, "sample_method": "evenly spaced expert IDs in every routed tensor",
              "scratch_policy": "one source slice, compressed frame and one restored slice; no output checkpoint",
              "summaries": summaries, "raw_samples": records}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))

if __name__ == "__main__":
    main()
