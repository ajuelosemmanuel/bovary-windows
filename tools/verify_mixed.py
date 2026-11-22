"""Verify the thirty-gate/up mixed quantization and unchanged payload bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

try:
    from .gguf_layout import CHUNK, hash_file, read_layout
except ImportError:
    from gguf_layout import CHUNK, hash_file, read_layout


def verify(source_path, candidate_path, expected_source_sha256=None):
    started = time.perf_counter()
    source, candidate = read_layout(source_path), read_layout(candidate_path)
    if Path(source_path).samefile(candidate_path):
        raise ValueError("source and candidate must be distinct files")
    if source["metadata"].get("general.architecture") != "gemma4" or candidate["metadata"].get("general.architecture") != "gemma4":
        raise ValueError("expected Gemma4 source and candidate")
    source_hparams = {key: value for key, value in source["metadata"].items() if key.startswith("gemma4.")}
    candidate_hparams = {key: value for key, value in candidate["metadata"].items() if key.startswith("gemma4.")}
    if source_hparams != candidate_hparams:
        raise ValueError("Gemma4 architecture metadata differs")
    if source["tokenizer_value_sha256"] != candidate["tokenizer_value_sha256"]:
        raise ValueError("tokenizer metadata differs, including raw types/values")
    original = {tensor["name"]: tensor for tensor in source["tensors"]}
    derived = {tensor["name"]: tensor for tensor in candidate["tensors"]}
    if original.keys() != derived.keys() or len(original) != 658:
        raise ValueError("tensor names/count differ from the expected checkpoint")
    selected = re.compile(r"blk\.([0-9]+)\.ffn_gate_up_exps\.weight")
    changed, unchanged, layers = [], [], set()
    copied_bytes = 0
    with Path(source_path).open("rb", buffering=0) as left, Path(candidate_path).open("rb", buffering=0) as right:
        for name, tensor in original.items():
            other = derived[name]
            if tensor["shape"] != other["shape"]:
                raise ValueError("tensor shape changed: " + name)
            match = selected.fullmatch(name)
            if match:
                layer = int(match[1])
                if layer not in range(30) or tensor["shape"] != [2816, 1408, 128] or tensor["type"] != "Q4_0" or other["type"] != "Q3_K":
                    raise ValueError("unexpected gate/up conversion: " + name)
                if tensor["nbytes"] % 144 or other["nbytes"] != tensor["nbytes"] // 144 * 110:
                    raise ValueError("unexpected gate/up payload size")
                layers.add(layer)
                changed.append({"name": name, "source_type": tensor["type"], "candidate_type": other["type"],
                                "source_bytes": tensor["nbytes"], "candidate_bytes": other["nbytes"]})
                continue
            if tensor["type"] != other["type"] or tensor["nbytes"] != other["nbytes"]:
                raise ValueError("unexpected type/size change: " + name)
            left.seek(tensor["offset"])
            right.seek(other["offset"])
            remaining, digest = tensor["nbytes"], hashlib.sha256()
            while remaining:
                amount = min(CHUNK, remaining)
                a, b = left.read(amount), right.read(amount)
                if len(a) != amount or len(b) != amount or a != b:
                    raise ValueError("unchanged tensor bytes differ or are truncated: " + name)
                digest.update(a)
                remaining -= amount
            copied_bytes += tensor["nbytes"]
            unchanged.append({"name": name, "bytes": tensor["nbytes"], "sha256": digest.hexdigest(), "byte_exact": True})
    if len(changed) != 30 or layers != set(range(30)):
        raise ValueError("expected exactly one changed gate/up tensor in each of thirty layers")
    source_hash, candidate_hash = hash_file(source_path), hash_file(candidate_path)
    if expected_source_sha256 and source_hash != expected_source_sha256.lower():
        raise ValueError("source checkpoint SHA256 differs from expected provenance")
    return {"verified": True, "recipe": "custom gate_up Q3_K; all other source payload bytes exact",
            "source": {"path": source["path"], "sha256": source_hash, "bytes": source["file_bytes"]},
            "candidate": {"path": candidate["path"], "sha256": candidate_hash, "bytes": candidate["file_bytes"],
                          "name": candidate["metadata"].get("general.name"),
                          "recipe_metadata": candidate["metadata"].get("quantize.recipe")},
            "architecture_metadata_exact": True, "tokenizer_raw_values_exact": True,
            "tokenizer_key_count": len(source["tokenizer_value_sha256"]),
            "tokenizer_value_sha256": source["tokenizer_value_sha256"],
            "unchanged_tensor_count": len(unchanged), "unchanged_bytes_compared": copied_bytes,
            "changed_tensor_count": len(changed), "changed_tensors": changed,
            "unchanged_tensors": unchanged, "comparison_buffer_bytes": CHUNK * 2,
            "disk_saving_bytes": source["file_bytes"] - candidate["file_bytes"],
            "elapsed_seconds": time.perf_counter() - started,
            "quality_limit": "Unchanged tensors/tokenizer exact; thirty requantized tensors remain quality-changing and need inference evaluation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source-sha256")
    parser.add_argument("--output", default="outputs/mixed-verification.json")
    args = parser.parse_args()
    try:
        result = verify(args.source, args.candidate, args.source_sha256)
    except (ValueError, OSError) as error:
        result = {"verified": False, "error": str(error)}
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise SystemExit(str(error))
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("changed_tensors", "unchanged_tensors", "tokenizer_value_sha256")}, indent=2))


if __name__ == "__main__":
    main()
