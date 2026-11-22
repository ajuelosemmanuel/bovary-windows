"""Serial fixed ten-item quality probe; not a broad model-quality benchmark."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/deps/python"))
import numpy as np
from benchmark import SerialLock

def log_softmax(values):
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite logits")
    maximum = values.max()
    return values - maximum - np.log(np.exp(values - maximum).sum())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", default="outputs/quality", type=Path)
    parser.add_argument("--cache-mib", type=int, default=1536)
    parser.add_argument("--pool", choices=("serial", "pinned2"), default="serial")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases_path = ROOT / "benchmarks/quality_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    exe = ROOT / "work/build/bovary-cuda/bovary-infer.exe"
    environment = {**os.environ, "BOVARY_STREAM": "1", "BOVARY_STREAM_IO": "unbuffered",
                   "BOVARY_STREAM_BUFFER_MIB": "8", "BOVARY_STREAM_POOL": args.pool}
    cuda = ROOT / "work/deps/cuda-13.3"
    environment["PATH"] = str(cuda / "bin/x64") + os.pathsep + str(cuda / "bin") + os.pathsep + environment.get("PATH", "")
    variants = {}
    commands = []
    with SerialLock(ROOT / "work/benchmark.lock"):
        for label, model in (("reference", args.reference.resolve()), ("candidate", args.candidate.resolve())):
            with model.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            variants[label] = {"path": str(model), "bytes": model.stat().st_size, "sha256": digest, "cases": []}
            env = {**environment, "BOVARY_STREAM_MODEL": str(model)}
            for case in cases:
                directory = output / label / case["id"]
                directory.mkdir(parents=True)
                prompt = "Answer the multiple-choice question with only its single capital letter A, B, C, or D.\n" + case["question"]
                (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
                command = [str(exe), "--model", str(model), "--prompt-file", str(directory / "prompt.txt"),
                           "--metrics", str(directory / "native-metrics.json"), "--logits", str(directory / "logits.f32"),
                           "--predict", "8", "--context", "4096", "--threads", "8", "--gpu-layers", "99", "--cpu-experts",
                           "--gpu-embedding", "--cache-mib", str(args.cache_mib), "--load-mode", "direct", "--choices", "ABCD"]
                print(f"Quality {label}: {case['id']}", flush=True)
                with (directory / "stdout.bin").open("wb") as stdout, (directory / "stderr.bin").open("wb") as stderr:
                    result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, timeout=600)
                commands.append({"variant": label, "case": case["id"], "command": command, "return_code": result.returncode})
                if result.returncode:
                    raise RuntimeError(f"Quality run failed: {label}/{case['id']}")
                native = json.loads((directory / "native-metrics.json").read_text(encoding="utf-8"))
                text = (directory / "stdout.bin").read_bytes().decode("utf-8", errors="replace").strip()
                if native.get("choices") != "ABCD" or len(native.get("choice_ids", [])) != 4:
                    raise ValueError("Missing validated single-token choices")
                vocabulary = native["vocab_size"]
                expected_rows = native["actual_generated_tokens"] + int(native["early_stop"])
                if (directory / "logits.f32").stat().st_size != vocabulary * 4 * expected_rows:
                    raise ValueError("Truncated or inconsistent logit dump")
                first = np.fromfile(directory / "logits.f32", dtype="<f4", count=vocabulary)
                ids = native["choice_ids"]
                if first.size != vocabulary or len(set(ids)) != 4 or not all(0 <= token < vocabulary for token in ids):
                    raise ValueError("Invalid vocabulary row or choice IDs")
                if not np.array_equal(first[ids].astype(np.float64), np.asarray(native["choice_logits"], dtype=np.float64)):
                    raise ValueError("Choice logits disagree with unmodified dump")
                if not np.allclose(log_softmax(first)[ids], native["choice_logprobs"], atol=1e-8, rtol=0):
                    raise ValueError("Choice full-vocabulary probabilities disagree")
                choice = "ABCD"[int(np.argmax(native["choice_logits"]))]
                match = re.match(r"^\s*(?:\*\*)?([ABCD])(?:\b|[).])", text)
                free_choice = match.group(1) if match else None
                row = {"id": case["id"], "answer": case["answer"], "constrained_choice": choice,
                       "constrained_correct": choice == case["answer"], "greedy_choice": free_choice,
                       "greedy_correct": free_choice == case["answer"], "exact_single_letter": text in "ABCD" and len(text) == 1,
                       "text": text, "choice_ids": native["choice_ids"],
                       "correct_letter_full_vocab_nll": -native["choice_logprobs"]["ABCD".index(case["answer"])],
                       "actual_generated_tokens": native["actual_generated_tokens"], "early_stop": native["early_stop"],
                       "first_logit_file": str(directory / "logits.f32"), "vocab_size": native["vocab_size"]}
                variants[label]["cases"].append(row)
        comparisons = []
        for left, right in zip(variants["reference"]["cases"], variants["candidate"]["cases"]):
            if left["id"] != right["id"] or left["choice_ids"] != right["choice_ids"] or left["vocab_size"] != right["vocab_size"]:
                raise ValueError("Quality inputs/tokenization mismatch")
            a = np.fromfile(left["first_logit_file"], dtype="<f4", count=left["vocab_size"])
            b = np.fromfile(right["first_logit_file"], dtype="<f4", count=right["vocab_size"])
            if a.size != left["vocab_size"] or b.size != right["vocab_size"]:
                raise ValueError("Incomplete first decision row")
            lp, lq = log_softmax(a), log_softmax(b)
            kl = float((np.exp(lp) * (lp - lq)).sum())
            comparisons.append({"id": left["id"], "conditional_first_decision_kl_nats": kl,
                                "correct_letter_nll_delta": right["correct_letter_full_vocab_nll"] - left["correct_letter_full_vocab_nll"]})
        for variant in variants.values():
            rows = variant["cases"]
            variant["constrained_accuracy"] = sum(x["constrained_correct"] for x in rows) / len(rows)
            variant["greedy_accuracy"] = sum(x["greedy_correct"] for x in rows) / len(rows)
            variant["single_letter_compliance"] = sum(x["exact_single_letter"] for x in rows) / len(rows)
            variant["mean_correct_letter_nll"] = sum(x["correct_letter_full_vocab_nll"] for x in rows) / len(rows)
    report = {"cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(), "cases": cases,
              "native_binary_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "prompt_format": "Gemma 4 canonical text-only/no-thinking turn; native tokenizer, no approximation of routing",
              "context": 4096, "cache_mib": args.cache_mib, "pool": args.pool,
              "variants": variants, "comparisons": comparisons, "commands": commands,
              "mean_conditional_first_decision_kl_nats": sum(x["conditional_first_decision_kl_nats"] for x in comparisons) / len(comparisons),
              "limitations": ["Ten fixed, authored multiple-choice questions: a small diagnostic, not a broad quality benchmark.",
                              "Full-vocabulary conditional KL covers the first answer decision on the same exact input prefix, not perplexity.",
                              "Constrained-choice ranking and actual greedy answer/compliance are reported separately.",
                              "Candidate is a lossy requantization of Q4_0, not an original BF16 quantization.",
                              "Quality-run timings are not mixed into architecture throughput benchmarks."]}
    (output / "quality-results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: {n:v[n] for n in ('bytes','constrained_accuracy','greedy_accuracy','single_letter_compliance','mean_correct_letter_nll')}
                      for k,v in variants.items()}, indent=2))

if __name__ == "__main__":
    main()
