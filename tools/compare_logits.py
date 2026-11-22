#!/usr/bin/env python3
"""Compare instrumented inference run directories without loading all logits."""
from __future__ import annotations

from array import array
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()


def compare(reference, candidate, atol=1e-4, rtol=1e-4, allow_context_difference=False):
    reference, candidate = Path(reference), Path(candidate)
    ref = json.loads((reference/"native-metrics.json").read_text(encoding="utf-8"))
    got = json.loads((candidate/"native-metrics.json").read_text(encoding="utf-8"))
    errors = []
    for key in ("vocab_size", "prompt_tokens", "requested_generated_tokens"):
        if ref.get(key) != got.get(key) or ref.get(key) is None:
            errors.append(f"Metric mismatch/missing: {key}: {ref.get(key)} vs {got.get(key)}")
    reference_context, candidate_context = ref.get("context"), got.get("context")
    if reference_context is None or candidate_context is None or (reference_context != candidate_context and not allow_context_difference):
        errors.append(f"Metric mismatch/missing: context: {reference_context} vs {candidate_context}")
    if (reference/"prompt.txt").exists() and (candidate/"prompt.txt").exists():
        if file_hash(reference/"prompt.txt") != file_hash(candidate/"prompt.txt"):
            errors.append("Prompt bytes differ")
    else:
        errors.append("Missing prompt.txt; same input cannot be verified")
    vocab = ref.get("vocab_size", 0)
    if not isinstance(vocab, int) or vocab < 1: raise ValueError("Invalid reference vocab_size")
    ref_path, got_path = reference/"logits.f32", candidate/"logits.f32"
    ref_size, got_size = ref_path.stat().st_size, got_path.stat().st_size
    row_size = vocab*4
    if ref_size % row_size or got_size % row_size: errors.append("Logit file is not a whole number of vocab rows")
    ref_rows, got_rows = ref_size//row_size, got_size//row_size
    if ref_rows != got_rows: errors.append(f"Different logit row counts: {ref_rows} vs {got_rows}")
    for label, metrics, rows in (("reference", ref, ref_rows), ("candidate", got, got_rows)):
        expected = metrics.get("actual_generated_tokens", 0) + int(bool(metrics.get("early_stop")))
        if rows != expected:
            errors.append(f"{label} logit rows {rows} != generated tokens plus optional EOG row {expected}")
    ref_tokens, got_tokens = ref.get("tokens", []), got.get("tokens", [])
    if len(ref_tokens) != ref.get("actual_generated_tokens") or len(got_tokens) != got.get("actual_generated_tokens"):
        errors.append("Token-array lengths do not match actual_generated_tokens")
    common_prefix = 0
    for a, b in zip(ref_tokens, got_tokens):
        if a != b: break
        common_prefix += 1
    token_match = ref_tokens == got_tokens and ref.get("early_stop") == got.get("early_stop")
    if not token_match: errors.append("Generated tokens/EOG stopping differ")
    # The decision row at first differing token has a shared input prefix. Rows
    # after it reflect different contexts and are not a numerical equivalence test.
    comparable_rows = min(ref_rows, got_rows, common_prefix+1)
    max_abs, sum_sq, count, violations, nonfinite, top1_mismatch, byte_exact_rows = 0.0, 0.0, 0, 0, 0, 0, 0
    per_row = []
    with ref_path.open("rb") as left, got_path.open("rb") as right:
        for row in range(comparable_rows):
            raw_a, raw_b = left.read(row_size), right.read(row_size)
            a, b = array("f"), array("f")
            a.frombytes(raw_a); b.frombytes(raw_b)
            if sys.byteorder != "little": a.byteswap(); b.byteswap()
            if raw_a == raw_b: byte_exact_rows += 1
            row_max, row_bad, row_nonfinite = 0.0, 0, 0
            for av, bv in zip(a, b):
                count += 1
                if not math.isfinite(av) or not math.isfinite(bv):
                    nonfinite += 1; row_nonfinite += 1
                    continue
                delta = abs(av-bv)
                sum_sq += delta*delta
                row_max = max(row_max, delta)
                if delta > atol + rtol*abs(av): violations += 1; row_bad += 1
            max_abs = max(max_abs, row_max)
            best_a = max(range(vocab), key=a.__getitem__)
            best_b = max(range(vocab), key=b.__getitem__)
            if best_a != best_b: top1_mismatch += 1
            per_row.append({"row": row, "max_absolute_error": row_max, "tolerance_violations": row_bad,
                "nonfinite_values": row_nonfinite, "reference_top1": best_a, "candidate_top1": best_b})
    if violations: errors.append(f"{violations} logits exceed atol+rtol*abs(reference)")
    if nonfinite: errors.append(f"{nonfinite} nonfinite logit pairs")
    if top1_mismatch: errors.append(f"{top1_mismatch} shared-prefix rows have different greedy top-1")
    return {"passed": not errors, "reference": str(reference.resolve()), "candidate": str(candidate.resolve()),
        "atol": atol, "rtol": rtol, "reference_rows": ref_rows, "candidate_rows": got_rows,
        "reference_context": reference_context, "candidate_context": candidate_context,
        "allow_context_difference": bool(allow_context_difference),
        "comparable_shared_prefix_rows": comparable_rows, "generated_tokens_match": token_match,
        "common_generated_token_prefix": common_prefix, "byte_exact_rows": byte_exact_rows,
        "logit_pairs": count, "max_absolute_error": max_abs,
        "root_mean_square_error": math.sqrt(sum_sq/(count-nonfinite)) if count>nonfinite else None,
        "tolerance_violations": violations, "nonfinite_pairs": nonfinite, "top1_mismatches": top1_mismatch,
        "errors": errors, "per_row": per_row,
        "limitations": ["Same model hash/build/tokenizer/generation settings must be verified separately from run metadata.",
            "Only shared-prefix decision rows are numerically comparable after token divergence.",
            "This checks memory-management equivalence on these prompts; it is not a broad model-quality evaluation."]
            + (["Different configured contexts were explicitly allowed. Equality on these shared short prefixes does not establish quality or correctness on a filled longer history."] if allow_context_difference else [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--allow-context-difference", action="store_true",
                        help="Explicitly compare different configured contexts; does not test filled-history quality")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0: raise ValueError("Tolerances must be nonnegative")
    result = compare(args.reference, args.candidate, args.atol, args.rtol,
                     allow_context_difference=args.allow_context_difference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_row"}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
