"""Export selected historical CPU measurements; never include prompts or logs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "outputs/cpu-only/benchmark/results.json"
    build_path = ROOT / "outputs/build-cpu/build-manifest.json"
    build = json.loads(build_path.read_text())
    rows = json.loads(source.read_text())
    records = []
    for row in rows:
        metrics = row["engine_metrics"]
        memory = row["memory_summary"]
        records.append({"run_id": row["run_id"], "repetition": row["repetition"], "return_code": row["return_code"],
                        "context_tokens": row["context"], "output_tokens": row["actual_generated_tokens"],
                        "prompt_sha256": row["prompt_sha256"], "end_to_end_s": row["end_to_end_seconds"],
                        "ttft_internal_s": metrics.get("native_ttft_process_internal_s"),
                        "prompt_tokens_per_s": metrics.get("engine_prompt_tokens_per_second"),
                        "decode_tokens_per_s": metrics.get("native_decode_tokens_per_second"),
                        "peak_process_working_set_bytes": memory["process_peak_working_set_bytes"]["max"],
                        "sampled_peak_private_commit_bytes": memory["process_private_commit_bytes"]["max"]})
    exported = {"schema_version": 1, "evidence_kind": "historical_local_measurement", "auto_qualification": False,
                "source_relative_path": "outputs/cpu-only/benchmark/results.json", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "hardware": {"cpu": "Intel Core i7-12700H", "physical_ram_bytes": 34029162496,
                             "os": "Windows 11 x64", "ssd": "SK hynix BC711 NVMe"},
                "configuration": {"model_sha256": "d208665ab1cd3a69f7a9a4bc59430e8448c8093d9b06334f566ac59d6d504a03",
                                  "model_bytes": 14618145824, "engine_revision": build["engine_commit"],
                                  "binary_sha256": build["binary"]["sha256"], "infer_source_sha256": build["application_source_sha256"],
                                  "device": "cpu-only", "threads": 8, "prompt_batch": 32, "logit_dumping": True},
                "limitations": ["Three short authored cases/repetitions; no production quality claim and no useful p95 estimate.",
                                "Fresh processes, uncontrolled file cache; no cache purge or CPU/GPU paired rerun.",
                                "Historic one-shot executable, not a remeasurement of the persistent worker.",
                                "Working set, private commitment, virtual mappings and OS cache overlap and are not additive.",
                                "No GPU initialized or queried in CPU runs; no power/electricity cost measurement."],
                "records": records}
    output = ROOT / "reports/historical-cpu.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(exported, target, indent=2)
        target.write("\n")
    print(output)


if __name__ == "__main__":
    main()
