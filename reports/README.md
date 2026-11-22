# Curated measurements

Report snapshots are distinct. `worker-validation.json` and `release-verification.json` describe the retained 0.2.0a1/v3 checkpoint, including its 86-test suite and a1 wheel. `continuation-validation.json` describes 0.2.0a2, its 125-test suite and real isolated consumer API checks. The native CPU binary is byte-identical across those historical snapshots; the Bovary-only alpha builds a new binary. Earlier raw measurements and finalized v3 artifacts remain unchanged; a2 software checks do not turn those authored diagnostics into production qualification.

`historical-cpu.json` selectively exports three actual CPU-only runs from 2026-11-14, with source/model/binary identities and relative evidence paths. It includes no prompts, answers, raw counters or personal paths. Recreate the export from the retained private workspace with `python scripts/export_public_benchmark.py`; existing exports are never overwritten.

Historical short-prompt medians: 3.245 decode tokens/s, 4.183 prompt tokens/s, 8.428 seconds internal startup-to-first-token, 9.394 GiB process working set and 1.141 GiB sampled private commitment. These are different accounting views; do not add them. Three observations do not provide a useful p95 or qualify routing quality. Timing varied considerably and full logit dumping was enabled. No electricity/cost or commercial comparison is inferred. This is the previous one-shot CPU executable, not a performance claim for the new worker.

Earlier approximately 0.5 GiB streaming process residency was one GPU profile's working set; it was never total machine memory. Keep dedicated/shared GPU allocations, mapped virtual bytes, resident RAM, private commitment and OS cache distinct. No private raw research output is included in this public source boundary.


`bovary-release-preparation.json` records the **0.2.0a12** candidate, including its 226-test result, adopted MIT license and inherited Unicode CPU diagnostics. Each report applies to the version and artifacts identified inside it. The release assessment describes subsequent a13 verification. [Release assessment](../docs/implementation-assessment.md)
