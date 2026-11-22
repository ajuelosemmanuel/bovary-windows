# Evaluation and evidence

`benchmarks/routing_cases.json` is a versioned, authored synthetic diagnostic with easy, ambiguous, malformed-output-sensitive and demanding code/reasoning cases. It is not representative held-out production evidence. The original 100-case/Wilson lower-bound policy is unchanged. A small passing diagnostic cannot qualify a workload.

Current semantics are `validators-v2`: JSON field expectations require matching types recursively, including nested objects and arrays. Prior `validators-v1` artifacts remain historical records and are rejected by the current evidence importer. Preflight-blocked cases remain in the denominator with zero backend attempts; actual attempts require retained snapshots.

Invalid validator configuration is rejected before any local or commercial attempt. It never becomes an output validation failure that could trigger paid escalation. Native failures after actual worker creation retain partial-response/error attempt snapshots; missing models, lock contention and other preflight rejections have zero backend attempts.

Local output limits are explicit evidence. The default numeric cap is 128 tokens; a local API request may use `output_tokens=None` only with a deadline and no escalation. CLI Until finished is bounded by remaining context, deadline and cancellation. The evaluation CLI still takes a positive numeric `--tokens` cap; it does not silently change a benchmark's output allowance.

A native `finish_reason` of `token_limit` or `context_limit` produces `status="truncated"`. Online task validation is `not_checked`, and truncation cannot trigger commercial escalation or count as an accepted/correct completion. Retain the partial text, actual attempt, observed token usage, effective `generation_limit` and stop reason. A natural `eos` end may proceed to normal validation; it is not a correctness attestation. Truncations remain in the denominator, so changing output limits changes the evaluated operating configuration.

Run a local diagnostic explicitly (this schedules actual CPU inference and may take several minutes):

```powershell
$env:PYTHONPATH = "$PWD/python"
python -m bovary.evaluation --dataset benchmarks/routing_cases.json --output work/evaluations/new-cpu-diagnostic --scenario local --deadline-s 1800
```

The output directory must be new. Existing results are never overwritten. Only the lead operator should schedule real model runs, serially through the shared inference lock. `run_evaluation(workbench, dataset, request_template, output_dir, scenarios=...)` also accepts injected fake adapters for CI. Run `python -m unittest discover -s tests -p 'test_runtime*.py' -v` for that zero-network diagnostic. Mocked tests verify control behavior; they are not model-quality results.

The complete paired scenarios are `local`, `commercial`, `routing` and `escalation`. They use identical cases and independent expected-value rubrics, count every case/error/refusal/truncation, and retain every attempt. `routing` executes the shared offline planner's provisional recommendation for evaluation only; it does not enable product Auto. Commercial scenarios remain blocked by default; an explicit data/account/spending authorization and approved provider config are required before live execution. A blocked case remains in the denominator. A bounded ordinary-code baseline runs where a case declares a deterministic transformation.

`attempts.jsonl` is flushed case by case. `evaluation.json` retains the dataset/rubric snapshot, content hashes, policy identity, raw result metadata, regraded summary and baselines. These files may contain private prompts/answers; they stay under ignored `work/` unless separately sanitized for publication. Hashes establish identity and integrity, not measurement truth or representativeness. The evidence importer recomputes grades from retained responses and refuses inconsistent summaries, missing cases, duplicate cases, stale validator versions or changed prompt/dataset hashes.

Verify an artifact and write a new metadata-only integrity report:

```powershell
python -m bovary.evidence work/evaluations/new-cpu-diagnostic/evaluation.json --output work/evaluations/new-cpu-diagnostic/integrity.json
```

The report explicitly marks measurement truth as unreviewed and Auto as disabled. Importing is not a representativeness or production qualification attestation.

Summaries report success and false acceptance separately, median/p95 end-to-end latency, TTFT, cold startup and warm observation counts, attempt/fallback counts, actual token categories when known, and cost per successful task only when all costs are known. Fewer than 100 latency observations receives an explicit unstable-p95 warning. The local model's startup is separate from warm serving; a restarted process is not a cold OS file-cache test. No electricity cost is inferred from unmeasured wattage. Provider usage details include cached/reasoning categories when returned; missing costs and usage remain null. Local/commercial paired evaluation must use the intended model, quantization, profile, prompt and validator versions.

Task costs are recomputed from explicit attempt snapshots and validation costs. An outer result claiming zero cannot conceal unknown local or validation computation. Provider API cost upper bounds remain visible on individual attempts; they do not establish a measured total task cost. Evidence rejects negative/nonfinite costs or durations, invalid token categories and token details exceeding their parent usage. Regrading does not authenticate that a measurement actually occurred.

Auto execution is unconditionally disabled in this alpha, including when callers supply a passing policy summary, forged identities or a trusted artifact digest. The importer is a review tool, not execution authority. v0.3 still needs binding to the effective native executable/model/runtime configuration plus reviewed representative held-out paired evidence showing quality and a predeclared operating benefit. The diagnostic qualification checker sketches those content checks but cannot establish actual runtime identity. Composite automatic escalation additionally requires separate evaluation. Manual local/commercial selection and explicit one-step validation escalation are available under runtime gates.

Future DSL-versus-concise-natural-language experiments can supply a different prompt representation while retaining the same dataset IDs, validators, paired scenarios, token accounting and end-to-end attempts. No custom DSL is introduced here.
