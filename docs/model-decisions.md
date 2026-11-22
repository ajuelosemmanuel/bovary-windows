# Choosing local or commercial inference

The packaged workbench uses this planner through `bovary.policy`.
Runtime Suggest binds its displayed configuration to the selected profile and
ignores unverified planner summary entries. Auto execution is disabled in this
alpha even when a supplied planner summary says `supported`. The evidence importer
regrades retained records; it does not grant execution authority. See
[architecture](architecture.md) and [evaluation](evaluation.md). The historical
summary format below remains useful for offline planning, not runtime trust.

Run `.\.venv\Scripts\python.exe -m bovary.policy` before choosing an inference backend. This is an offline recommendation tool; it does not execute prompts or call an API. Task complexity is the starting signal. Workload evaluations can replace that initial assumption with evidence.

The local model is Gemma 4 26B-A4B Q4_0: a mixture-of-experts checkpoint with a smaller active parameter count, rather than a tiny dense model. The commercial example is `gpt-6.1-sol`; the policy can name another model. Each has a configuration identity because quantization, context, prompting, and reasoning settings affect the comparison.

## Initial complexity policy

| Task or condition | Initial recommendation |
|---|---|
| Extraction, classification, formatting | Local; provisional until evaluated |
| Summarization, translation, general requests | Medium complexity; commercial unless local has passing workload evidence |
| Coding, reasoning, multiple reasoning steps, high consequences | High complexity; commercial unless local has passing workload evidence |
| Local output fails a task-specific validator | Recommend commercial; execution requires explicit escalation permission |
| Cloud disabled and local does not qualify | Block and request review or evaluation |

These are conservative, editable task defaults, not an automatic understanding of arbitrary prompt difficulty. A one-line code transformation might be low complexity; ambiguous extraction across many documents might be high. Set `--complexity low|medium|high` to describe the actual task. `--multistep` and `--high-stakes` raise it to high even when a lower complexity is explicitly supplied. High-consequence answers also require human review.

CPU/GPU selection is separate from the local/commercial recommendation. CPU remains the default, GPU requires an explicit profile, and a recommendation grants no device or network permission.

The policy lives in `benchmarks/decision_policy.json`. Its initial `local_max_complexity` is `low`. Raising that ceiling is a provisional policy choice, not quality evidence. No commercial API access or performance is presumed to have been tested.

## Examples

Run these commands from the project root after the README has created and installed `.venv`; activation is not required:

```powershell
# Simple, bounded extraction; local recommendation.
.\.venv\Scripts\python.exe -m bovary.policy --task extract --input-tokens 500 --output-tokens 128

# Demanding reasoning; permit a commercial recommendation.
.\.venv\Scripts\python.exe -m bovary.policy --task reason --multistep --allow-cloud --input-tokens 1500 --output-tokens 500

# Escalate a local answer that failed your application validator.
.\.venv\Scripts\python.exe -m bovary.policy --task extract --validation-failed --allow-cloud

# Confidential work: cloud cannot be recommended, even with --allow-cloud.
.\.venv\Scripts\python.exe -m bovary.policy --task reason --allow-cloud --offline

# Require measurements before choosing within a budget and latency limit.
.\.venv\Scripts\python.exe -m bovary.policy --task summarize --allow-cloud --workload support-summary-v1 --monthly-requests 1000 --max-monthly-cost-usd 20 --max-latency-s 30

# Retain the request and policy hash in a new audit file.
.\.venv\Scripts\python.exe -m bovary.policy --task extract --output outputs/my-model-decision.json
```

The JSON response includes the recommendation, effective complexity, model/configuration, reasons, both candidate assessments, quality evidence and estimated monthly costs. Status means:

- `provisional`: complexity rules select a candidate whose fitness still needs evaluation, or input size is unspecified.
- `supported`: the chosen candidate clears the configured workload quality threshold and any supplied constraints. This does not certify broader tasks or establish an economic advantage when costs are unknown.
- `benchmark_required`: missing budget/latency measurements or incompatible comparison datasets prevent a choice.
- `blocked`: neither candidate qualifies under current restrictions.

The exit code is 0 for a recommendation, including provisional ones, and 2 for no recommendation or invalid inputs. Downstream execution should inspect `status` as well as the exit code. Cloud recommendations require `--allow-cloud`; this flag permits recommendations only. API execution and API account access remain separate. `--offline` takes precedence.

## Replace heuristics with workload evidence

1. Define a workload and a grading rubric: for example, invoice extraction with schema validation and field accuracy, or code changes judged by independent tests. Include easy, ambiguous, and failure cases. Use held-out cases rather than examples used to tune the prompt.
2. Evaluate each candidate on the same cases and rubric using its intended configuration. Count every attempted case, including errors, refusals, malformed answers and truncated outputs. Record end-to-end latency, including model loading, queueing, network time and retries as applicable. Define a successful answer before comparing models.
3. Grade representative cases separately for each complexity band. Add an entry to that model's `evaluations` array. The following illustrates the schema with fictional measurements; do not treat it as project evidence:

```json
{
  "workload": "invoice-extraction-v1",
  "complexity": "low",
  "model": "gemma-4-26B-A4B-it-Q4_0",
  "configuration": "balanced-4k",
  "dataset_id": "held-out-invoices-v1-rubric-v1",
  "source": "outputs/your-graded-invoice-results.json",
  "representative": true,
  "cases": 200,
  "correct": 198,
  "p95_latency_s": 12.4
}
```

Use `--workload invoice-extraction-v1` for that evidence. The default workload is `default`. Model, configuration, workload and complexity must match exactly. Use `dataset_id` to identify both the cases and grading rubric, ideally their content hashes. `source` points to retained raw evidence; the planner trusts supplied summaries and does not grade responses or verify that artifact. It rejects duplicate matching evaluations instead of choosing one silently.

The default minimum is 100 representative cases and a **95% Wilson interval lower bound of at least 90% success**. This conservative gate accounts for the size of the supplied sample; it does not prove that authored cases represent production. Insufficient or explicitly unrepresentative evidence remains unknown. Sufficient evidence below the threshold excludes the candidate, including a local model on a simple task. Strong evidence can qualify local inference above its initial complexity ceiling. A local validation failure still recommends escalation; actual commercial execution requires explicit permission and all provider/privacy/budget gates.

When both candidates pass, their dataset/rubric identity must match before selection. If comparable costs are known, choose the lower total monthly cost; otherwise prefer local and report that costs are missing. When only one candidate passes, prefer its evidence over the other candidate's provisional complexity assessment. Re-evaluate and update identities after changing the model, quantization, launcher preset, hardware, prompts, workload or commercial reasoning settings. Recheck results as production behavior changes.

The existing `outputs/quality/quality-results.json` contains ten authored multiple-choice cases comparing Q4 and mixed weights. It is neither a commercial-model comparison nor representative evidence for coding, document processing, translation or other application work. The planner deliberately does not promote it into a workload qualification. Likewise, architecture throughput measurements are not task-success scores or commercial end-to-end latency measurements.

## Cost, latency and context constraints

Populate each model's `cost_per_request_usd` and `monthly_fixed_cost_usd` for the workload and complexity being planned. Keep separate policy files when workloads have different cost profiles. Unknown values stay `null`; explicitly enter zero only when justified.

```
monthly cost = monthly fixed cost + monthly requests * average cost per request
```

For local inference, include incremental electricity and any allocated hardware, maintenance and setup costs. For commercial inference, include input tokens, billed output/reasoning tokens, tools and retries under your actual pricing tier. Include failed attempts in the average. If the task requires a local attempt followed by a commercial fallback, include that first attempt's cost and latency in the commercial escalation scenario. The planner compares the supplied scenarios; it does not infer fallback rates, tokenization or retry costs. It reports infrastructure cost, not the business cost of incorrect answers.

The initial policy has no dollar or latency estimates because this project has not measured comparable commercial workloads or local power consumption. `--max-monthly-cost-usd` excludes scenarios over budget and requires both cost fields. `--max-latency-s` requires representative p95 end-to-end latency. Constraints with missing evidence prevent that candidate from being recommended; the planner can still select another candidate with adequate evidence.

`--input-tokens` must include instructions, conversation history, retrieved content and turn overhead under the candidate tokenizer. Use a conservative count covering both tokenizers. Input plus output must fit the configured context capacity. Omitting input size leaves context fit unchecked and the result provisional. Configure actual limits for your chosen deployment; the commercial context ceiling is not an output-token limit or a guarantee of API availability. This planner does not replace runtime token/output-limit checks.

## Sources

The process follows the [official OpenAI model-selection guidance](https://developers.openai.com/api/docs/guides/model-selection): establish workload quality needs, then consider latency and cost. The [GPT-6.1 Sol model documentation](https://developers.openai.com/api/docs/models/gpt-6.1-sol), checked on 14 November 2026, supplies the example commercial identifier and context ceiling. Actual account access, prices and measurements must be checked for the deployment. No commercial superiority on this project's tasks is inferred from the model catalog.
