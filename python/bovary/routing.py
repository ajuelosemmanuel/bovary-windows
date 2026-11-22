"""One execution interface for GUI, CLI and paired evaluations."""
from dataclasses import replace
import json
from pathlib import Path
import time

from .contract import Event, ExecutionError, Result
from .policy import decide
from .provider import input_upper_bound
from .validators import validate, validate_specification


class Workbench:
    def __init__(self, local=None, commercial=None, policy=None, qualification=None,
                 trusted_evidence_sha256=()):
        self.local, self.commercial = local, commercial
        self.policy = policy or json.loads((Path(__file__).parent / "data/decision_policy.json").read_text())
        self.qualification = qualification
        self.trusted_evidence_sha256 = tuple(trusted_evidence_sha256)

    def unload(self):
        if self.local is not None:
            self.local.unload()

    def suggest(self, request, validation_failed=False):
        # The heuristic is shared with the historical offline planner. Privacy
        # and actual backend identities remain runtime gates below.
        policy = json.loads(json.dumps(self.policy))
        local = policy["models"]["local"]
        local["context_tokens"] = request.context_tokens
        local["configuration"] = request.local_configuration
        commercial = policy["models"]["commercial"]
        config = getattr(self.commercial, "config", None)
        if config is not None:
            commercial["model"] = config.model
            commercial["configuration"] = f"{config.configuration}-ctx{min(request.context_tokens, config.context_tokens)}-pricing{config.pricing_version or 'unknown'}"
            commercial["context_tokens"] = min(request.context_tokens, config.context_tokens)
        else:
            commercial["configuration"] = "unconfigured-commercial"
        # The historical planner accepts supplied summaries. Runtime Suggest
        # must not label those as verified workload evidence.
        local["evaluations"] = []
        commercial["evaluations"] = []
        result = decide(policy, task=request.task, workload=request.workload,
                        allow_cloud="commercial" in request.allowed_destinations,
                        offline=request.offline or request.private,
                        input_tokens=input_upper_bound(request.prompt), output_tokens=request.output_tokens,
                        validation_failed=validation_failed)
        result["reasons"].append("Runtime configuration identities replace historical policy defaults; unverified summary evaluations are excluded")
        if "local" not in request.allowed_destinations and result["recommendation"] == "local":
            result["recommendation"] = None
            result["status"] = "blocked"
            result["reasons"].append("The request disallows the recommended local destination")
        return result

    def execute(self, request, on_event=None, cancel=None):
        start = time.monotonic()
        attempts = []
        reasons = []
        destination = request.mode
        try:
            request.check()
            validate_specification(request.validator)
            if request.allow_escalation and request.validator is None:
                raise ExecutionError('invalid_validator', 'Explicit escalation requires a task validator')
            if cancel is not None and cancel.is_set():
                raise ExecutionError("cancelled", "Request was cancelled before execution")
            if request.mode in ("suggest", "auto"):
                decision = self.suggest(request)
                reasons.extend(decision["reasons"])
                if request.mode == "suggest":
                    return Result(request.request_id, decision["recommendation"] or "none", status="suggested",
                                  model=decision.get("model") or "", reasons=reasons,
                                  configuration=decision["candidates"][decision["recommendation"]]["configuration"] if decision["recommendation"] else "",
                                  validation={"status": "not_run", "policy_status": decision["status"], "decision": decision},
                                  timing={"end_to_end_s": time.monotonic() - start})
                destination = decision["recommendation"]
                # Retained evidence can be inspected, but execution must also
                # bind the actual native binary/model/configuration. Caller
                # identity strings and reviewed summaries cannot supply that.
                raise ExecutionError("auto_disabled", "Auto execution is disabled in this alpha: effective runtime identity binding and representative paired evidence review remain outstanding")
            else:
                reasons.append(f"Explicit manual {destination} selection; workload qualification is not claimed")
            if destination not in request.allowed_destinations:
                raise ExecutionError("destination_forbidden", "Selected destination is not allowed")
            if destination == "commercial" and (request.offline or request.private):
                raise ExecutionError("privacy", "Privacy/offline restrictions forbid commercial execution")
            for number in range(2):
                request.check()  # refreshed before every attempt/escalation
                adapter = self.local if destination == "local" else self.commercial
                if adapter is None:
                    raise ExecutionError("backend_unavailable", f"{destination} backend is unavailable")
                if on_event:
                    on_event(Event("attempt", request.request_id, data={"destination": destination, "number": number + 1}))
                result = adapter.execute(replace(request, mode=destination), on_event=on_event, cancel=cancel)
                attempts.append(result.to_dict())
                request.check()
                if cancel is not None and cancel.is_set():
                    raise ExecutionError("cancelled", "Cancelled before task validation")
                result.validation = validate(result.text, request.validator, deadline=request.deadline, cancel=cancel) if result.status == "complete" else {"status": "not_checked", "passed": None}
                request.check()
                if cancel is not None and cancel.is_set():
                    raise ExecutionError("cancelled", "Cancelled after task validation")
                if result.status == "complete" and result.validation["passed"] is False:
                    result.status = "needs_review"
                    result.reasons.append("Task-specific validation failed")
                attempts[-1] = result.to_dict()
                if result.status not in ("complete", "needs_review") or result.error:
                    break
                if result.validation["passed"] is not False:
                    break
                if destination != "local" or number > 0 or not request.allow_escalation:
                    break
                decision = self.suggest(request, validation_failed=True)
                reasons.extend(decision["reasons"])
                if decision["recommendation"] != "commercial":
                    reasons.append("Validation failed and escalation is forbidden or unsupported")
                    break
                destination = "commercial"
                reasons.append("One explicitly permitted escalation after failed local validation")
            result.attempts = attempts
            result.reasons = reasons + result.reasons
            result.timing["end_to_end_s"] = time.monotonic() - start
            elapsed_before = 0.0
            composite_ttft = None
            for attempt in attempts:
                attempt_timing = attempt.get("timing", {})
                if attempt_timing.get("ttft_s") is not None:
                    composite_ttft = elapsed_before + attempt_timing["ttft_s"]
                    break
                elapsed_before += attempt_timing.get("end_to_end_s", 0.0)
            result.timing["ttft_s"] = composite_ttft
            if len(attempts) > 1:
                costs = [x["cost_usd"] for x in attempts]
                result.cost_usd = sum(costs) if all(x is not None for x in costs) else None
                if result.cost_usd is None:
                    result.cost_status = "unknown_total"
            return result
        except ExecutionError as error:
            if error.attempt is not None:
                attempts.append(error.attempt)
            if error.code in ('cancelled', 'deadline') and any(attempt.get('destination') == 'local' for attempt in attempts):
                # Validation can expire after native generation has returned.
                # Unload only after this request used local execution. Another
                # caller may now own the adapter; its busy guard must prevail.
                try:
                    self.unload()
                except ExecutionError as cleanup_error:
                    reasons.append(f"Local cleanup deferred ({cleanup_error.code}); original interruption preserved")
            if attempts and not attempts[-1].get("validation"):
                attempts[-1]["validation"] = {"status": "interrupted", "passed": None, "error": error.code}
            partial_text = attempts[0].get('text', '') if len(attempts) == 1 and attempts[0].get('destination') == 'local' else ''
            return Result(request.request_id, destination or "none", text=partial_text, status="cancelled" if error.code == "cancelled" else "failed",
                          reasons=reasons, attempts=attempts, timing={"end_to_end_s": time.monotonic() - start},
                          cost_status="unknown_total" if attempts else "unknown",
                          error={"code": error.code, "message": str(error), "uncertain_billing": error.uncertain_billing})
