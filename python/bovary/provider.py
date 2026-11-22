"""Opt-in OpenAI Responses adapter. Construction and imports perform no IO.

No automatic HTTP retry: a lost response may already have incurred a charge.
"""
from dataclasses import asdict, dataclass
import hashlib
import http.client
import json
import os
from pathlib import Path
import threading
import time

from .contract import Event, ExecutionError, Result, _finite_number


@dataclass(frozen=True)
class ProviderConfig:
    enabled: bool = False
    account: str = ""
    permitted_data: tuple = ("public",)
    session_cap_usd: float = 0.0
    model: str = "gpt-6.1-sol"
    configuration: str = "responses-v1"
    context_tokens: int = 1050000
    max_output_tokens: int = 32768
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    pricing_version: str = ""
    timeout_s: float = 30.0
    # Capabilities are deployment assumptions until independently confirmed.
    capabilities_checked: str = "2026-11-14"

    @classmethod
    def from_json(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("Provider configuration must be an object")
        return cls(**value)

    def validate(self):
        if type(self.enabled) is not bool:
            raise ExecutionError("provider_config", "enabled must be boolean")
        for name in ("account", "model", "configuration", "pricing_version", "capabilities_checked"):
            if not isinstance(getattr(self, name), str):
                raise ExecutionError("provider_config", f"{name} must be text")
        if not self.model.strip() or not self.configuration.strip():
            raise ExecutionError("provider_config", "Model and configuration identity must be nonempty")
        for name in ("session_cap_usd", "timeout_s"):
            value = getattr(self, name)
            if not _finite_number(value) or value < 0:
                raise ExecutionError("provider_config", f"Invalid {name}")
        for name in ("input_usd_per_million", "output_usd_per_million"):
            value = getattr(self, name)
            if value is not None and (not _finite_number(value) or value < 0):
                raise ExecutionError("provider_config", f"Invalid {name}")
        if not 0 < self.timeout_s <= 300:
            raise ExecutionError("provider_config", "timeout_s must be positive and at most 300")
        if not isinstance(self.permitted_data, (list, tuple)) or not self.permitted_data or any(not isinstance(x, str) or not x for x in self.permitted_data):
            raise ExecutionError("provider_config", "Explicit permitted data classes required")
        for name in ("context_tokens", "max_output_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ExecutionError("provider_config", f"Invalid {name}")


def input_upper_bound(prompt):
    """UTF-8 bytes bound text tokens conservatively; reserve protocol overhead.

    This rejects some fitting prompts. It makes no outbound token-count request.
    Runtime rejects a reported usage exceeding the reservation.
    """
    return len(prompt.encode("utf-8")) + 256


class ResponsesTransport:
    """Fixed HTTPS origin; no arbitrary endpoints, redirects or prompt logging."""

    def events(self, payload, *, key, timeout, cancel):
        connection = http.client.HTTPSConnection("api.openai.com", timeout=timeout)
        finished = threading.Event()
        interrupted = threading.Event()
        deadline = time.monotonic() + timeout

        def watch():
            while not finished.wait(.05):
                if (cancel is not None and cancel.is_set()) or time.monotonic() >= deadline:
                    interrupted.set()
                    sock = connection.sock
                    if sock is not None:
                        try:
                            import socket
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    connection.close()
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            connection.request("POST", "/v1/responses", json.dumps(payload).encode("utf-8"),
                               {"Authorization": "Bearer " + key, "Content-Type": "application/json", "Accept": "text/event-stream"})
            response = connection.getresponse()
            if response.status != 200:
                # Never expose a provider body that may echo private content.
                raise ExecutionError("rate_limit" if response.status == 429 else "provider_http",
                                     f"Provider returned HTTP {response.status}; no retry was attempted",
                                     uncertain_billing=response.status >= 500 or response.status in (408, 409))
            data = []
            size = 0
            while True:
                if interrupted.is_set():
                    raise ExecutionError("cancelled" if cancel is not None and cancel.is_set() else "deadline", "Commercial stream interrupted", uncertain_billing=True)
                line = response.readline(1048577)
                if not line:
                    break
                size += len(line)
                if len(line) > 1048576 or size > 4194304:
                    raise ExecutionError("provider_protocol", "Provider event exceeded bounds", uncertain_billing=True)
                line = line.decode("utf-8").rstrip("\r\n")
                if not line:
                    if data:
                        raw = "\n".join(data)
                        data, size = [], 0
                        if raw != "[DONE]":
                            yield json.loads(raw)
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
            if data:
                yield json.loads("\n".join(data))
        except ExecutionError:
            raise
        except (OSError, ValueError, http.client.HTTPException) as error:
            code = "cancelled" if cancel is not None and cancel.is_set() else "deadline" if time.monotonic() >= deadline else "provider_transport"
            raise ExecutionError(code, "Commercial connection failed; billing may be uncertain", uncertain_billing=True) from error
        finally:
            finished.set()
            connection.close()
            watcher.join(timeout=1)


class OpenAIAdapter:
    def __init__(self, config=None, transport=None):
        self.config = config or ProviderConfig()
        self.config.validate()
        self.transport = transport or ResponsesTransport()
        self._lock = threading.Lock()
        self._committed = 0.0
        self._disabled = False

    @property
    def committed_usd(self):
        """Known charges plus reservations with unknown reconciliation."""
        with self._lock:
            return self._committed

    def execute(self, request, on_event=None, cancel=None):
        request.check()
        if request.output_tokens is None:
            raise ExecutionError('invalid_request', 'Commercial inference requires a numeric output limit')
        cfg = self.config
        if request.offline or request.private or "commercial" not in request.allowed_destinations:
            raise ExecutionError("privacy", "Commercial inference forbidden by request privacy/destinations")
        if not cfg.enabled or not cfg.account.strip():
            raise ExecutionError("live_disabled", "Live provider authorization is disabled or no approved account is named")
        if request.identity.get("data_class") not in cfg.permitted_data:
            raise ExecutionError("data_forbidden", "Request data class is outside provider authorization")
        if cancel is not None and cancel.is_set():
            raise ExecutionError("cancelled", "Cancelled before provider request")
        if request.deadline is None:
            raise ExecutionError("deadline_required", "Commercial inference requires a bounded deadline")
        if cfg.input_usd_per_million is None or cfg.output_usd_per_million is None or not cfg.pricing_version:
            raise ExecutionError("pricing_unknown", "Set checked deployment pricing before paid execution")
        bound = input_upper_bound(request.prompt)
        if bound + request.output_tokens > min(request.context_tokens, cfg.context_tokens) or request.output_tokens > cfg.max_output_tokens:
            raise ExecutionError("context", "Conservative input plus output exceeds configured provider limits")
        reservation = (bound * cfg.input_usd_per_million + request.output_tokens * cfg.output_usd_per_million) / 1e6
        if request.max_cost_usd <= 0 or reservation > request.max_cost_usd:
            raise ExecutionError("budget", "Conservative reservation exceeds request spending limit")
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ExecutionError("credentials", "OPENAI_API_KEY is missing; no request was sent")
        with self._lock:
            if self._disabled or self.config is not cfg:
                raise ExecutionError("live_disabled", "Provider session is disabled pending billing reconciliation")
            if cfg.session_cap_usd <= 0 or self._committed + reservation > cfg.session_cap_usd:
                raise ExecutionError("budget", "Provider session spending cap exhausted")
            self._committed += reservation
        start = time.monotonic()
        timeout = min(cfg.timeout_s, request.deadline - start)
        if timeout <= 0:
            with self._lock:
                self._committed -= reservation
            raise ExecutionError("deadline", "Deadline expired before provider request")
        result = Result(request.request_id, "commercial", model=cfg.model, configuration=cfg.configuration,
                        cost_status="reserved", usage={"reservation_usd": reservation, "pricing_version": cfg.pricing_version})
        terminal = None
        ttft = None
        events = None
        streamed_bytes = 0
        # Client memory bound, not a token-count or provider billing guarantee.
        max_stream_bytes = min(8 * 1024 * 1024, request.output_tokens * 256 + 4096)
        try:
            payload = {"model": cfg.model, "input": request.prompt, "max_output_tokens": request.output_tokens,
                       "stream": True, "store": False, "tools": []}
            events = self.transport.events(payload, key=key, timeout=timeout, cancel=cancel)
            for event in events:
                if cancel is not None and cancel.is_set():
                    raise ExecutionError("cancelled", "Commercial request cancelled", uncertain_billing=True)
                if time.monotonic() >= request.deadline:
                    raise ExecutionError("deadline", "Commercial deadline expired", uncertain_billing=True)
                kind = event.get("type")
                if kind == "response.output_text.delta":
                    delta = event.get("delta", "")
                    if not isinstance(delta, str):
                        raise ExecutionError("provider_protocol", "Malformed text delta", uncertain_billing=True)
                    streamed_bytes += len(delta.encode("utf-8"))
                    if streamed_bytes > max_stream_bytes:
                        with self._lock:
                            self._disabled = True
                        raise ExecutionError("provider_output_limit", "Commercial text exceeded the client output bound; session disabled pending reconciliation", uncertain_billing=True)
                    if ttft is None:
                        ttft = time.monotonic() - start
                    result.text += delta
                    if on_event:
                        on_event(Event("text", request.request_id, text=delta))
                elif kind in ("response.completed", "response.incomplete", "response.failed"):
                    terminal = event.get("response", {})
                    if kind != "response.completed":
                        result.status = "needs_review"
                        result.error = {"code": "provider_incomplete", "message": "Provider did not complete the answer"}
                elif kind in ("response.refusal.delta", "response.refusal.done"):
                    result.status = "needs_review"
                    result.error = {"code": "provider_refusal", "message": "Provider refused the request"}
                elif kind in ("error", "response.error"):
                    raise ExecutionError("provider_error", "Provider returned an error event", uncertain_billing=True)
            if terminal is None:
                raise ExecutionError("provider_protocol", "Stream ended without a terminal response", uncertain_billing=True)
            usage = terminal.get("usage")
            if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")):
                result.cost_status = "unknown_reserved"
            elif usage["input_tokens"] > bound or usage["output_tokens"] > request.output_tokens:
                result.cost_status = "unknown_reserved"
                result.status = "needs_review"
                result.error = {"code": "usage_exceeds_reservation", "message": "Reported usage exceeds conservative limits; disable paid execution and reconcile account billing"}
                with self._lock:
                    self._disabled = True
                    self.config = ProviderConfig()
            else:
                result.usage.update(usage)
                result.cost_usd = (usage["input_tokens"] * cfg.input_usd_per_million + usage["output_tokens"] * cfg.output_usd_per_million) / 1e6
                # Full input rate includes cached tokens; no discount is assumed.
                result.cost_status = "upper_bound_reported_usage"
                with self._lock:
                    self._committed -= reservation - result.cost_usd
            # A generator can finish without yielding another event after an
            # interruption. Keep reconciled usage while reporting that final
            # request state truthfully, including for direct adapter callers.
            if cancel is not None and cancel.is_set():
                raise ExecutionError("cancelled", "Commercial request cancelled", uncertain_billing=result.cost_usd is None)
            if time.monotonic() >= request.deadline:
                raise ExecutionError("deadline", "Commercial deadline expired", uncertain_billing=result.cost_usd is None)
            return result
        except ExecutionError as error:
            # A known rejection before inference releases its allowance. All
            # uncertain failures retain it, preventing a retry overspend.
            if result.cost_usd is None:
                if not error.uncertain_billing:
                    with self._lock:
                        self._committed -= reservation
                result.cost_status = "unknown_reserved" if error.uncertain_billing else "not_billed"
            result.status = "cancelled" if error.code == "cancelled" else "failed"
            result.error = {"code": error.code, "message": str(error), "uncertain_billing": error.uncertain_billing}
            return result
        except Exception as error:
            result.status = "failed"
            result.error = {"code": "provider_protocol", "message": "Unexpected provider failure", "uncertain_billing": True}
            result.cost_status = "unknown_reserved"
            return result
        finally:
            if events is not None and hasattr(events, "close"):
                events.close()
            result.timing = {"end_to_end_s": time.monotonic() - start, "ttft_s": ttft, "cold": False}
            result.usage["config_sha256"] = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()
