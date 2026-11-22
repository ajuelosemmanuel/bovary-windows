"""Shared execution contract; adapters never choose a different destination."""
from dataclasses import asdict, dataclass, field
import math
import time
import uuid

STREAMING_GPU_PROFILES = ("balanced", "gpu-more-vram", "gpu-cache-2g", "gpu-prefill-2g")
MAX_HOST_CACHE_MIB = 4096


class ExecutionError(RuntimeError):
    def __init__(self, code, message, *, uncertain_billing=False, attempt=None):
        super().__init__(message)
        self.code = code
        self.uncertain_billing = uncertain_billing
        self.attempt = attempt


def _finite_number(value):
    """Check JSON-style numbers without overflowing on arbitrary Python ints."""
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass
class Request:
    prompt: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task: str = "general"
    workload: str = "default"
    history: list = field(default_factory=list)
    mode: str = "local"
    allowed_destinations: tuple = ("local",)
    local_profile: str = "cpu-only"
    allow_gpu: bool = False
    offline: bool = True
    private: bool = True
    context_tokens: int = 4096
    output_tokens: int | None = 128  # None: local generation until EOG or context limit
    threads: int = 8
    deadline: float | None = None  # absolute time.monotonic() deadline
    max_cost_usd: float = 0.0
    allow_escalation: bool = False
    validator: dict | None = None
    identity: dict = field(default_factory=dict)
    cpu_prompt_batch: int | None = None
    host_cache_mib: int = 0

    @property
    def local_configuration(self):
        label = f"{self.local_profile}-ctx{self.context_tokens}-threads{self.threads}-gemma4-text-v1"
        if self.cpu_prompt_batch not in (None, 32):
            label += f"-batch{self.cpu_prompt_batch}"
        if self.host_cache_mib:
            label += f"-host{self.host_cache_mib}mib"
        return label

    def check(self):
        for key in ('allow_gpu', 'offline', 'private', 'allow_escalation'):
            if type(getattr(self, key)) is not bool:
                raise ExecutionError('invalid_request', f'{key} must be boolean')
        for key in ('request_id', 'task', 'workload', 'local_profile'):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ExecutionError('invalid_request', f'{key} must be nonempty text')
        if not isinstance(self.identity, dict) or not isinstance(self.history, list):
            raise ExecutionError('invalid_request', 'Invalid identity/history')
        if self.cpu_prompt_batch is not None and (
            self.local_profile != 'cpu-only' or type(self.cpu_prompt_batch) is not int
            or not 1 <= self.cpu_prompt_batch <= 128
        ):
            raise ExecutionError('invalid_request', 'CPU prompt batch requires the CPU profile and a size from 1 to 128')
        if (type(self.host_cache_mib) is not int or not 0 <= self.host_cache_mib <= MAX_HOST_CACHE_MIB
                or self.host_cache_mib and (self.local_profile not in STREAMING_GPU_PROFILES or self.mode != "local")):
            raise ExecutionError("invalid_request", "Host cache requires a local streaming CUDA profile and a budget from 0 to 4096 MiB")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ExecutionError("invalid_request", "Prompt must be nonempty text")
        try:
            self.prompt.encode('utf-8')
        except UnicodeEncodeError as error:
            raise ExecutionError('invalid_request', 'Prompt must contain valid Unicode text') from error
        if self.history:
            raise ExecutionError("unsupported_history", "Independent requests require explicit history flattened into the prompt")
        if self.mode not in ("local", "commercial", "suggest", "auto"):
            raise ExecutionError("invalid_request", "Unknown routing mode")
        if not isinstance(self.allowed_destinations, (tuple, list)) or not self.allowed_destinations or any(x not in ("local", "commercial") for x in self.allowed_destinations):
            raise ExecutionError("invalid_request", "Invalid allowed destinations")
        if type(self.context_tokens) is not int or not 64 <= self.context_tokens <= 262144:
            raise ExecutionError('invalid_request', 'Invalid context_tokens')
        if self.output_tokens is None:
            if self.mode != 'local' or self.allow_escalation:
                raise ExecutionError('invalid_request', 'Until finished is local-only and cannot use commercial escalation; choose a numeric output limit')
            if self.deadline is None:
                raise ExecutionError('deadline_required', 'Until finished requires a bounded deadline')
        elif type(self.output_tokens) is not int or not 1 <= self.output_tokens < self.context_tokens:
            raise ExecutionError('invalid_request', 'Invalid output_tokens')
        for key, low, high in (("threads", 1, 256),):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ExecutionError("invalid_request", f"Invalid {key}")
        if self.task not in ('general', 'extract', 'classify', 'format', 'summarize', 'translate', 'reason', 'code'):
            raise ExecutionError('invalid_request', 'Unknown task')
        if self.validator is not None and not isinstance(self.validator, dict):
            raise ExecutionError('invalid_request', 'Validator must be an object')
        if not _finite_number(self.max_cost_usd) or self.max_cost_usd < 0:
            raise ExecutionError("invalid_request", "Invalid spending limit")
        if self.deadline is not None:
            if not _finite_number(self.deadline):
                raise ExecutionError('invalid_request', 'Invalid deadline')
            if time.monotonic() >= self.deadline:
                raise ExecutionError("deadline", "Request deadline expired")
        if self.local_profile != "cpu-only" and not self.allow_gpu:
            raise ExecutionError("gpu_forbidden", "GPU profiles require explicit GPU permission")


@dataclass
class Event:
    kind: str
    request_id: str
    text: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class Result:
    request_id: str
    destination: str
    text: str = ""
    status: str = "complete"
    model: str = ""
    configuration: str = ""
    reasons: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    attempts: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    cost_usd: float | None = None
    cost_status: str = "unknown"
    error: dict | None = None

    def to_dict(self):
        return asdict(self)
