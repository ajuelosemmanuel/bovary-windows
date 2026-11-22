"""Lazy construction shared by command line and GUI; no import-time inference."""
import os
import hashlib
import json
from pathlib import Path


def load_validator(path):
    """Load a bounded validator configuration shared by CLI and GUI."""
    if path is None:
        return None
    from .contract import ExecutionError
    from .validators import validate_specification
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ExecutionError('invalid_validator', 'Validator file exceeds 64 KiB')
        specification = json.loads(raw.decode('utf-8-sig'))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ExecutionError('invalid_validator', 'Validator file must contain valid JSON') from error
    if not isinstance(specification, dict):
        raise ExecutionError('invalid_validator', 'Validator file must contain a JSON object')
    validate_specification(specification)
    return specification


def provider_identity(path):
    """One authorization/configuration identity across path aliases and copies."""
    if path is None:
        return None
    configuration = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    canonical = json.dumps(configuration, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def project_home():
    configured = os.environ.get("BOVARY_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "CMakeLists.txt").exists():
        return checkout
    return Path.cwd().resolve()


def create_workbench(root=None, provider_config=None):
    from .contract import ExecutionError
    from .local import LocalAdapter
    from .routing import Workbench
    from .provider import OpenAIAdapter, ProviderConfig
    from .policy import default_policy_path
    root = Path(root) if root else project_home()
    commercial = None
    if provider_config:
        config = json.loads(Path(provider_config).read_text(encoding="utf-8-sig"))
        try:
            commercial = OpenAIAdapter(ProviderConfig(**config))
        except TypeError as error:
            raise ExecutionError('provider_config', 'Provider configuration must use the documented fields and types') from error
    policy = json.loads(default_policy_path(root).read_text(encoding='utf-8-sig'))
    return Workbench(local=LocalAdapter(root=root), commercial=commercial, policy=policy)
