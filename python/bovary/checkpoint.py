"""The verified Gemma checkpoint used by the offline workflows."""
from .contract import ExecutionError
from .local import LocalAdapter
from .modelproof import GEMMA_BYTES, GEMMA_REPOSITORY, GEMMA_REVISION, GEMMA_SHA256


PIN = {
    "repository": GEMMA_REPOSITORY,
    "revision": GEMMA_REVISION,
    "bytes": GEMMA_BYTES,
    "sha256": GEMMA_SHA256,
    "verified": True,
}


class PinnedGemma(LocalAdapter):
    def _checkpoint(self, *, interrupt=None):
        model, manifest = super()._checkpoint(interrupt=interrupt)
        if any(manifest.get(key) != value for key, value in PIN.items()):
            raise ExecutionError("model_unavailable", "Verification proof must identify the pinned Gemma checkpoint")
        return model, manifest
