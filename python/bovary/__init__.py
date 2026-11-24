"""Bovary alpha inference workbench. Importing starts no backends."""
__version__ = "0.2.0a18"

from .contract import Event, ExecutionError, Request, Result
from .runtime import create_workbench

__all__ = ['__version__', 'Event', 'ExecutionError', 'Request', 'Result', 'create_workbench']
