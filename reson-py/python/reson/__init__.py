import sys

# Keep a stable handle to the compiled extension module, then re-export symbols.
from . import reson as _rust
from .reson import *  # noqa: F401,F403

# Override Deserializable with the Python version (has model_validate, model_dump, etc.).
from .deserializable import Deserializable

# Override reson.types with our Python _types package (includes Python Deserializable).
from . import _types as _types_module
sys.modules["reson.types"] = _types_module

__doc__ = _rust.__doc__
__all__ = list(getattr(_rust, "__all__", []))
if "Deserializable" not in __all__:
    __all__.append("Deserializable")
