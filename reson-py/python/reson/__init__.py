# Re-export everything from the Rust module
from .reson import *

# Override Deserializable with the Python version (has model_validate, model_dump, etc.)
from .deserializable import Deserializable

# Override reson.types with our Python _types package (includes Python Deserializable)
import sys
from . import _types as _types_module
sys.modules["reson.types"] = _types_module

__doc__ = reson.__doc__
if hasattr(reson, "__all__"):
    __all__ = reson.__all__
