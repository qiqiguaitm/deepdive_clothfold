"""Opt-in startup hook for R4 capture and the shared RoboTwin Warp shim."""

import sys
import types

try:
    import warp as _warp

    if not hasattr(_warp, "torch"):
        _torch = types.ModuleType("warp.torch")
        for _name in (
            "device_from_torch",
            "device_to_torch",
            "dtype_from_torch",
            "dtype_to_torch",
            "from_torch",
            "to_torch",
            "stream_from_torch",
            "stream_to_torch",
        ):
            if hasattr(_warp, _name):
                setattr(_torch, _name, getattr(_warp, _name))
        _warp.torch = _torch
        sys.modules["warp.torch"] = _torch
except Exception:
    pass

from hook import install


install()
