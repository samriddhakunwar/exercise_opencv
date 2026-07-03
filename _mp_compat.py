"""
_mp_compat.py
-------------
Import this module as the FIRST import in any file that uses mediapipe.

Reason:
    MediaPipe's `optional_dependencies.py` does:
        from tensorflow.tools.docs import doc_controls
    If TensorFlow is installed but protobuf is mismatched (< 4.21, missing
    `runtime_version`) the entire tensorflow import chain crashes before
    mediapipe can load.

    We pre-register lightweight stub modules in sys.modules so Python
    returns our stubs instead of ever attempting the broken real import.
    The single decorator mediapipe actually uses (`do_not_generate_docs`)
    is provided as a no-op lambda.
"""

import sys
import types


def _stub(qualified_name: str) -> types.ModuleType:
    """Create, register, and return a blank module named *qualified_name*."""
    mod = types.ModuleType(qualified_name)
    sys.modules[qualified_name] = mod
    return mod


def _patch() -> None:
    """Inject the stub hierarchy when needed."""

    # If the real tensorflow.tools.docs.doc_controls is already importable,
    # we have nothing to do.
    try:
        import importlib
        importlib.import_module("tensorflow.tools.docs.doc_controls")
        return   # real module is fine — no patch needed
    except Exception:
        pass  # Fall through and apply the stub

    # Build / complete the module hierarchy in sys.modules.
    # We must not replace a partially-initialised real module;
    # only add entries that are genuinely absent.
    hierarchy = [
        "tensorflow",
        "tensorflow.tools",
        "tensorflow.tools.docs",
        "tensorflow.tools.docs.doc_controls",
    ]
    stubs: dict[str, types.ModuleType] = {}
    for name in hierarchy:
        if name not in sys.modules:
            stubs[name] = _stub(name)
        else:
            stubs[name] = sys.modules[name]

    # Provide the no-op decorator.
    stubs["tensorflow.tools.docs.doc_controls"].do_not_generate_docs = (
        lambda f: f
    )

    # Wire attribute access (so `module.sub` works in addition to
    # `sys.modules` look-ups).
    try:
        stubs["tensorflow"].tools = stubs["tensorflow.tools"]
    except Exception:
        pass
    try:
        stubs["tensorflow.tools"].docs = stubs["tensorflow.tools.docs"]
    except Exception:
        pass
    try:
        stubs["tensorflow.tools.docs"].doc_controls = (
            stubs["tensorflow.tools.docs.doc_controls"]
        )
    except Exception:
        pass


_patch()
