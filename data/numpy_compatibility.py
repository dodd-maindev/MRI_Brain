"""Compatibility patch for legacy libraries (like pylidc) on NumPy 2.0+."""

import numpy as np


def patch_numpy_legacy_aliases() -> None:
    """Restore deprecated NumPy 1.x type aliases removed in NumPy 2.0."""
    aliases = {
        "int": int,
        "float": float,
        "bool": bool,
    }
    for name, python_type in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, python_type)


patch_numpy_legacy_aliases()
