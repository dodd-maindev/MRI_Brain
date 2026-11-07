"""Compatibility patch for legacy libraries (like pylidc) on Python 3.12+ & NumPy 2.0+."""

import configparser
import numpy as np


def patch_legacy_environment() -> None:
    """Restore removed Python 3.12+ and NumPy 2.0+ attributes."""
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser

    aliases = {
        "int": int,
        "float": float,
        "bool": bool,
        "complex": complex,
    }
    for name, python_type in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, python_type)


patch_legacy_environment()
