"""Compatibility patch for legacy libraries (like pylidc) on NumPy 2.0+ and Python 3.12+."""

import configparser
from contextlib import nullcontext
import warnings
import numpy as np


def patch_legacy_environment() -> None:
    """Restore deprecated NumPy 1.x aliases, NEP50 stub, and SafeConfigParser."""
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser

    if not hasattr(np, "_no_nep50_warning"):
        setattr(np, "_no_nep50_warning", nullcontext)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        aliases = {"int": int, "float": float, "bool": bool}
        for name, python_type in aliases.items():
            if not hasattr(np, name):
                setattr(np, name, python_type)


patch_legacy_environment()
