"""Compatibility patch for legacy libraries (like pylidc) on NumPy 2.0+ and Python 3.12+."""

import configparser
import numpy as np


def patch_legacy_environment() -> None:
    """Restore deprecated NumPy 1.x type aliases and configparser.SafeConfigParser."""
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser

    aliases = {"int": int, "float": float, "bool": bool}
    for name, python_type in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, python_type)


patch_legacy_environment()

