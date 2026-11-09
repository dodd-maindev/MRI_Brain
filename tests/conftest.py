"""PyTest configuration injecting repository paths into sys.path."""

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
parent_dir = repo_root.parent

for candidate_path in [str(parent_dir), str(repo_root)]:
    if candidate_path not in sys.path:
        sys.path.insert(0, candidate_path)

