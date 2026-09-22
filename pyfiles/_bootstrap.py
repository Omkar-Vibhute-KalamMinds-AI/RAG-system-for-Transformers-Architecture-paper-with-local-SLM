"""Ensure `pyfiles` is on sys.path (works on Windows and Linux CI)."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_pkg = str(_PKG_DIR)
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)
