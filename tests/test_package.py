"""Smoke tests: the package imports and the PyZX features it relies on exist."""
import importlib

import pyzx

import zxqec


def test_version() -> None:
    assert zxqec.__version__


def test_subpackages_import() -> None:
    for name in ("qprogram", "spacetime", "tableau", "topological", "gui"):
        importlib.import_module(f"zxqec.{name}")


def test_pyzx_spacetime_available() -> None:
    # The 1+1D time layer this library builds on; requires PyZX from the fork
    # until it is part of a release.
    assert hasattr(pyzx, "spacetime")
    assert callable(pyzx.spacetime.time_slice)
