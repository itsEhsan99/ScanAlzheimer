"""Smoke test: verify the package is importable and correctly installed."""

import ScanAlzheimer


def test_package_imports():
    assert ScanAlzheimer.__version__ == "0.1.0"


def test_subpackages_importable():
    from ScanAlzheimer import data, evaluation, models, preprocessing

    assert all(m is not None for m in (data, evaluation, models, preprocessing))
