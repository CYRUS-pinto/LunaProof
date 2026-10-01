"""
Test Suite — Diagnostic Visualization & Results Packaging
=========================================================
"""

from __future__ import annotations

import os
import zipfile
import numpy as np
import pytest

from lunaproof.benchmark_visualization import generate_master_diagnostic_plots, package_lunaproof_results_zip


def test_package_lunaproof_results_zip(tmp_path):
    ref_img = np.zeros((128, 128), dtype=np.uint8)
    aligned_src = np.full((128, 128), 100, dtype=np.uint8)

    pts_ref = np.array([[20, 20], [50, 50], [80, 80], [100, 100]], dtype=np.float32)
    pts_src = pts_ref + 1.0
    residuals_m = np.array([0.25, 0.30, 0.22, 0.28], dtype=np.float32)

    zip_out = os.path.join(tmp_path, "lunaproof_results.zip")
    temp_dir = os.path.join(tmp_path, "temp_export")

    zip_path = package_lunaproof_results_zip(
        ref_img, aligned_src, pts_ref, pts_src, residuals_m,
        output_zip_path=zip_out, temp_dir=temp_dir
    )

    assert os.path.isfile(zip_path)
    assert zipfile.is_zipfile(zip_path)

    with zipfile.ZipFile(zip_path, 'r') as z:
        names = z.namelist()
        assert any("registration" in n or "telemetry" in n for n in names)
        assert any("tie_points" in n or "csv" in n for n in names)

