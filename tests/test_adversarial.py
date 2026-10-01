"""
Test Suite — Adversarial Photogrammetric Stress Tests
=====================================================
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaproof.adversarial_benchmarks import (
    run_all_adversarial_stress_tests,
    stress_test_180deg_grazing_shadow,
    stress_test_falsification_noise_attack,
    stress_test_iirs_thermal_decoupling,
    stress_test_zero_overlap_injection,
)


def test_adversarial_suite():
    dem = np.zeros((128, 128), dtype=np.float32)
    y, x = np.ogrid[:128, :128]
    dem -= 50.0 * np.exp(-((x - 64)**2 + (y - 64)**2) / (2 * 15**2))

    results = run_all_adversarial_stress_tests(dem)
    assert len(results) == 4
    for r in results:
        assert r["passed"], f"Adversarial stress test {r['test_name']} failed!"
