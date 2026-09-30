"""Tests for Quadtree Gini spatial dispersion and gate_v2 (lunaproof/match.py)."""

import numpy as np
import pytest
from lunaproof.match import (
    calculate_spatial_gini,
    gate_v1,
    gate_v2,
    spatial_confidence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IMG_SHAPE = (640, 640)


def _uniform_points(n: int = 200, seed: int = 0) -> np.ndarray:
    """N points uniformly scattered over a 640×640 image."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 640, (n, 2)).astype(np.float32)


def _clustered_points(n: int = 200, cx: float = 100.0, cy: float = 100.0,
                       sigma: float = 20.0, seed: int = 0) -> np.ndarray:
    """N points clustered tightly in one corner (high Gini)."""
    rng = np.random.default_rng(seed)
    return np.clip(
        rng.normal([cx, cy], sigma, (n, 2)), 0, 640
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Gini coefficient properties
# ---------------------------------------------------------------------------

def test_gini_empty_returns_one():
    assert calculate_spatial_gini(np.zeros((0, 2)), IMG_SHAPE) == pytest.approx(1.0)


def test_gini_uniform_is_low():
    pts  = _uniform_points(400)
    gini = calculate_spatial_gini(pts, IMG_SHAPE, num_bins=16)
    assert gini < 0.55, f"Uniform distribution Gini {gini:.3f} should be < 0.55"


def test_gini_clustered_is_high():
    pts  = _clustered_points(400, cx=50.0, cy=50.0, sigma=10.0)
    gini = calculate_spatial_gini(pts, IMG_SHAPE, num_bins=16)
    assert gini > 0.55, f"Clustered distribution Gini {gini:.3f} should be > 0.55"


def test_gini_clustered_greater_than_uniform():
    g_uniform   = calculate_spatial_gini(_uniform_points(300),    IMG_SHAPE)
    g_clustered = calculate_spatial_gini(_clustered_points(300),  IMG_SHAPE)
    assert g_clustered > g_uniform


def test_gini_in_range():
    for seed in range(5):
        pts  = _uniform_points(100, seed=seed)
        gini = calculate_spatial_gini(pts, IMG_SHAPE)
        assert 0.0 <= gini <= 1.0, f"Gini out of [0,1]: {gini}"


# ---------------------------------------------------------------------------
# Spatial confidence score
# ---------------------------------------------------------------------------

def test_spatial_confidence_uniform_high():
    pts = _uniform_points(400)
    sc  = spatial_confidence(pts, IMG_SHAPE)
    assert sc['spatial_confidence_score'] > 0.3


def test_spatial_confidence_clustered_low():
    pts = _clustered_points(400)
    sc  = spatial_confidence(pts, IMG_SHAPE)
    # Clustered → high Gini → low Sc
    assert sc['spatial_confidence_score'] < 0.3


def test_spatial_confidence_empty():
    sc = spatial_confidence(np.zeros((0, 2), np.float32), IMG_SHAPE)
    assert sc['grid_coverage'] == 0.0
    assert sc['gini'] == pytest.approx(1.0)
    assert sc['spatial_confidence_score'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Gate v1 vs Gate v2 behaviour
# ---------------------------------------------------------------------------

def _res(n_inliers, inlier_ratio, coverage):
    return dict(n_inliers=n_inliers, inlier_ratio=inlier_ratio, coverage=coverage)


def test_gate_v1_success():
    assert gate_v1(_res(50, 0.60, 0.70)) == 'SUCCESS'


def test_gate_v1_degraded():
    assert gate_v1(_res(25, 0.40, 0.20)) == 'DEGRADED'


def test_gate_v1_refused_low_inliers():
    assert gate_v1(_res(10, 0.10, 0.05)) == 'REFUSED'


def test_gate_v2_success_with_uniform_points():
    """gate_v2 with well-spread inlier points should return SUCCESS."""
    pts = _uniform_points(300)
    res = dict(
        n_inliers=120, inlier_ratio=0.60, coverage=0.70,
        inlier_pts=pts
    )
    assert gate_v2(res, IMG_SHAPE) == 'SUCCESS'


def test_gate_v2_refused_clustered_points():
    """gate_v2 with clustered inlier points should not return SUCCESS."""
    pts = _clustered_points(300, cx=50.0, cy=50.0, sigma=8.0)
    res = dict(
        n_inliers=80, inlier_ratio=0.50, coverage=0.10,
        inlier_pts=pts
    )
    label = gate_v2(res, IMG_SHAPE)
    assert label in ('REFUSED', 'DEGRADED')


def test_gate_v2_refused_too_few_inliers():
    res = dict(n_inliers=5, inlier_ratio=0.05, coverage=0.05)
    assert gate_v2(res) == 'REFUSED'


def test_gate_v2_stricter_than_v1_on_clustered():
    """v2 should be at least as strict as v1 for spatially clustered matches."""
    pts = _clustered_points(200)
    res_v2 = dict(n_inliers=50, inlier_ratio=0.45, coverage=0.35, inlier_pts=pts)
    res_v1 = dict(n_inliers=50, inlier_ratio=0.45, coverage=0.35)
    label_v1 = gate_v1(res_v1)
    label_v2 = gate_v2(res_v2, IMG_SHAPE)
    # ORDER: REFUSED < DEGRADED < SUCCESS
    order = {'REFUSED': 0, 'DEGRADED': 1, 'SUCCESS': 2}
    assert order[label_v2] <= order[label_v1], (
        f"gate_v2 ({label_v2}) should be ≤ gate_v1 ({label_v1}) for clustered pts"
    )


def test_gate_bytehat_gini_boundary():
    """ByteHats reported Gini=0.559 (just above 0.55 threshold) → not SUCCESS."""
    pts = _clustered_points(227, sigma=50.0)   # ~0.56 Gini
    res = dict(
        n_inliers=110, inlier_ratio=0.485, coverage=0.50,
        inlier_pts=pts
    )
    # ByteHats' Gini=0.559 is above our 0.55 ceiling → DEGRADED or REFUSED
    label = gate_v2(res, IMG_SHAPE)
    assert label in ('DEGRADED', 'REFUSED'), (
        f"ByteHats-like Gini should not yield SUCCESS; got {label}"
    )
