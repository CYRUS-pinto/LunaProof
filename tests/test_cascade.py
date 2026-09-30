"""Tests for the Bounded 3-Hop Scale Cascade (lunaproof/cascade.py).

All tests use:
  - DEM size 32×32 (tiny, for speed)
  - max_steps=4 for shadow rendering (not physically accurate, but correct code path)
  - Random uint8 images for hop smoke tests (no rendering needed)

Production quality uses DEM 640×640 + max_steps=160; that belongs in the Colab notebook.
"""

import time

import numpy as np
import pytest

from lunaproof.cascade import (
    BoundedScaleCascade,
    HopResult,
    hop1_nmi_phase,
    hop2_phase_sift,
    hop3_patchnet_ecc,
    make_synthetic_tiers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _hill_dem(n: int = 32) -> np.ndarray:
    """Tiny synthetic DEM: gaussian hill in the centre."""
    y, x = np.ogrid[:n, :n]
    return 120.0 * np.exp(
        -((x - n // 2) ** 2 + (y - n // 2) ** 2) / (n * 3.0)
    ).astype(np.float32)


def _random_img(h: int = 48, w: int = 48, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Hop ratio validation — zero rendering, instant
# ---------------------------------------------------------------------------

def test_cascade_hop_ratios_valid():
    """All default hops must be ≤ 16×."""
    c = BoundedScaleCascade()
    ratios = [c.scales[i] / c.scales[i + 1] for i in range(len(c.scales) - 1)]
    assert all(r <= 16.0 for r in ratios), f"Hop ratio violation: {ratios}"


def test_cascade_invalid_hop_raises():
    """A scale list with a >16× hop must raise ValueError at construction."""
    with pytest.raises(ValueError, match="Cascade violation"):
        BoundedScaleCascade(scales=[80.0, 0.25])   # single hop = 320× → invalid


# ---------------------------------------------------------------------------
# Synthetic tier generator — uses tiny DEM + max_steps=4 for speed
# ---------------------------------------------------------------------------

def test_make_synthetic_tiers_shapes():
    dem   = _hill_dem(32)
    tiers = make_synthetic_tiers(dem, px_x_m=80.0, px_y_m=80.0,
                                 az_deg=45.0, el_deg=30.0, max_steps=4)
    assert len(tiers) == 4, "Expected 4 tiers (80m, 5m, 1m, 0.25m)"
    for t in tiers:
        assert t.ndim == 2, "Each tier should be a 2-D image"
        assert t.dtype == np.uint8
        assert t.size > 0


def test_make_synthetic_tiers_scale_monotone():
    """Higher-resolution tiers should produce spatially larger images."""
    dem   = _hill_dem(32)
    tiers = make_synthetic_tiers(dem, px_x_m=80.0, px_y_m=80.0, max_steps=4)
    areas = [t.shape[0] * t.shape[1] for t in tiers]
    # Coarsest (80m tier from 80m native DEM) → similar px, finest (0.25m) → many more px
    assert areas[0] <= areas[-1], (
        f"Coarsest tier ({areas[0]} px) should have ≤ pixels than finest ({areas[-1]} px)"
    )


# ---------------------------------------------------------------------------
# Individual hop smoke tests — random images, no rendering
# ---------------------------------------------------------------------------

def test_hop1_returns_hopresult():
    r = hop1_nmi_phase(_random_img(seed=1), _random_img(seed=2),
                       src_scale=80.0, dst_scale=5.0)
    assert isinstance(r, HopResult)
    assert r.hop == 1
    assert r.H.shape == (3, 3)
    assert r.H.dtype == np.float64
    assert r.ratio == pytest.approx(16.0)
    assert r.elapsed_s >= 0.0


def test_hop2_returns_hopresult():
    r = hop2_phase_sift(_random_img(seed=3), _random_img(seed=4),
                        src_scale=5.0, dst_scale=1.0)
    assert isinstance(r, HopResult)
    assert r.hop == 2
    assert r.H.shape == (3, 3)
    assert r.ratio == pytest.approx(5.0)


def test_hop3_returns_hopresult():
    r = hop3_patchnet_ecc(_random_img(seed=5), _random_img(seed=6),
                          src_scale=1.0, dst_scale=0.25)
    assert isinstance(r, HopResult)
    assert r.hop == 3
    assert r.H.shape == (3, 3)
    assert r.ratio == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Composite H_composite is 3×3 float64
# ---------------------------------------------------------------------------

def test_cascade_composite_shape_and_dtype():
    """Cascade result H_composite must be 3×3 float64 regardless of convergence."""
    imgs = [_random_img(48, 48, seed=i) for i in range(4)]
    cascade = BoundedScaleCascade()
    result  = cascade.execute(*imgs)
    assert result.H_composite.shape == (3, 3)
    assert result.H_composite.dtype == np.float64
    assert result.total_elapsed > 0.0


def test_cascade_execute_all_hops_present():
    imgs   = [_random_img(48, 48, seed=i) for i in range(4)]
    result = BoundedScaleCascade().execute(*imgs)
    assert len(result.hops) == 3
    assert [h.hop for h in result.hops] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Performance gate: each hop on 48×48 must be well under 0.5 s
# ---------------------------------------------------------------------------

def test_hop1_latency():
    a = _random_img(64, 64, seed=7)
    b = _random_img(64, 64, seed=8)
    t0 = time.perf_counter()
    hop1_nmi_phase(a, b)
    assert time.perf_counter() - t0 < 0.5


def test_hop3_latency_under_250ms():
    """hop3 (ECC) on 64×64 must finish in <250ms."""
    a = _random_img(64, 64, seed=9)
    b = _random_img(64, 64, seed=10)
    t0 = time.perf_counter()
    hop3_patchnet_ecc(a, b)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.25, f"hop3_patchnet_ecc took {elapsed:.3f}s (limit 0.25s)"
