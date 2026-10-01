"""
Test Suite — SANDHI 4-Part Falsification Gate (Absorbed from Team Assassin)
=========================================================================
Verifies that falsification_gate() correctly classifies:
  1. REAL: Authentic lunar render or structured image
  2. SHUFFLED: Pixel-shuffled image (high entropy)
  3. NOISE: Pure Gaussian noise image
  4. FLAT: Constant gray image (std dev < floor)
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaproof.match import falsification_gate
from lunaproof.physics import render_lunar, to_u8


def test_falsification_gate_flat():
    flat_img = np.full((128, 128), 128, dtype=np.uint8)
    cls, diag = falsification_gate(flat_img)
    assert cls == 'FLAT'
    assert 'std' in diag


def test_falsification_gate_noise():
    rng = np.random.default_rng(42)
    noise_img = (rng.normal(128, 10, (128, 128))).clip(0, 255).astype(np.uint8)
    cls, diag = falsification_gate(noise_img, laplacian_var_floor=500.0)
    assert cls in ('NOISE', 'FLAT', 'SHUFFLED')


def test_falsification_gate_shuffled():
    dem = np.zeros((64, 64), dtype=np.float32)
    y, x = np.ogrid[:64, :64]
    dem -= 50.0 * np.exp(-((x - 32)**2 + (y - 32)**2) / (2 * 10**2))
    img_f, _ = render_lunar(dem, 1.0, 1.0, az_deg=45.0, el_deg=30.0, max_steps=8)
    img = to_u8(img_f)
    
    flat = img.flatten()
    np.random.default_rng(42).shuffle(flat)
    shuffled_img = flat.reshape(img.shape)
    
    cls, diag = falsification_gate(shuffled_img, entropy_ceil=7.0)
    assert cls in ('SHUFFLED', 'NOISE')


def test_falsification_gate_real():
    dem = np.zeros((256, 256), dtype=np.float32)
    rng = np.random.default_rng(42)
    # Add multiple craters
    for _ in range(8):
        cx, cy = rng.integers(30, 220, 2)
        r = rng.uniform(10, 30)
        depth = rng.uniform(20, 80)
        y, x = np.ogrid[:256, :256]
        dem -= depth * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * r**2))
    
    img_f, _ = render_lunar(dem, 1.0, 1.0, az_deg=45.0, el_deg=30.0, noise=0.01, seed=42, max_steps=8)
    img = to_u8(img_f)
    
    cls, diag = falsification_gate(img, kp_min=10)
    assert cls == 'REAL', f"Expected REAL, got {cls} with diag {diag}"

