import numpy as np
import pytest
from lunaproof.phasecong import phase_congruency, pc_to_u8

def test_phase_congruency_shape_and_range():
    H, W = 128, 128
    rng = np.random.default_rng(42)
    img_u8 = (rng.random((H, W)) * 255).astype(np.uint8)

    pc, M = phase_congruency(img_u8)

    assert pc.shape == (H, W), f"Expected shape ({H}, {W}), got {pc.shape}"
    assert M.shape == (H, W), f"Expected shape ({H}, {W}), got {M.shape}"

    assert np.isfinite(pc).all(), "Phase congruency map contains NaN or Inf"
    assert (pc >= 0.0).all(), "Phase congruency map contains negative values"

    u8 = pc_to_u8(pc)
    assert u8.shape == (H, W)
    assert u8.dtype == np.uint8
