import numpy as np
import pytest
from lunaproof.physics import render_lunar, to_u8

def test_shading_azimuth_difference_and_validity():
    # Steep synthetic peak to ensure cast shadows
    N = 128
    y, x = np.ogrid[:N, :N]
    dem = 800.0 * np.exp(-((x - 64)**2 + (y - 64)**2) / (2 * 8**2)).astype(np.float32)

    px_x = 10.0
    px_y = 10.0

    # Test azimuth 90 vs 270 shadow masks differ
    img1, shadow1 = render_lunar(dem, px_x, px_y, az_deg=90, el_deg=10, noise=0.0)
    img2, shadow2 = render_lunar(dem, px_x, px_y, az_deg=270, el_deg=10, noise=0.0)

    assert shadow1.any(), "Expected cast shadow at 10 deg elevation"
    assert shadow2.any(), "Expected cast shadow at 10 deg elevation"
    assert not np.array_equal(shadow1, shadow2), "Shadow masks should differ between East and West sun angles"

    # Test no NaN or Inf for elevations in {5, 10, 30, 60}
    for el in [5, 10, 30, 60]:
        img, shadow = render_lunar(dem, px_x, px_y, az_deg=120, el_deg=el, seed=1)
        assert np.isfinite(img).all(), f"NaN or Inf found in image for elevation {el}"
        assert not np.isnan(shadow).any(), f"NaN found in shadow mask for elevation {el}"
        assert (img >= 0.0).all() and (img <= 1.0).all(), f"Image range invalid for el={el}"

def test_to_u8():
    x = np.array([-0.1, 0.0, 0.5, 1.0, 1.2], dtype=np.float32)
    u8 = to_u8(x)
    assert u8.dtype == np.uint8
    assert np.array_equal(u8, np.array([0, 0, 127, 255, 255], dtype=np.uint8))
