import numpy as np
import pytest
from lunaproof.physics import render_lunar

def test_shading_azimuth_flip():
    # Steep synthetic hill in the center to create clear cast shadows
    N = 128
    y, x = np.ogrid[:N, :N]
    dem = 800.0 * np.exp(-((x - 64)**2 + (y - 64)**2) / (2 * 8**2)).astype(np.float32)
    
    img1, mask1 = render_lunar(dem, 10.0, 10.0, az_deg=90, el_deg=10, noise=0.0)
    img2, mask2 = render_lunar(dem, 10.0, 10.0, az_deg=270, el_deg=10, noise=0.0)
    
    # Shadow masks must differ when sun azimuth flips 180 degrees
    assert mask1.any(), "Expected cast shadow at 10 deg elevation for 800m peak"
    assert mask2.any(), "Expected cast shadow at 10 deg elevation for 800m peak"
    assert not np.array_equal(mask1, mask2)

@pytest.mark.parametrize("el", [5, 10, 30, 60])
def test_shading_elevations_no_nan(el):
    N = 64
    dem = np.random.uniform(0, 100, (N, N)).astype(np.float32)
    img, mask = render_lunar(dem, 5.0, 5.0, az_deg=45, el_deg=el)
    
    assert np.isfinite(img).all()
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert mask.dtype == bool
