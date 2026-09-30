import os
import numpy as np
import pytest
from lunaproof.dem_io import fetch_window

def test_fetch_window_round_trip_byte_orders(tmp_path):
    size = 64
    W = 23040
    H = 11520
    ppd = 64
    lat = 0.0
    lon_e = 0.0
    
    r0 = int(round((90 - lat) * ppd - size / 2))
    start_byte = r0 * W * 2
    length_byte = size * W * 2
    
    # Create fake terrain int16 data at the exact offset
    fake_dem = (np.sin(np.linspace(0, 10, size * W)).reshape(size, W) * 500.0).astype(np.int16)
    
    # Test little-endian file (<i2)
    le_file = tmp_path / "fake_le.img"
    with open(le_file, "wb") as f:
        f.seek(start_byte + length_byte - 1)
        f.write(b'\x00')
        f.seek(start_byte)
        f.write(fake_dem.astype("<i2").tobytes())
        
    dem_le, px_x, px_y, lat_c = fetch_window(str(le_file), lat, lon_e, size, ppd=ppd, W=W, H=H, scale=0.5)
    assert dem_le.shape == (size, size)
    assert dem_le.std() > 20
    
    # Test big-endian file (>i2)
    be_file = tmp_path / "fake_be.img"
    with open(be_file, "wb") as f:
        f.seek(start_byte + length_byte - 1)
        f.write(b'\x00')
        f.seek(start_byte)
        f.write(fake_dem.astype(">i2").tobytes())
        
    dem_be, px_x, px_y, lat_c = fetch_window(str(be_file), lat, lon_e, size, ppd=ppd, W=W, H=H, scale=0.5)
    assert dem_be.shape == (size, size)
    assert dem_be.std() > 20
