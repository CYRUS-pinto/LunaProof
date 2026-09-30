import os
import tempfile
import numpy as np
import pytest
from lunaproof.dem_io import fetch_window, boxes_overlap

def test_fetch_window_byte_order_round_trip():
    size = 64
    W = 23040
    H = 11520
    ppd = 64
    # Set lat/lon such that r0 = 0 and c0 = 0
    # r0 = int(round((90 - lat) * 64 - 32)) = 0 => lat = 89.5
    # c0 = int(round(lon_e * 64 - 32)) = 0 => lon_e = 0.5
    lat = 90.0 - (size / 2.0) / ppd
    lon_e = (size / 2.0) / ppd

    rng = np.random.default_rng(123)
    dem_synth = (1000.0 + rng.normal(0, 100, size=(size, size))).astype(np.float32)
    int16_vals = (dem_synth / 0.5).astype(np.int16)

    strip = np.zeros((size, W), dtype=np.int16)
    strip[:, :size] = int16_vals

    for byte_order, dt in [('little-endian', '<i2'), ('big-endian', '>i2')]:
        raw_bytes = strip.astype(dt).tobytes()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            dem_fetched, px_x, px_y, lat_c = fetch_window(tmp_path, lat=lat, lon_e=lon_e, size=size, ppd=ppd, W=W, H=H)
            assert dem_fetched.shape == (size, size)
            assert np.abs(dem_fetched - dem_synth).max() < 1.0, f"Round trip failed for {byte_order}"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
