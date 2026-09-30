import numpy as np
import os
import math
import requests

MOON_R = 1737400.0
M_PER_DEG = 2 * math.pi * MOON_R / 360.0

def read_range(src, start, length):
    if os.path.exists(src):
        with open(src, 'rb') as f:
            f.seek(start)
            return f.read(length)
    r = requests.get(src, headers={'Range': f'bytes={start}-{start+length-1}'}, timeout=180)
    if r.status_code != 206:
        raise RuntimeError(f'server did not honour Range request (HTTP {r.status_code}) for {src}')
    return r.content

def fetch_window(src, lat, lon_e, size, ppd=64, W=23040, H=11520, scale=0.5):
    """LOLA LDEM cylindrical GDR: row=(90-lat)*ppd, col=lon_east*ppd, int16 DN*0.5 = metres. Returns dem(m), px_x_m, px_y_m."""
    r0 = int(round((90 - lat) * ppd - size / 2)); r0 = max(0, min(H - size, r0))
    c0 = int(round(lon_e * ppd - size / 2))
    raw = read_range(src, r0 * W * 2, size * W * 2)
    dem = None
    for dt in ('<i2', '>i2'):                                  # detect byte order by physical sanity
        a = np.frombuffer(raw, dtype=dt).reshape(size, W)
        cols = np.arange(c0, c0 + size) % W
        z = a[:, cols].astype(np.float32) * scale
        if np.abs(z).max() < 12000 and z.std() > 20:
            dem = z; break
    if dem is None: raise RuntimeError('DEM window failed sanity check (bad byte order / wrong file?)')
    lat_c = 90 - (r0 + size / 2) / ppd
    px_y = M_PER_DEG / ppd
    px_x = px_y * math.cos(math.radians(lat_c))
    return dem, px_x, px_y, lat_c

def boxes_overlap(a, b, size_deg):
    dlat = abs(a[0] - b[0]); dlon = abs(a[1] - b[1]); dlon = min(dlon, 360 - dlon)
    return dlat < size_deg and dlon < size_deg
