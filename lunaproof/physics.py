import math
import numpy as np
import cv2

def render_lunar(dem_m, px_x_m, px_y_m, az_deg, el_deg, noise=0.01, psf_sigma=0.8,
                 max_steps=160, seed=0):

    """Lommel-Seeliger shading + real cast shadows of a DEM. Compass azimuth (0=N, 90=E), image is north-up.
    Nadir viewing (e = slope angle). No atmosphere => no fill light. Returns float image in [0,1] and shadow mask."""
    rng = np.random.default_rng(seed)
    dem = dem_m.astype(np.float32)
    H, W = dem.shape
    dz_drow, dz_dcol = np.gradient(dem, px_y_m, px_x_m)          # metres / metre
    gx, gn = dz_dcol, -dz_drow                                   # east, north gradients
    nz = 1.0 / np.sqrt(1 + gx**2 + gn**2)
    nx, ny = -gx * nz, -gn * nz
    az, el = math.radians(az_deg), math.radians(el_deg)
    L = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)], np.float32)
    mu0 = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, None)    # cos(incidence)
    mu = nz                                                      # cos(emission), nadir camera
    ls = mu0 / (mu0 + mu + 1e-6)                                 # Lommel-Seeliger
    # cast shadows: march toward the sun, compare terrain height to the sun ray
    dcol, drow = math.sin(az), -math.cos(az)
    step_m = math.hypot(dcol * px_x_m, drow * px_y_m)
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float32)
    shadow = np.zeros((H, W), bool)
    tan_el = math.tan(el)
    for s in range(1, max_steps + 1):
        mapx, mapy = cc + s * dcol, rr + s * drow
        h = cv2.remap(dem, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1e9)
        shadow |= (h - dem) > s * step_m * tan_el
    img = ls * (~shadow)
    img = img * (0.6 + 0.4 * math.sin(el))                       # brightness depends on sun elevation
    img = cv2.GaussianBlur(img, (0, 0), psf_sigma)               # optics PSF
    img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    lo, hi = np.percentile(img, [1, 99.5])
    img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    return img.astype(np.float32), shadow

def to_u8(x):
    return np.clip(x * 255, 0, 255).astype(np.uint8)
