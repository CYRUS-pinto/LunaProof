"""
LunaProof real_data.py — v4 Standalone Real Data Loader
Loads real LRO NAC stereo pairs from NASA PDS archive.
Falls back to physics-correct Lommel-Seeliger synthetic generation.
"""
import os
import cv2
import numpy as np
import urllib.request
import urllib.error
import struct

# Real LRO NAC stereo pair URLs (NASA PDS archive — no login required)
LRO_NAC_PAIRS = {
    "copernicus": {
        "L": "https://pds.lroc.asu.edu/data/LRO-L-LROC-2-EDR-V1.0/LROLROC_0001/DATA/COM/2010121/M108974394LE.IMG",
        "R": "https://pds.lroc.asu.edu/data/LRO-L-LROC-2-EDR-V1.0/LROLROC_0001/DATA/COM/2010121/M108974394RE.IMG",
        "region": "Copernicus Crater Equatorial",
    },
    "sp_shackleton": {
        "L": "https://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLROC_2001/DATA/NAC_ROI/NAC_ROI_SHACKLETON_E118S8990_256P.IMG",
        "region": "Shackleton Crater South Pole",
    },
}


def _pds_img_to_numpy(raw_bytes, rows=512, cols=512):
    """Parse a PDS3 .IMG file to numpy array."""
    try:
        label_end = raw_bytes.find(b"END\r\n") + 5
        if label_end < 5:
            label_end = 2048
        block_size = 512
        offset = ((label_end + block_size - 1) // block_size) * block_size
        pixel_data = raw_bytes[offset: offset + rows * cols]
        if len(pixel_data) < rows * cols:
            pixel_data = raw_bytes[-rows * cols:]
        img = np.frombuffer(pixel_data[: rows * cols], dtype=np.uint8).reshape(rows, cols)
        return img
    except Exception:
        return None


def _make_synthetic_lunar_pair(size=(512, 512), sun_az_delta=120, seed=42):
    """
    Generate a physics-correct Lommel-Seeliger synthetic lunar image pair.
    Returns (img_a, img_b) with sun azimuth differing by sun_az_delta degrees.
    Ground truth: same DEM, different illumination direction.
    """
    rng = np.random.default_rng(seed)
    h, w = size
    # Multi-scale crater field (DEM)
    dem = np.zeros((h, w), np.float32)
    y, x = np.ogrid[:h, :w]
    for _ in range(30):
        cx = rng.uniform(20, w - 20)
        cy = rng.uniform(20, h - 20)
        r = rng.uniform(8, min(h, w) // 5)
        d = rng.uniform(25, 80)
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        dem -= d * np.exp(-(dist ** 2) / (2 * (r / 2) ** 2))
        dem += d * 0.4 * np.exp(-((dist - r) ** 2) / (2 * (r * 0.25) ** 2))

    def _shade(dem_in, sun_az, sun_el=25):
        az_r = np.radians(sun_az)
        el_r = np.radians(sun_el)
        lx = np.cos(el_r) * np.sin(az_r)
        ly = np.cos(el_r) * np.cos(az_r)
        lz = np.sin(el_r)
        gx = cv2.Sobel(dem_in, cv2.CV_32F, 1, 0, ksize=3)
        gy_s = cv2.Sobel(dem_in, cv2.CV_32F, 0, 1, ksize=3)
        mg = np.sqrt(gx ** 2 + gy_s ** 2 + 1.0)
        nx, ny, nz = -gx / mg, -gy_s / mg, 1.0 / mg
        ci = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
        ce = np.clip(nz, 0.01, 1.0)
        return np.clip((ci / (ci + ce + 1e-6)) * 255, 0, 255).astype(np.uint8)

    base_az = float(rng.uniform(30, 150))
    img_a = _shade(dem, base_az)
    img_b = _shade(dem, base_az + sun_az_delta)
    return img_a, img_b


def load_real_lro_pair(pair_key="copernicus", cache_dir="cache_pds", size=(512, 512)):
    """
    Attempt to download a real LRO NAC stereo pair from NASA PDS.
    Returns (img_L, img_R, pair_info). Images may be None if download fails.
    """
    os.makedirs(cache_dir, exist_ok=True)
    pair_info = LRO_NAC_PAIRS.get(pair_key, {})
    results = {}
    for side in ["L", "R"]:
        url = pair_info.get(side)
        if not url:
            continue
        cache_path = os.path.join(cache_dir, f"{pair_key}_{side}.png")
        img = None
        if os.path.isfile(cache_path):
            img = cv2.imread(cache_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "LunaProof/4.0 (SIH26166)"}
                )
                with urllib.request.urlopen(req, timeout=20) as r:
                    raw = r.read()
                img = _pds_img_to_numpy(raw, *size)
                if img is not None:
                    cv2.imwrite(cache_path, img)
            except Exception as e:
                print(f"    [WARN] Failed to fetch {pair_key}/{side}: {e}")
        if img is not None:
            img = cv2.resize(img, size)
        results[side] = img
    return results.get("L"), results.get("R"), pair_info


def load_real_lunar_pds_image(key="copernicus", cache_dir="cache_pds", crop_size=(512, 512)):
    """Load a real LRO NAC image, falling back to synthetic if unavailable."""
    img_l, img_r, info = load_real_lro_pair(key, cache_dir, crop_size)
    if img_l is not None:
        return img_l
    img_a, _ = _make_synthetic_lunar_pair(crop_size, seed=2026)
    return img_a


class RealPDSDataFetcher:
    """Fetch real LRO NAC stereo pairs. Gracefully falls back to synthetic."""

    def __init__(self, cache_dir="cache_pds"):
        self.cache_dir = cache_dir
        self._real_loaded = False

    def fetch_real_pair(self, pair_type="LRO_NAC_stereo"):
        img_l, img_r, info = load_real_lro_pair("copernicus", self.cache_dir)
        if img_l is not None and img_r is not None:
            self._real_loaded = True
            return img_l, img_r, {
                "pair_type": pair_type,
                "source": "NASA LRO NAC PDS Archive",
                "sensor_src": "LRO NAC-L (0.5m GSD)",
                "sensor_ref": "LRO NAC-R (0.5m GSD)",
                "region": info.get("region", "Copernicus"),
                "real_pds_source": "https://pds.lroc.asu.edu/data/",
                "is_real_pds_data": True,
            }
        # Fallback: physics-correct synthetic
        img_a, img_b = _make_synthetic_lunar_pair((512, 512), sun_az_delta=90, seed=2026)
        return img_a, img_b, {
            "pair_type": pair_type,
            "source": "Synthetic (LRO-style Lommel-Seeliger physics)",
            "sensor_src": "Synthetic OHRC-equiv",
            "sensor_ref": "Synthetic OHRC-equiv",
            "real_pds_source": "Fallback: LRO fetch timed out",
            "is_real_pds_data": False,
        }
