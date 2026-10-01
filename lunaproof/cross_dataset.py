"""
LunaProof cross_dataset.py — v4 Cross-Dataset Generalization Evaluator
Evaluates registration on equatorial and south-pole terrain pairs.
Standalone, no internal lunaproof imports.
"""
import numpy as np
import cv2


class CrossDatasetEvaluator:
    """Cross-Dataset Generalization Evaluator for Multi-Modal Lunar Imagery."""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def evaluate_cross_dataset(self, matcher_fn=None, n=30):
        """
        Evaluate registration on equatorial (Dataset A) and south pole (Dataset B).
        matcher_fn: optional callable(img_a, img_b) -> dict with 'status' key.
        """
        eq_pass = []
        sp_pass = []
        for i in range(n):
            seed_i = 1000 + i
            h, w = 256, 256
            dem_eq = self._make_dem(h, w, seed_i, craters=15)
            a_eq = self._shade(dem_eq, 60, 35)
            b_eq = self._shade(dem_eq, 180, 40)
            eq_pass.append(self._eval_pair(a_eq, b_eq, matcher_fn))

            dem_sp = self._make_dem(h, w, seed_i + 50000, craters=25)
            a_sp = self._shade(dem_sp, 10, 8)
            b_sp = self._shade(dem_sp, 160, 6)
            sp_pass.append(self._eval_pair(a_sp, b_sp, matcher_fn))

        eq_rate = float(np.mean(eq_pass))
        sp_rate = float(np.mean(sp_pass))
        gen = float((eq_rate + sp_rate) / 2)
        return {
            "dataset_a_equatorial": {"falsification_pass_rate": eq_rate, "n_pairs": n},
            "dataset_b_south_pole": {"falsification_pass_rate": sp_rate, "n_pairs": n},
            "generalization_score": gen,
            "status": "PASS" if gen >= 0.85 else ("BORDERLINE" if gen >= 0.70 else "FAIL"),
        }

    def _make_dem(self, h, w, seed, craters=15):
        rng = np.random.default_rng(seed)
        y, x = np.ogrid[:h, :w]
        dem = np.zeros((h, w), np.float32)
        for _ in range(craters):
            cx = rng.uniform(15, w - 15)
            cy = rng.uniform(15, h - 15)
            r = rng.uniform(8, h // 5)
            d = rng.uniform(20, 70)
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            dem -= d * np.exp(-(dist ** 2) / (2 * (r / 2) ** 2))
            dem += d * 0.4 * np.exp(-((dist - r) ** 2) / (2 * (r * 0.25) ** 2))
        return dem

    def _shade(self, dem, sun_az, sun_el):
        az_r, el_r = np.radians(sun_az), np.radians(sun_el)
        lx = np.cos(el_r) * np.sin(az_r)
        ly = np.cos(el_r) * np.cos(az_r)
        lz = np.sin(el_r)
        gx = cv2.Sobel(dem, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(dem, cv2.CV_32F, 0, 1, ksize=3)
        mg = np.sqrt(gx ** 2 + gy ** 2 + 1.0)
        nx, ny, nz = -gx / mg, -gy / mg, 1.0 / mg
        ci = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
        ce = np.clip(nz, 0.01, 1.0)
        return np.clip((ci / (ci + ce + 1e-6)) * 255, 0, 255).astype(np.uint8)

    def _eval_pair(self, a, b, matcher_fn):
        if a.std() < 4.0:
            return False
        if matcher_fn is not None:
            try:
                result = matcher_fn(a, b)
                return result.get("status") == "SUCCESS"
            except Exception:
                return False
        # Default: SIFT + RANSAC
        sift = cv2.SIFT_create(nfeatures=200, contrastThreshold=0.003)
        ks, ds = sift.detectAndCompute(a, None)
        kr, dr = sift.detectAndCompute(b, None)
        if ds is None or dr is None or len(ks) < 8:
            return False
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        raw = bf.knnMatch(ds, dr, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return False
        pts_s = np.float32([ks[m.queryIdx].pt for m in good])
        pts_r = np.float32([kr[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(pts_s, pts_r, cv2.RANSAC, 3.0)
        return H is not None and mask is not None and mask.sum() >= 6
