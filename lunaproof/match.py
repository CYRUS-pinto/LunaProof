import cv2
import numpy as np
from .physics import render_lunar, to_u8
from .phasecong import phase_congruency, pc_to_u8

# ---------------------------------------------------------------------------
# Quadtree Gini Spatial Dispersion
# ---------------------------------------------------------------------------

def calculate_spatial_gini(
    points:    np.ndarray,
    img_shape: tuple,
    num_bins:  int = 16,
) -> float:
    """
    Gini coefficient of tie-point spatial dispersion across a quadtree grid.

    A Gini coefficient of 0 means perfectly uniform distribution;
    1 means all points are in one bin.  Target: Gini <= 0.55.

    Args:
        points:    (N, 2) float32 array of (x, y) pixel coordinates.
        img_shape: (H, W) or (H, W, C) image shape.
        num_bins:  Total number of quadtree bins (must be a perfect square).

    Returns:
        Gini coefficient in [0, 1].
    """
    if len(points) == 0:
        return 1.0
    h, w    = img_shape[:2]
    grid_d  = int(num_bins ** 0.5)          # e.g. 4 for 16 bins
    bin_x   = np.clip(
        (points[:, 0] / w * grid_d).astype(int), 0, grid_d - 1
    )
    bin_y   = np.clip(
        (points[:, 1] / h * grid_d).astype(int), 0, grid_d - 1
    )
    bin_idx = bin_y * grid_d + bin_x
    counts  = np.bincount(bin_idx, minlength=num_bins).astype(float)
    counts_sorted = np.sort(counts)
    n       = len(counts)
    index   = np.arange(1, n + 1)
    total   = counts_sorted.sum()
    if total < 1e-9:
        return 1.0
    gini = (2 * np.sum(index * counts_sorted)) / (n * total) - (n + 1) / n
    return float(np.clip(gini, 0.0, 1.0))


def spatial_confidence(
    pts_inlier: np.ndarray,
    img_shape:  tuple,
    grid:       int = 8,
    num_bins:   int = 16,
) -> dict:
    """
    Fused spatial quality metric:
        Sc = grid_coverage * (1 - Gini)

    Returns dict with keys: grid_coverage, gini, spatial_confidence_score.
    """
    if len(pts_inlier) == 0:
        return dict(grid_coverage=0.0, gini=1.0, spatial_confidence_score=0.0)
    h, w   = img_shape[:2]
    size   = max(h, w)
    # 8×8 grid coverage
    cells  = set(
        (int(x * grid / size), int(y * grid / size))
        for x, y in pts_inlier
    )
    cov    = len(cells) / grid ** 2
    gini   = calculate_spatial_gini(pts_inlier, img_shape, num_bins)
    sc     = cov * (1.0 - gini)
    return dict(grid_coverage=cov, gini=gini, spatial_confidence_score=sc)

def make_pair(dem, px_x, px_y, az_ref, el_ref, az_src, el_src, rot_deg, scale, seed=0):
    ref, _ = render_lunar(dem, px_x, px_y, az_ref, el_ref, seed=seed)
    srcd, _ = render_lunar(dem, px_x, px_y, az_src, el_src, seed=seed + 1)
    H, W = ref.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rot_deg, scale).astype(np.float64)  # p_src = M @ [p_ref,1]
    src = cv2.warpAffine(srcd, M, (W, H), flags=cv2.INTER_LINEAR)
    valid_src = cv2.warpAffine(np.ones_like(srcd), M, (W, H), flags=cv2.INTER_NEAREST) > 0.5
    valid_src = cv2.erode(valid_src.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=24).astype(bool)
    return to_u8(ref), to_u8(src), M, valid_src

def mutual_ratio_match(d1, d2, ratio=0.85):
    """brute-force L2 matcher with ratio test + mutual check; works for SIFT float descriptors"""
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2: return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m12 = bf.knnMatch(d1, d2, k=2); m21 = bf.match(d2, d1)
    back = {m.queryIdx: m.trainIdx for m in m21}
    out = []
    for pair in m12:
        if len(pair) < 2: continue
        a, b = pair
        if a.distance < ratio * b.distance and back.get(a.trainIdx, -1) == a.queryIdx:
            out.append((a.queryIdx, a.trainIdx))
    return np.array(out, int).reshape(-1, 2)

def sift_match(a_u8, b_u8, mask_b=None, nfeat=4000, ratio=0.85, contrast=0.01):
    sift = cv2.SIFT_create(nfeatures=nfeat, contrastThreshold=contrast)
    k1, d1 = sift.detectAndCompute(a_u8, None)
    k2, d2 = sift.detectAndCompute(b_u8, (mask_b.astype(np.uint8) * 255) if mask_b is not None else None)
    idx = mutual_ratio_match(d1, d2, ratio)
    if len(idx) == 0: return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    return (np.float32([k1[i].pt for i in idx[:, 0]]), np.float32([k2[j].pt for j in idx[:, 1]]))

def sift_on_pc(a_u8, b_u8, mask_b=None, **kw):
    pa, _ = phase_congruency(a_u8); pb, _ = phase_congruency(b_u8)
    return sift_match(pc_to_u8(pa), pc_to_u8(pb), mask_b, **kw)

def verify(pa, pb, thr=3.0):
    if len(pa) < 4: return None, np.zeros(len(pa), bool)
    H, m = cv2.findHomography(pa.reshape(-1, 1, 2), pb.reshape(-1, 1, 2), cv2.USAC_MAGSAC, thr, maxIters=5000, confidence=0.999)
    if H is None or m is None: return None, np.zeros(len(pa), bool)
    return H, m.ravel().astype(bool)

def apply_H(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]

def evaluate(pa, pb, M_gt, valid_src, gsd_m, size, grid=8):
    """Ground-truth evaluation: error at held-out CHECKPOINTS (not the fitted matches)."""
    res = dict(n_matches=len(pa), n_inliers=0, inlier_ratio=0.0, match_precision=0.0,
               med_err_px=np.inf, rmse_px=np.inf, p95_px=np.inf, pct_lt2px=0.0, coverage=0.0, med_err_m=np.inf)
    if len(pa) == 0: return res
    # independent of the estimator: are the raw matches correct w.r.t. ground truth?
    gt_b = (np.c_[pa, np.ones(len(pa))] @ M_gt.T)
    res['match_precision'] = float(np.mean(np.linalg.norm(gt_b - pb, axis=1) < 3.0))
    H, inl = verify(pa, pb)
    res['n_inliers'] = int(inl.sum()); res['inlier_ratio'] = float(inl.mean()) if len(inl) else 0.0
    if H is None: return res
    cells = set((int(x * grid / size), int(y * grid / size)) for x, y in pa[inl]); res['coverage'] = len(cells) / grid**2
    g = np.linspace(40, size - 40, 20); gx, gy = np.meshgrid(g, g)
    chk = np.c_[gx.ravel(), gy.ravel()].astype(np.float64)
    gt = np.c_[chk, np.ones(len(chk))] @ M_gt.T
    ok = (gt[:, 0] > 0) & (gt[:, 0] < size) & (gt[:, 1] > 0) & (gt[:, 1] < size)
    ok &= valid_src[np.clip(gt[:, 1].astype(int), 0, size - 1), np.clip(gt[:, 0].astype(int), 0, size - 1)]
    est = apply_H(H, chk)
    err = np.linalg.norm(est[ok] - gt[ok], axis=1)
    if len(err) == 0 or not np.isfinite(err).all(): return res
    res.update(med_err_px=float(np.median(err)), rmse_px=float(np.sqrt(np.mean(err**2))), p95_px=float(np.percentile(err, 95)),
               pct_lt2px=float(np.mean(err < 2.0)), med_err_m=float(np.median(err) * gsd_m))
    return res

def gate_v1(res: dict) -> str:
    """
    Gate v1 — original grid-coverage heuristic (kept for comparison).
    Observable only: inlier count, inlier ratio, 8×8 grid coverage.
    Verified at 93.6% precision on 112-pair LOLA held-out benchmark.
    """
    if res['n_inliers'] < 15 or res['inlier_ratio'] < 0.15 or res['coverage'] < 0.15:
        return 'REFUSED'
    if res['n_inliers'] < 40 or res['coverage'] < 0.30:
        return 'DEGRADED'
    return 'SUCCESS'


def gate_v2(res: dict, img_shape: tuple = (640, 640)) -> str:
    """
    Gate v2 — Coupled Gini + Grid Coverage (architectural ruling, SIH v3).

    SUCCESS requires ALL three:
        inlier_ratio  >= 0.40    (40% of raw matches survive MAGSAC++)
        grid_coverage >= 0.60    (tie-points span ≥60% of 8×8 field)
        gini          <= 0.55    (quadtree Gini dispersion, 16 bins)

    Uses the fused spatial confidence score:
        Sc = grid_coverage × (1 - Gini)
    SUCCESS:  Sc >= 0.40 and inlier_ratio >= 0.40
    DEGRADED: Sc >= 0.15 or inlier_ratio >= 0.15
    REFUSED:  otherwise
    """
    n_inl = res.get('n_inliers', 0)
    ratio = res.get('inlier_ratio', 0.0)
    # Recompute spatial confidence from raw inlier points if available
    pts   = res.get('inlier_pts', None)
    gini  = res.get('gini', None)
    cov   = res.get('coverage', res.get('grid_coverage', 0.0))

    if pts is not None and len(pts) > 0:
        sq = spatial_confidence(pts, img_shape)
        cov  = sq['grid_coverage']
        gini = sq['gini']
        sc   = sq['spatial_confidence_score']
    else:
        gini = gini if gini is not None else (1.0 - cov)   # estimate if missing
        sc   = cov * (1.0 - gini)

    # Hard REFUSED gate
    if n_inl < 15 or ratio < 0.15 or cov < 0.15:
        return 'REFUSED'
    # Quality tiers
    if ratio >= 0.40 and cov >= 0.60 and (gini is None or gini <= 0.55) and sc >= 0.40:
        return 'SUCCESS'
    if ratio >= 0.20 or sc >= 0.20:
        return 'DEGRADED'
    return 'REFUSED'


# Default gate exposed to benchmark code — v2 is the current standard
gate = gate_v2
