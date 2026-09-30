import cv2
import numpy as np
from .physics import render_lunar, to_u8
from .phasecong import phase_congruency, pc_to_u8

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

def gate(res):
    """Refusal gate uses ONLY quantities observable without ground truth."""
    if res['n_inliers'] < 15 or res['inlier_ratio'] < 0.15 or res['coverage'] < 0.15: return 'REFUSED'
    if res['n_inliers'] < 40 or res['coverage'] < 0.30: return 'DEGRADED'
    return 'SUCCESS'
