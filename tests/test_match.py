import numpy as np
import cv2
import pytest
from lunaproof.match import evaluate

def test_evaluate_ground_truth_accuracy():
    size = 320
    gsd_m = 470.0
    rot_deg = 5.0
    scale = 1.05

    # Ground truth affine matrix
    M_gt = cv2.getRotationMatrix2D((size / 2, size / 2), rot_deg, scale).astype(np.float64)
    valid_src = np.ones((size, size), dtype=bool)

    # Generate 50 points in reference image
    g = np.linspace(40, size - 40, 7)
    gx, gy = np.meshgrid(g, g)
    pa = np.c_[gx.ravel(), gy.ravel()].astype(np.float32)

    # Exact transformed points
    pb_exact = (np.c_[pa, np.ones(len(pa))] @ M_gt.T).astype(np.float32)

    # Evaluate exact ground truth matches
    res_exact = evaluate(pa, pb_exact, M_gt, valid_src, gsd_m, size)
    assert res_exact['med_err_px'] < 0.01, f"Exact ground truth error should be < 0.01 px, got {res_exact['med_err_px']}"

    # Evaluate matches shifted by 10 px in X
    pb_shifted = pb_exact.copy()
    pb_shifted[:, 0] += 10.0
    res_shifted = evaluate(pa, pb_shifted, M_gt, valid_src, gsd_m, size)
    assert abs(res_shifted['med_err_px'] - 10.0) < 1.0, f"Shifted ground truth error should be ~10 px, got {res_shifted['med_err_px']}"
