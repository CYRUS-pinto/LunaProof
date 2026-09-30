import numpy as np
import cv2
from lunaproof.match import evaluate

def test_evaluate_exact_ground_truth():
    size = 320
    # Perfect rotation matrix (e.g. 10 deg rotation, 1.0 scale)
    M_gt = cv2.getRotationMatrix2D((size / 2, size / 2), 10.0, 1.0).astype(np.float64)
    
    # Generate points pa on a grid
    gx, gy = np.meshgrid(np.linspace(50, 270, 15), np.linspace(50, 270, 15))
    pa = np.c_[gx.ravel(), gy.ravel()].astype(np.float32)
    
    # Transform pa using M_gt to create exact pb matches
    pb = (np.c_[pa, np.ones(len(pa))] @ M_gt.T).astype(np.float32)
    
    valid_src = np.ones((size, size), bool)
    res = evaluate(pa, pb, M_gt, valid_src, gsd_m=470.0, size=size)
    
    assert res['med_err_px'] < 0.01

def test_evaluate_shifted_matches_proves_ground_truth():
    size = 320
    M_gt = cv2.getRotationMatrix2D((size / 2, size / 2), 0.0, 1.0).astype(np.float64)
    
    gx, gy = np.meshgrid(np.linspace(50, 270, 15), np.linspace(50, 270, 15))
    pa = np.c_[gx.ravel(), gy.ravel()].astype(np.float32)
    
    # Shift pb by 10 pixels in X
    pb = pa.copy()
    pb[:, 0] += 10.0
    
    valid_src = np.ones((size, size), bool)
    res = evaluate(pa, pb, M_gt, valid_src, gsd_m=470.0, size=size)
    
    # Prove error is measured against ground truth (should be ~10 px)
    assert 9.5 <= res['med_err_px'] <= 10.5
