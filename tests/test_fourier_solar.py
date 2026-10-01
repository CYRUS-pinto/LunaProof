"""
Test Suite — Fourier-Mellin Pre-alignment and Solar Vector Filtering
===================================================================
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaproof.fourier_mellin import estimate_fourier_mellin_scale_rotation
from lunaproof.solar_prior import compute_solar_shadow_vector, filter_keypoints_by_solar_perpendicularity
import cv2


def test_fourier_mellin():
    img1 = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(img1, (64, 64), 30, 255, -1)

    scale, rot, resp = estimate_fourier_mellin_scale_rotation(img1, img1)
    assert scale > 0.5
    assert isinstance(rot, float)
    assert resp >= 0.0


def test_solar_shadow_vector():
    lx, ly, lz = compute_solar_shadow_vector(90.0, 45.0)
    assert pytest.approx(lx**2 + ly**2 + lz**2, abs=1e-3) == 1.0


def test_filter_keypoints_by_solar_perpendicularity():
    img = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(img, (64, 64), 30, 255, 2)
    sift = cv2.SIFT_create()
    kps, _ = sift.detectAndCompute(img, None)

    filtered = filter_keypoints_by_solar_perpendicularity(kps, img, sun_az_deg=90.0, sun_el_deg=30.0)
    assert isinstance(filtered, list)
    assert len(filtered) > 0
