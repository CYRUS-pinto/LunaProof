"""
Tests for Interactive Image Pair & Single Image Verification Engine
"""
import numpy as np
import pytest
from lunaproof.inference import auto_detect_camera_modality, verify_custom_image_pair
from lunaproof.physics import render_lunar_patch


def test_auto_detect_camera_modality():
    img_large = np.zeros((600, 600), dtype=np.uint8)
    assert "OHRC" in auto_detect_camera_modality(img_large)

    img_medium = np.zeros((200, 200), dtype=np.uint8)
    assert "TMC-2" in auto_detect_camera_modality(img_medium)

    img_small = np.zeros((64, 64), dtype=np.uint8)
    assert "IIRS" in auto_detect_camera_modality(img_small)


def test_verify_custom_image_pair():
    img_a = render_lunar_patch(128, sun_az=30.0, sun_el=35.0, seed=101)
    img_b = render_lunar_patch(128, sun_az=150.0, sun_el=40.0, seed=101)

    report = verify_custom_image_pair(img_a, img_b)
    assert report["status"] in ("SUCCESS", "REFUSED")
    assert "falsification_gate" in report
    assert "median_rmse_px" in report

