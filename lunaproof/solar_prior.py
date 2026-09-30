"""
LunaProof – Solar Vector Shadow Prior & Keypoint Weighter
=========================================================
Ingests solar azimuth (az_deg) and solar elevation (el_deg) from PDS4 / SPICE kernels
to compute the solar illumination vector. Filters keypoints and feature matches based
on crater rim edge orientations perpendicular to cast shadows.

ISRO PS 26166 Sun Angle Invariance Prior.
"""

from __future__ import annotations

import math
import cv2
import numpy as np


def compute_solar_shadow_vector(sun_az_deg: float, sun_el_deg: float) -> tuple[float, float, float]:
    """
    Computes the 3D unit illumination vector (lx, ly, lz) from solar azimuth and elevation.

    Args:
        sun_az_deg: Solar azimuth in degrees (0° North, 90° East).
        sun_el_deg: Solar elevation angle above horizon in degrees.

    Returns:
        (lx, ly, lz) normalized illumination direction vector.
    """
    az_rad = math.radians(sun_az_deg)
    el_rad = math.radians(sun_el_deg)

    lx = math.cos(el_rad) * math.sin(az_rad)
    ly = math.cos(el_rad) * math.cos(az_rad)
    lz = math.sin(el_rad)

    norm = math.sqrt(lx**2 + ly**2 + lz**2) + 1e-9
    return lx / norm, ly / norm, lz / norm


def filter_keypoints_by_solar_perpendicularity(
    keypoints: list[cv2.KeyPoint],
    image: np.ndarray,
    sun_az_deg: float,
    sun_el_deg: float,
    angle_tolerance_deg: float = 30.0,
) -> list[cv2.KeyPoint]:
    """
    Filters keypoints to prioritize structural crater edges running perpendicular
    to the solar illumination vector (where shadow boundaries are sharpest and
    most invariant).

    Args:
        keypoints: OpenCV KeyPoint list.
        image: Grayscale uint8 image.
        sun_az_deg: Solar azimuth angle.
        sun_el_deg: Solar elevation angle.
        angle_tolerance_deg: Angular tolerance in degrees around perpendicular direction.

    Returns:
        Filtered list of OpenCV KeyPoint objects.
    """
    if len(keypoints) == 0:
        return []

    g = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad_ang = np.arctan2(gy, gx) * (180.0 / np.pi) % 180.0

    # Solar shadow vector in 2D plane is along (sun_az_deg + 180) % 360
    # Perpendicular directions are (sun_az_deg ± 90) % 180
    perp_angle_1 = (sun_az_deg + 90.0) % 180.0
    perp_angle_2 = (sun_az_deg - 90.0) % 180.0

    filtered_kps = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < g.shape[1] and 0 <= y < g.shape[0]:
            edge_ang = grad_ang[y, x]
            diff1 = abs(edge_ang - perp_angle_1) % 180.0
            diff2 = abs(edge_ang - perp_angle_2) % 180.0
            min_diff = min(diff1, 180.0 - diff1, diff2, 180.0 - diff2)

            if min_diff <= angle_tolerance_deg:
                filtered_kps.append(kp)

    return filtered_kps if len(filtered_kps) >= 10 else list(keypoints)

