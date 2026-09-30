"""
LunaProof – Log-Polar Fourier-Mellin Shift Estimator
===================================================
Provides analytical coarse scale and rotation estimation in the frequency domain
between multi-resolution lunar images (e.g. IIRS 80m vs TMC-2 5m).

Converts magnitude FFT into log-polar space where scale and rotation become 2D shifts.

Absorbed for ISRO PS 26166 scale/rotation pre-alignment.
"""

from __future__ import annotations

import cv2
import numpy as np


def estimate_fourier_mellin_scale_rotation(
    img_ref: np.ndarray,
    img_src: np.ndarray,
) -> tuple[float, float, float]:
    """
    Estimates relative scale factor, rotation angle (degrees), and phase correlation
    confidence between img_ref and img_src using the Fourier-Mellin transform.

    Args:
        img_ref: Reference image (H, W) uint8 or float32.
        img_src: Source image (H, W) uint8 or float32.

    Returns:
        (scale_ratio, rotation_deg, correlation_peak)
    """
    g1 = img_ref if img_ref.ndim == 2 else cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    g2 = img_src if img_src.ndim == 2 else cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    g1_f = g1.astype(np.float32)
    g2_f = g2.astype(np.float32)

    # Apply Hanning window to reduce boundary ringing
    h1, w1 = g1_f.shape
    h2, w2 = g2_f.shape
    win1 = cv2.createHanningWindow((w1, h1), cv2.CV_32F)
    win2 = cv2.createHanningWindow((w2, h2), cv2.CV_32F)

    # 2D FFT & Shift
    f1 = np.fft.fftshift(np.fft.fft2(g1_f * win1))
    f2 = np.fft.fftshift(np.fft.fft2(g2_f * win2))

    mag1 = np.log(np.abs(f1) + 1.0)
    mag2 = np.log(np.abs(f2) + 1.0)

    # Log-polar transformation (OpenCV 4+ warpPolar)
    center = (w1 / 2.0, h1 / 2.0)
    max_radius = min(h1, w1) / 2.0

    if hasattr(cv2, 'warpPolar'):
        lp1 = cv2.warpPolar(mag1, (w1, h1), center, max_radius, cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR)
        lp2 = cv2.warpPolar(mag2, (w1, h1), center, max_radius, cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR)
    elif hasattr(cv2, 'logPolar'):
        M = max_radius / np.log(max_radius + 1e-5)
        lp1 = cv2.logPolar(mag1, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        lp2 = cv2.logPolar(mag2, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    else:
        # Radial projection fallback
        lp1, lp2 = mag1, mag2

    # Phase correlation in log-polar space
    shift, response = cv2.phaseCorrelate(lp1, lp2)

    # Decode angle and scale from log-polar shift
    angle_deg = float((shift[0] * 360.0 / lp1.shape[1]) % 360.0)
    scale_factor = float(np.exp(shift[1] / (max_radius + 1e-5)))

    return scale_factor, angle_deg, float(response)


def align_fourier_mellin(img_src: np.ndarray, img_ref: np.ndarray) -> tuple[np.ndarray, float, float]:
    scale_factor, angle_deg, resp = estimate_fourier_mellin_scale_rotation(img_ref, img_src)
    h, w = img_src.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale_factor)
    aligned_src = cv2.warpAffine(img_src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned_src, angle_deg, scale_factor


