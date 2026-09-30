"""
LunaProof – Checkerboard Visual Overlay Generator
=================================================
Generates an alternating 8x8 checkerboard overlay of target and aligned source images
to visually verify register fit quality (absorbed from Team ByteHats LUNARIS X).
"""

from __future__ import annotations

import numpy as np
import cv2


def draw_checkerboard_overlay(
    img_target: np.ndarray,
    img_aligned: np.ndarray,
    grid_size: int = 8,
) -> np.ndarray:
    """
    Creates an alternating grid_size x grid_size checkerboard overlay between
    img_target and img_aligned. Continuous edges across grid lines indicate
    high-precision geometric registration.

    Args:
        img_target  : Target image (H, W) or (H, W, 3) uint8.
        img_aligned : Aligned source image warped into target frame (same size).
        grid_size   : Number of checkerboard cells per axis (default 8).

    Returns:
        Combined uint8 image with alternating checkerboard blocks.
    """
    assert img_target.shape == img_aligned.shape, "Images must have identical dimensions"
    H, W = img_target.shape[:2]
    
    cell_h = H / grid_size
    cell_w = W / grid_size
    
    output = img_target.copy()
    
    for row in range(grid_size):
        for col in range(grid_size):
            if (row + col) % 2 == 1:
                r_start = int(row * cell_h)
                r_end   = int((row + 1) * cell_h)
                c_start = int(col * cell_w)
                c_end   = int((col + 1) * cell_w)
                
                output[r_start:r_end, c_start:c_end] = img_aligned[r_start:r_end, c_start:c_end]
                
    # Draw subtle grid boundary lines
    for row in range(1, grid_size):
        r_pos = int(row * cell_h)
        cv2.line(output, (0, r_pos), (W, r_pos), (255, 255, 255), 1)
    for col in range(1, grid_size):
        c_pos = int(col * cell_w)
        cv2.line(output, (c_pos, 0), (c_pos, H), (255, 255, 255), 1)
        
    return output
