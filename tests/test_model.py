"""
Test Suite — LunaProof HardNetLunar Neural Descriptor
====================================================
Verifies descriptor extraction in both PyTorch and NumPy fallback modes.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaproof.model import extract_patch_descriptors


def test_extract_patch_descriptors_numpy():
    patches = [np.random.default_rng(i).integers(0, 256, (32, 32), dtype=np.uint8) for i in range(5)]
    descs = extract_patch_descriptors(patches, model=None)
    assert descs.shape == (5, 128)
    assert descs.dtype == np.float32
    # Check L2 normalization
    norms = np.linalg.norm(descs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_extract_patch_descriptors_empty():
    descs = extract_patch_descriptors([], model=None)
    assert descs.shape == (0, 128)
