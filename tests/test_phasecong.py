import numpy as np
from lunaproof.phasecong import phase_congruency, pc_to_u8

def test_phase_congruency_shape_and_range():
    img_u8 = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    pc, M = phase_congruency(img_u8)
    
    assert pc.shape == img_u8.shape
    assert M.shape == img_u8.shape
    assert np.isfinite(pc).all()
    assert (pc >= 0).all()

def test_pc_to_u8():
    pc = np.random.uniform(0, 1, (64, 64)).astype(np.float32)
    u8 = pc_to_u8(pc)
    assert u8.shape == (64, 64)
    assert u8.dtype == np.uint8
