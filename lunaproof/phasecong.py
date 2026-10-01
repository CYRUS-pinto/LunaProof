import math
import numpy as np
import cv2

def phase_congruency(img_u8, nscale=4, norient=6, min_wl=8.0, mult=2.1, sigma_onf=0.55,
                     dtheta_on_sigma=1.3, k=2.0, cutoff=0.5, gain=10.0, pad=32):
    f = img_u8.astype(np.float32) / 255.0
    f = np.pad(f, pad, mode='reflect')
    H, W = f.shape
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    radius = np.sqrt(fx**2 + fy**2); radius[0, 0] = 1.0
    theta = np.arctan2(-fy, fx)
    sint, cost = np.sin(theta), np.cos(theta)
    lowpass = 1.0 / (1.0 + (radius / 0.45) ** 30)
    F = np.fft.fft2(f)
    log_gabor = []
    for s in range(nscale):
        fo = 1.0 / (min_wl * mult ** s)
        lg = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(sigma_onf) ** 2)) * lowpass
        lg[0, 0] = 0
        log_gabor.append(lg)
    theta_sigma = math.pi / norient / dtheta_on_sigma
    pc_sum = np.zeros((H, W), np.float32)
    covx2 = np.zeros_like(pc_sum); covy2 = np.zeros_like(pc_sum); covxy = np.zeros_like(pc_sum)
    for o in range(norient):
        ang = o * math.pi / norient
        ds = sint * math.cos(ang) - cost * math.sin(ang)
        dc = cost * math.cos(ang) + sint * math.sin(ang)
        spread = np.exp(-np.arctan2(ds, dc) ** 2 / (2 * theta_sigma ** 2))
        sumE = np.zeros((H, W), np.float32); sumO = np.zeros_like(sumE); sumAn = np.zeros_like(sumE)
        maxAn = np.zeros_like(sumE); EOs = []; tau = None
        for s in range(nscale):
            EO = np.fft.ifft2(F * log_gabor[s] * spread)
            An = np.abs(EO).astype(np.float32)
            sumAn += An; sumE += EO.real.astype(np.float32); sumO += EO.imag.astype(np.float32)
            maxAn = np.maximum(maxAn, An)
            if s == 0: tau = np.median(An) / math.sqrt(math.log(4))
            EOs.append(EO)
        xen = np.sqrt(sumE**2 + sumO**2) + 1e-4
        mE, mO = sumE / xen, sumO / xen
        energy = np.zeros((H, W), np.float32)
        for EO in EOs:
            energy += (EO.real * mE + EO.imag * mO).astype(np.float32) - np.abs(EO.real * mO - EO.imag * mE).astype(np.float32)
        T = tau * math.sqrt(math.pi / 2) + k * tau * math.sqrt((4 - math.pi) / 2)
        energy = np.maximum(energy - T, 0)
        width = (sumAn / (maxAn + 1e-4) - 1) / (nscale - 1)
        weight = 1.0 / (1.0 + np.exp((cutoff - width) * gain))
        pc = weight * energy / (sumAn + 1e-4)
        pc_sum += pc
        covx2 += (pc * math.cos(ang)) ** 2; covy2 += (pc * math.sin(ang)) ** 2; covxy += (pc * math.cos(ang)) * (pc * math.sin(ang))
    a, b, c = covx2, covxy, covy2
    M = 0.5 * (a + c + np.sqrt(4 * b**2 + (a - c) ** 2))       # max moment: edge-ness
    sl = (slice(pad, H - pad), slice(pad, W - pad))
    return (pc_sum / norient)[sl], M[sl]

def pc_to_u8(pc, blur=2.0):
    pc = cv2.GaussianBlur(pc.astype(np.float32), (0, 0), blur)
    hi = np.percentile(pc, 99.5) + 1e-9
    return np.clip(pc / hi * 255, 0, 255).astype(np.uint8)

def extract_phase_congruency(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img_u8 = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    else:
        img_u8 = img
    pc, _ = phase_congruency(img_u8)
    return pc_to_u8(pc)

