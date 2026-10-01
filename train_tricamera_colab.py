"""
LunaProof – Tri-Camera Master Descriptor Training Script
======================================================
Team: Maximus2 (ID: 185903) | PS: SIH26166 (ISRO SAC)
Objective: Train Tri-Camera Phase Descriptor with Triplet Margin Loss (Margin=0.5)
           for multi-modal Chandrayaan-2 scale octaves (OHRC 0.25m, TMC-2 5m, IIRS 80m).

Runs seamlessly on Google Colab (T4 / A100 GPU) or local PyTorch environment.
Outputs:
  - lunaproof_tricamera_best.pt (PyTorch 128-D descriptor checkpoint)
  - fig2_training_curve.png (Triplet Margin Loss + Val Top-1 Retrieval Plot)
"""

import os
import time
import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


def render_lunar_patch(size=64, sun_az=90, sun_el=30, scale_factor=1.0, seed=42):
    rng = np.random.default_rng(seed)
    y, x = np.ogrid[:size, :size]
    dem = np.zeros((size, size), dtype=np.float32)

    cx, cy = size / 2, size / 2
    r = size / 3.2
    depth = 80.0
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    dem -= depth * np.exp(-(dist**2) / (2 * (r/2)**2))

    for k in range(3):
        scx = rng.uniform(12, size - 12)
        scy = rng.uniform(12, size - 12)
        sr = rng.uniform(6, 14)
        sdepth = rng.uniform(25, 55)
        sdist = np.sqrt((x - scx)**2 + (y - scy)**2)
        dem -= sdepth * np.exp(-(sdist**2) / (2 * sr**2))

    az_rad, el_rad = np.radians(sun_az), np.radians(sun_el)
    lx, ly, lz = np.cos(el_rad)*np.sin(az_rad), np.cos(el_rad)*np.cos(az_rad), np.sin(el_rad)
    gx = cv2.Sobel(dem, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dem, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2 + 1.0)
    nx, ny, nz = -gx / mag, -gy / mag, 1.0 / mag
    cos_i = np.clip(nx*lx + ny*ly + nz*lz, 0.0, 1.0)
    img = np.clip((cos_i / (cos_i + nz + 1e-6)) * 255.0, 0, 255).astype(np.uint8)

    if scale_factor != 1.0:
        h_s = max(12, int(size * scale_factor))
        w_s = max(12, int(size * scale_factor))
        low_res = cv2.resize(img, (w_s, h_s), interpolation=cv2.INTER_AREA)
        img = cv2.resize(low_res, (size, size), interpolation=cv2.INTER_LINEAR)

    f = img.astype(np.float32) / 255.0
    k1 = cv2.Scharr(f, cv2.CV_32F, 1, 0)
    k2 = cv2.Scharr(f, cv2.CV_32F, 0, 1)
    mag = np.sqrt(k1**2 + k2**2)
    blur = cv2.GaussianBlur(mag, (5, 5), 1.0)
    pc = mag / (blur + 1e-3)
    return cv2.normalize(pc, None, 0, 1.0, cv2.NORM_MINMAX).astype(np.float32)


if HAS_TORCH:
    class TriCameraLunarDataset(Dataset):
        def __init__(self, num_samples=2400, is_val=False):
            self.num_samples = num_samples
            self.is_val = is_val
            self.seed_offset = 50000 if is_val else 1000

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            seed = self.seed_offset + idx
            az_a = float(np.random.uniform(0, 360))
            el_a = float(np.random.uniform(20, 45))
            anchor = render_lunar_patch(64, sun_az=az_a, sun_el=el_a, scale_factor=1.0, seed=seed)

            az_b = (az_a + 180 + np.random.uniform(-30, 30)) % 360
            el_b = float(np.random.uniform(20, 45))
            scale_b = float(np.random.choice([0.75, 0.5]))
            positive = render_lunar_patch(64, sun_az=az_b, sun_el=el_b, scale_factor=scale_b, seed=seed)

            neg_seed = seed + 10000
            hard_neg = render_lunar_patch(64, sun_az=az_b, sun_el=el_b, scale_factor=scale_b, seed=neg_seed)

            return (
                torch.from_numpy(anchor).unsqueeze(0),
                torch.from_numpy(positive).unsqueeze(0),
                torch.from_numpy(hard_neg).unsqueeze(0),
            )

    class HardNetLunar(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        def forward(self, x):
            desc = self.features(x).view(x.size(0), -1)
            return F.normalize(desc, p=2, dim=1)


def train_lunaproof_model(epochs=15, batch_size=64, lr=1e-3, save_path="lunaproof_tricamera_best.pt"):
    if not HAS_TORCH:
        print("[Error] PyTorch is required to run train_lunaproof_model(). Run in Google Colab or PyTorch environment.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LunaProof] Active Training Device: {device}")

    train_loader = DataLoader(TriCameraLunarDataset(num_samples=2400, is_val=False), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TriCameraLunarDataset(num_samples=600, is_val=True), batch_size=batch_size, shuffle=False)

    model = HardNetLunar().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    triplet_loss = nn.TripletMarginLoss(margin=0.5, p=2)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_top1": []}

    print("\nStarting LunaProof Master Training across IIRS/TMC-2/OHRC octaves...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for anchors, positives, hard_negs in train_loader:
            anchors, positives, hard_negs = anchors.to(device), positives.to(device), hard_negs.to(device)
            z_a = model(anchors)
            z_p = model(positives)
            z_n = model(hard_negs)

            loss = triplet_loss(z_a, z_p, z_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for v_a, v_p, v_n in val_loader:
                v_a, v_p, v_n = v_a.to(device), v_p.to(device), v_n.to(device)
                vz_a, vz_p, vz_n = model(v_a), model(v_p), model(v_n)

                dist_pos = torch.norm(vz_a - vz_p, p=2, dim=1)
                dist_neg = torch.norm(vz_a - vz_n, p=2, dim=1)

                correct += (dist_pos < dist_neg).sum().item()
                total += v_a.size(0)

        val_acc = (correct / total) * 100.0
        history["train_loss"].append(avg_loss)
        history["val_top1"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} | Val Top-1 Pair Retrieval: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    elapsed = time.time() - t_start
    print(f"\n[COMPLETE] Total Time: {elapsed:.1f}s | Best Validation Top-1 Retrieval: {best_val_acc:.2f}%")
    print(f"Saved model checkpoint to: {save_path}")

    if HAS_PLT:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Triplet Margin Loss', color=color)
        ax1.plot(range(1, epochs + 1), history["train_loss"], color=color, marker='o', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Val Top-1 Retrieval (%)', color=color)
        ax2.plot(range(1, epochs + 1), history["val_top1"], color=color, marker='s', linestyle='--', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('LunaProof Tri-Camera Descriptor Training Curve (Triplet Margin Loss)')
        fig.tight_layout()
        plot_path = "fig2_training_curve.png"
        plt.savefig(plot_path, dpi=200)
        print(f"Plot saved: {plot_path}")

    return history


if __name__ == "__main__":
    train_lunaproof_model(epochs=15, batch_size=64)
