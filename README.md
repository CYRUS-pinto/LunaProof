# LunaProof — Moon Image Registration Under Extreme Illumination Shift

**SIH 2026 / ISRO SIH26166** · Team **Maximus2** (ID 185903)

LunaProof provides physically verified, ground-truth-evaluated image registration for lunar surface observation across extreme sun-angle changes (~300x scale gap, multi-modal spectral differences).

---

## 1. Problem
Lunar orbital images (e.g. Chandrayaan-2 OHRC at 0.25 m, TMC-2 at 5 m, IIRS at ~80 m) exhibit severe geometric and photometric discrepancies. Because the Moon has no atmosphere, there is no Rayleigh scattering or atmospheric fill light. When the solar azimuth or elevation angle changes, cast shadow boundaries flip completely, invalidating classical intensity-gradient feature descriptors.

## 2. Why the Moon Breaks SIFT
Standard keypoint matchers such as SIFT rely heavily on local intensity gradients. On lunar terrain:
- Shading changes dynamically according to the Lommel-Seeliger scattering law.
- Shadow edges move or reverse, causing gradient orientation vectors to rotate by 180 degrees.
- Under a 180° sun-angle flip, classical SIFT drops to a 0.0% success rate on held-out test terrain.

![Why SIFT Fails](assets/why_sift_fails.png)

## 3. Method
LunaProof solves illumination breakdown through a multi-stage physical and phase-invariant pipeline:
1. **Lommel–Seeliger Photometric Shading & Ray-Cast Shadows**: Realistic physical rendering of lunar topography.
2. **Phase Congruency Mapping**: Extracting Kovesi log-Gabor phase congruency maps that capture structural edge-ness invariant to illumination and shading.
3. **Multi-Matcher Suite**: Comparing standard SIFT, Phase-Congruency SIFT (PC-SIFT), zero-shot deep matcher (LoFTR), and a custom patch descriptor trained on lunar terrain.
4. **MAGSAC++ Robust Homography & Metrology**: Estimating spatial transforms and measuring sub-pixel registration error at held-out checkpoints (independent of keypoint matches).
5. **Refusal Gate**: Automatically flagging un-reliable or degraded registrations before downstream GIS usage.

![Pipeline Flow](assets/flow_pipeline.png)

## 4. Data
Evaluated on **NASA LRO LOLA LDEM_64** global digital elevation model (real lunar topography at 64 PPD, ~474 m/px). The dataset is strictly partitioned into 15 non-overlapping spatial regions across train, validation, and test splits (112 test pairs total) to guarantee zero data leakage:
- **Train**: Serenitatis, Imbrium, Fecunditatis, Nubium, Procellarum, Descartes Highland, Farside A, Farside B, Hadley
- **Validation**: Tranquillitatis, South Highland
- **Test**: Tycho, Copernicus, Aristarchus, Orientale

---

## 5. Results (Verified Held-Out Benchmark)

All metrics were computed on held-out test regions (`Tycho`, `Copernicus`, `Aristarchus`, `Orientale`) across 112 evaluation pairs and verified against exact ground truth (`results/results_all_pairs.csv` & `results/summary.json`).

![Success vs Sun Gap](results/fig3_success_vs_sun_gap.png)

### Method Performance Summary (Held-Out Test Set)

| Method | Overall Success Rate (< 2.0 px) | 0° Sun Gap | 60° Sun Gap | 120° Sun Gap | 180° Sun Gap | Median Error (px) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SIFT Baseline** | 28.57% | 100.0% | 0.0% | 0.0% | 0.0% | ∞ |
| **LoFTR (Pretrained)** | 73.21% | 100.0% | 100.0% | 50.0% | 43.75% | 1.154 |
| **Phase-Congruency SIFT (PC-SIFT)** | 88.39% | 100.0% | 93.75% | 62.50% | 100.0% | 0.685 |
| **PC + Learned Descriptor (PatchNet)** | **98.21%** | **100.0%** | **100.0%** | **93.75%** | **100.0%** | **0.685** |

### Refusal Gate Reliability

The refusal gate operates using strictly observable quantities without ground-truth knowledge (inlier count, inlier ratio, spatial coverage):

| Gate Status | Criteria | Ground Truth Correctness (< 2.0 px) |
|---|---|:---:|
| **SUCCESS** | Inliers ≥ 40 and Coverage ≥ 30% | **93.60%** |
| **DEGRADED** | Inliers 15-39 or Coverage 15-29% | 75.00% |
| **REFUSED** | Inliers < 15 or Inlier Ratio < 15% or Coverage < 15% | 15.70% |

### Qualitative Registration Showcase

![Registration Showcase](results/fig5_showcase.png)

---

### Where it FAILS

Honest evaluation reveals specific illumination regimes where registration breakdown occurs (success rate < 50.0%):

1. **Classical SIFT Breakdown**:
   - **All Sun Gaps ≥ 60° (60°, 90°, 120°, 150°, 180°)**: Classical SIFT drops to a **0.0% success rate** across both high sun (35°/30°) and low sun (12°/10°) cases. Intensity gradient flips cause matching to fail entirely.
2. **LoFTR Zero-Shot Breakdown**:
   - **120° Sun Gap (High Sun)**: Drops to **37.5% success rate**.
   - **150° Sun Gap (High Sun)**: Drops to **37.5% success rate**.
   - **150° Sun Gap (Low Sun)**: Drops to **25.0% success rate**.
   - **180° Sun Gap (High Sun)**: Drops to **37.5% success rate**.
   *(LoFTR was pretrained on terrestrial outdoor scenes and struggles with extreme low-elevation lunar cast shadows).*
3. **PC-SIFT Dip at 90° & 120° Gaps**:
   - **90° Sun Gap**: Drops to **62.5% success rate**.
   - **120° Sun Gap (Low Sun)**: Dips to **50.0% success rate** (due to grazing-incidence shadow elongation obscuring crater rims).

---

## 6. Honest Limitations
- Benchmark terrain (~474 m/px) is coarser than high-resolution OHRC (0.25 m).
- Photometric rendering assumes Lommel–Seeliger reflection model without localized regolith albedo maps.
- Evaluated on synthetic sun-angle pairs rendered from real topography; real orbital pair validation is qualitative due to lack of sub-meter ground truth.
- Learned PatchNet descriptor is not invariant to rotations beyond ±20 degrees without prior coarse alignment.

## 7. How to Run in Colab
1. Open `LunaProof_Real_Benchmark_Train.ipynb` in Google Colab.
2. Set Runtime type to **T4 GPU** (`Runtime > Change runtime type`).
3. Click **Runtime > Run all** (Total runtime ~35–50 minutes).
4. Download `lunaproof_results.zip` once execution completes.

## 8. Roadmap
- [ ] **SPICE / PDS4 Ingestion**: Native kernel integration for ancillary geometry.
- [ ] **OHRC Integration**: High-resolution 0.25 m/px imagery pipeline.
- [ ] **IIRS Spectral Bridge**: Multi-spectral band registration.
- [ ] **Scale Cascade**: Hop ratios (80 m IIRS → 5 m TMC-2 → 1 m LRO NAC → 0.25 m OHRC).

![Scale Cascade](assets/flow_cascade.png)
