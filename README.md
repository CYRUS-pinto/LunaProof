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
- Feature descriptors fail to find repeatable matches across large sun-angle differences (e.g. 90° or 180° sun flips).

## 3. Method
LunaProof solves illumination breakdown through a multi-stage physical and phase-invariant pipeline:
1. **Lommel–Seeliger Photometric Shading & Ray-Cast Shadows**: Realistic physical rendering of lunar topography.
2. **Phase Congruency Mapping**: Extracting Kovesi log-Gabor phase congruency maps that capture structural edge-ness invariant to illumination and shading.
3. **Multi-Matcher Suite**: Comparing standard SIFT, Phase-Congruency SIFT (PC-SIFT), zero-shot deep matcher (LoFTR), and a custom patch descriptor trained on lunar terrain.
4. **MAGSAC++ Robust Homography & Metrology**: Estimating spatial transforms and measuring sub-pixel registration error at held-out checkpoints (independent of keypoint matches).
5. **Refusal Gate**: Automatically flagging un-reliable or degraded registrations before downstream GIS usage.

## 4. Data
Evaluated on **NASA LRO LOLA LDEM_64** global digital elevation model (real lunar topography at 64 PPD, ~474 m/px). The dataset is strictly partitioned into non-overlapping spatial regions across train, validation, and test splits to guarantee zero data leakage:
- **Train**: Serenitatis, Imbrium, Fecunditatis, Nubium, Procellarum, Descartes Highland, Farside A, Farside B, Hadley
- **Validation**: Tranquillitatis, South Highland
- **Test**: Tycho, Copernicus, Aristarchus, Orientale

## 5. Results
*(Placeholder: To be populated from verified benchmark results in Prompt B)*

- **Success vs. Sun-Angle Gap**: [Pending Colab run completion]
- **Method Comparison**: [Pending Colab run completion]
- **Refusal Gate Reliability**: [Pending Colab run completion]

## 6. Honest Limitations
- Benchmark terrain (~474 m/px) is coarser than high-resolution OHRC (0.25 m).
- Photometric rendering assumes Lommel–Seeliger reflection model without localized regolith albedo maps.
- Evaluated on synthetic sun-angle pairs rendered from real topography; real orbital pair validation is qualitative due to lack of sub-meter ground truth.

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
