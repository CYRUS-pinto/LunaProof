# LunaProof v3 — Execution Checklist

## Layer 1: Bounded Scale Cascade
- [x] `lunaproof/cascade.py` — NMI+phase-corr Hop1, PC-SIFT Hop2, PatchNet+ECC Hop3
- [x] `lunaproof/dem_io.py` — `downsample_tile()` in cascade.py `make_synthetic_tiers()`
- [x] `tests/test_cascade.py` — 3-hop integration test on synthetic tiles

## Layer 2: Gini Gate Upgrade
- [x] `lunaproof/match.py` — `calculate_spatial_gini()`, `gate_v2()`, `spatial_confidence()`
- [x] `tests/test_gate_gini.py` — clustered vs. spread Gini test

## Layer 3: GIS Exporter
- [x] `lunaproof/gis_export.py` — COG GeoTIFF + tie-point CSV + JSON telemetry
- [x] `tests/test_gis_export.py` — telemetry schema + CSV format test

## Layer 4: SpiceyPy Bridge
- [x] `lunaproof/spice_bridge.py` — PDS4 XML primary, SPICE BSP fallback

## Housekeeping
- [x] `requirements.txt` — add rasterio, spiceypy
- [x] `README.md` — v3 cascade diagram + gate v2 numbers
- [/] `git commit` — running tests first
- [ ] All tests green (target: 13 existing + 8+ new = 21+)
