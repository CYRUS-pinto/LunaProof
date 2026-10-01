"""Tests for GIS export and SPICE bridge (lunaproof/gis_export.py, spice_bridge.py)."""

import json
import os
import tempfile

import numpy as np
import pytest
from lunaproof.gis_export import (
    _json_safe,
    export_cog,
    export_tiepoints_csv,
    export_telemetry,
    export_isro_gis_package,
)
from lunaproof.spice_bridge import (
    FootprintPolygon,
    SolarGeometry,
    estimate_footprint_overlap,
    resolve_pair_geometry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_img(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.randint(0, 256, (h, w), dtype=np.uint8)


def _fake_pts(n: int = 30, size: float = 640.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    pts_src = rng.uniform(0, size, (n, 2)).astype(np.float32)
    pts_ref = pts_src + rng.normal(0, 0.5, (n, 2)).astype(np.float32)
    inliers = rng.integers(0, 2, n).astype(bool)
    H       = np.eye(3, dtype=np.float64)
    return pts_src, pts_ref, inliers, H


# ---------------------------------------------------------------------------
# JSON serialiser
# ---------------------------------------------------------------------------

def test_json_safe_numpy_int():
    assert _json_safe(np.int64(42)) == 42


def test_json_safe_numpy_float():
    assert _json_safe(np.float32(3.14)) == pytest.approx(3.14, abs=1e-4)


def test_json_safe_numpy_array():
    arr = np.array([1.0, 2.0, 3.0])
    assert _json_safe(arr) == [1.0, 2.0, 3.0]


def test_json_safe_unknown_raises():
    with pytest.raises(TypeError):
        _json_safe(object())


# ---------------------------------------------------------------------------
# Telemetry JSON
# ---------------------------------------------------------------------------

def test_export_telemetry_creates_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "run.json")
        out  = export_telemetry(
            {"gate_status": "SUCCESS", "n_inliers": 120, "gini": 0.30},
            output_path=path,
        )
        assert os.path.isfile(out)
        with open(out) as f:
            data = json.load(f)
        assert data["gate_status"] == "SUCCESS"
        assert data["n_inliers"] == 120
        assert "timestamp_utc" in data
        assert "lunaproof_version" in data


def test_export_telemetry_numpy_values():
    """Numpy scalars must serialise without TypeError."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "run2.json")
        export_telemetry(
            {
                "n_inliers": np.int32(55),
                "gini":      np.float32(0.42),
                "H":         np.eye(3),
            },
            output_path=path,
        )
        with open(path) as f:
            data = json.load(f)
        assert data["n_inliers"] == 55


# ---------------------------------------------------------------------------
# Tie-point CSV
# ---------------------------------------------------------------------------

def test_export_tiepoints_csv_format():
    pts_src, pts_ref, inliers, H = _fake_pts(20)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tp.csv")
        out  = export_tiepoints_csv(
            pts_src, pts_ref, inliers, H,
            origin_lon=0.0, origin_lat=0.0, pixel_size_deg=1e-4,
            gsd_m=0.25, output_path=path,
        )
        assert os.path.isfile(out)
        with open(out) as f:
            lines = f.readlines()
        # header + 20 data rows
        assert len(lines) == 21
        header = lines[0].strip()
        assert "point_id" in header
        assert "longitude_deg" in header
        assert "residual_px" in header
        assert "is_inlier" in header
        # Each data row: 10 fields
        for row in lines[1:]:
            assert len(row.split(",")) == 10


def test_export_tiepoints_residuals_near_zero_for_identity():
    """Identity homography → residuals should be essentially 0."""
    n       = 15
    pts     = np.random.rand(n, 2).astype(np.float32) * 100
    inliers = np.ones(n, bool)
    H       = np.eye(3, dtype=np.float64)
    with tempfile.TemporaryDirectory() as d:
        path = export_tiepoints_csv(
            pts, pts, inliers, H,
            origin_lon=0.0, origin_lat=0.0, pixel_size_deg=1e-4,
            gsd_m=0.25, output_path=os.path.join(d, "tp.csv"),
        )
        with open(path) as f:
            rows = f.readlines()[1:]
        for row in rows:
            fields    = row.split(",")
            res_px    = float(fields[7])
            assert res_px < 0.01, f"Residual should be ~0 for identity H, got {res_px}"


# ---------------------------------------------------------------------------
# COG export (skips gracefully if rasterio absent)
# ---------------------------------------------------------------------------

def test_export_cog_skips_without_rasterio(monkeypatch):
    """If rasterio is not installed, export_cog should print a warning and return ''."""
    import lunaproof.gis_export as ge
    monkeypatch.setattr(ge, "_HAVE_RASTERIO", False)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.tif")
        result = export_cog(_fake_img(), 0.0, 0.0, 1e-4, path)
    assert result == ""


def test_export_cog_with_rasterio():
    """If rasterio is installed, export_cog should create a .tif file."""
    pytest.importorskip("rasterio")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.tif")
        out  = export_cog(_fake_img(), origin_lon=20.0, origin_lat=5.0,
                          pixel_size_deg=1e-4, output_path=path)
        assert os.path.isfile(out)
        assert out.endswith(".tif") or out.endswith(".tif")


# ---------------------------------------------------------------------------
# Full GIS package bundle
# ---------------------------------------------------------------------------

def test_export_isro_gis_package_creates_csv_and_json():
    """At minimum, CSV and JSON must be created regardless of rasterio."""
    pts_src, pts_ref, inliers, H = _fake_pts(25)
    meta = {"gate_status": "SUCCESS", "method": "PC+Learned"}
    with tempfile.TemporaryDirectory() as d:
        paths = export_isro_gis_package(
            registered_img = _fake_img(),
            pts_src        = pts_src,
            pts_ref        = pts_ref,
            inlier_mask    = inliers,
            H              = H,
            run_meta       = meta,
            output_dir     = d,
        )
        assert os.path.isfile(paths["csv"])
        assert os.path.isfile(paths["json"])


# ---------------------------------------------------------------------------
# SPICE bridge — footprint overlap
# ---------------------------------------------------------------------------

def test_overlap_identical_footprints():
    corners = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    fp = FootprintPolygon(corners_lonlat=corners)
    ov = estimate_footprint_overlap(fp, fp)
    assert ov == pytest.approx(1.0, abs=0.01)


def test_overlap_non_overlapping_footprints():
    fp1 = FootprintPolygon(corners_lonlat=[(0, 1), (1, 1), (1, 0), (0, 0)])
    fp2 = FootprintPolygon(corners_lonlat=[(5, 6), (6, 6), (6, 5), (5, 5)])
    ov  = estimate_footprint_overlap(fp1, fp2)
    assert ov == pytest.approx(0.0, abs=0.01)


def test_overlap_partial_footprints():
    fp1 = FootprintPolygon(corners_lonlat=[(0, 2), (2, 2), (2, 0), (0, 0)])
    fp2 = FootprintPolygon(corners_lonlat=[(1, 2), (3, 2), (3, 0), (1, 0)])
    ov  = estimate_footprint_overlap(fp1, fp2)
    assert 0.0 < ov < 1.0


def test_resolve_pair_geometry_fallback():
    """Without XML or SPICE, resolver should return fallback geometry."""
    pair = resolve_pair_geometry(
        fallback_az_ref=45.0, fallback_el_ref=30.0,
        fallback_az_src=225.0, fallback_el_src=30.0,
    )
    assert pair.ref.source == "fallback"
    assert pair.src.source == "fallback"
    assert pair.delta_az_deg == pytest.approx(180.0, abs=0.01)
    assert pair.overlap_frac == pytest.approx(1.0)   # no footprints → assume overlap


def test_resolve_geometry_geometry_pair_fields():
    pair = resolve_pair_geometry(
        fallback_az_ref=0.0, fallback_el_ref=35.0,
        fallback_az_src=90.0, fallback_el_src=10.0,
    )
    assert hasattr(pair, 'delta_az_deg')
    assert hasattr(pair, 'overlap_frac')
    assert 0.0 <= pair.overlap_frac <= 1.0
    assert pair.delta_az_deg == pytest.approx(90.0, abs=0.01)


def test_sha256_fingerprint():
    from lunaproof.gis_export import sha256_fingerprint
    img = np.zeros((64, 64), dtype=np.uint8)
    h1 = sha256_fingerprint(img)
    assert len(h1) == 64
    assert isinstance(h1, str)
    
    # Check that metadata alters hash deterministically
    h2 = sha256_fingerprint(img, {"sensor": "OHRC"})
    assert h1 != h2

