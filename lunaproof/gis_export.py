"""
LunaProof GIS Export — ISRO SAC Geospatial Deliverables
=========================================================
Exports three artefacts per registration run:

  1. Cloud-Optimised GeoTIFF (.tif)
     Registered imagery with selenographic CRS + affine transform.
     Uses IAU 2015 Moon body-fixed CRS (ESRI:104903 / EPSG-equivalent).

  2. Tie-Point Telemetry Table (.csv)
     Columns: point_id, src_x, src_y, ref_x, ref_y, longitude, latitude,
              residual_px, residual_m, is_inlier

  3. Run Diagnostics JSON (.json)
     Execution timestamp, method, sun geometry delta, inlier stats,
     Gini coefficient, spatial confidence score, gate status.

rasterio is an optional dependency; if absent the COG step is skipped
(the CSV and JSON are always produced).

Team Maximus2 (ID 185903) | SIH26166
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np

# Lazy import — rasterio may not be installed in all environments
try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    _HAVE_RASTERIO = True
except ImportError:
    _HAVE_RASTERIO = False


# ---------------------------------------------------------------------------
# IAU 2015 Moon ellipsoid (mean radius 1737.4 km, sphere for SIH purposes)
# WKT for ESRI:104903 — Sphere_Moon_2000
# ---------------------------------------------------------------------------
_MOON_CRS_WKT = (
    'GEOGCS["GCS_Moon_2000",'
    'DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
    'PRIMEM["Reference_Meridian",0.0],'
    'UNIT["Degree",0.0174532925199433]]'
)

_DEG_TO_M_MOON = 2 * 3.14159265358979 * 1_737_400.0 / 360.0


# ---------------------------------------------------------------------------
# 1. Cloud-Optimised GeoTIFF Export
# ---------------------------------------------------------------------------
def export_cog(
    img:          np.ndarray,
    origin_lon:   float,
    origin_lat:   float,
    pixel_size_deg: float,
    output_path:  str,
    band_names:   Optional[list[str]] = None,
) -> str:
    """
    Write a Cloud-Optimised GeoTIFF with selenographic metadata.

    Args:
        img:            2-D (H×W) or 3-D (H×W×C) uint8 or uint16 array.
        origin_lon:     West-edge longitude in degrees (E positive).
        origin_lat:     North-edge latitude in degrees.
        pixel_size_deg: Pixel size in degrees (square pixels assumed).
        output_path:    Output .tif path.
        band_names:     Optional list of band descriptions.

    Returns:
        Absolute path to the written file.
    """
    if not _HAVE_RASTERIO:
        print("[WARN] rasterio not installed — skipping COG export.")
        return ""

    img = np.atleast_3d(img)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    # rasterio expects (bands, height, width)
    arr    = np.moveaxis(img, -1, 0)
    n_bands, h, w = arr.shape

    transform = from_bounds(
        left   = origin_lon,
        bottom = origin_lat - h * pixel_size_deg,
        right  = origin_lon + w * pixel_size_deg,
        top    = origin_lat,
        width  = w, height = h,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_path = output_path + ".tmp.tif"

    with rasterio.open(
        tmp_path, 'w',
        driver   = 'GTiff',
        height   = h, width = w,
        count    = n_bands,
        dtype    = arr.dtype,
        crs      = CRS.from_wkt(_MOON_CRS_WKT),
        transform= transform,
    ) as dst:
        dst.write(arr)
        if band_names:
            for i, name in enumerate(band_names[:n_bands], 1):
                dst.update_tags(i, name=name)

    # Convert to Cloud-Optimised GeoTIFF
    try:
        from rasterio.shutil import copy as rio_copy
        rio_copy(tmp_path, output_path, driver="GTiff",
                 copy_src_overviews=True, tiled=True,
                 compress="DEFLATE", blockxsize=512, blockysize=512)
        os.remove(tmp_path)
    except Exception:
        # Fallback: just rename — still a valid GeoTIFF
        os.rename(tmp_path, output_path)

    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# 2. Tie-Point Telemetry CSV
# ---------------------------------------------------------------------------
def export_tiepoints_csv(
    pts_src:      np.ndarray,       # (N,2) pixel coords in source image
    pts_ref:      np.ndarray,       # (N,2) pixel coords in reference image
    inlier_mask:  np.ndarray,       # (N,) bool
    H:            np.ndarray,       # 3×3 estimated homography
    origin_lon:   float,
    origin_lat:   float,
    pixel_size_deg: float,
    gsd_m:        float,
    output_path:  str,
) -> str:
    """
    Export a tie-point CSV with sub-pixel residuals and geodetic coordinates.

    Geodetic coordinates are derived from the pixel coords of the reference
    image using the affine origin + pixel_size_deg metadata.

    Residuals are computed as ||H(p_src) - p_ref||.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    n = len(pts_src)
    # Estimate in homogeneous coords
    p_src_h  = np.c_[pts_src, np.ones(n)]
    est      = (H @ p_src_h.T).T
    est      = est[:, :2] / est[:, 2:3]
    residuals_px = np.linalg.norm(est - pts_ref, axis=1)
    residuals_m  = residuals_px * gsd_m

    # Reference pixel → selenographic lon/lat
    lons = origin_lon + pts_ref[:, 0] * pixel_size_deg
    lats = origin_lat - pts_ref[:, 1] * pixel_size_deg   # row 0 = north edge

    lines = [
        "point_id,src_x_px,src_y_px,ref_x_px,ref_y_px,"
        "longitude_deg,latitude_deg,residual_px,residual_m,is_inlier"
    ]
    for i in range(n):
        lines.append(
            f"{i},"
            f"{pts_src[i,0]:.4f},{pts_src[i,1]:.4f},"
            f"{pts_ref[i,0]:.4f},{pts_ref[i,1]:.4f},"
            f"{lons[i]:.6f},{lats[i]:.6f},"
            f"{residuals_px[i]:.4f},{residuals_m[i]:.4f},"
            f"{int(inlier_mask[i])}"
        )

    with open(output_path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# 3. Run Diagnostics JSON
# ---------------------------------------------------------------------------
def export_telemetry(
    run_dict:    dict,
    output_path: str,
) -> str:
    """
    Write the registration run diagnostics as an indented JSON file.

    Recommended keys in run_dict:
        timestamp_utc, method, sun_az_ref_deg, sun_az_src_deg, delta_az_deg,
        sun_el_ref_deg, sun_el_src_deg, n_matches, n_inliers, inlier_ratio,
        gini_coefficient, grid_coverage, spatial_confidence_score,
        gate_status, med_err_px, rmse_px, gsd_m, elapsed_s,
        spice_kernels_used, dem_source, tile_region
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Always stamp with current UTC time
    out = {
        "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
        "lunaproof_version": "3.0.0",
        **run_dict,
    }

    with open(output_path, "w") as f:
        json.dump(out, f, indent=4, default=_json_safe)

    return os.path.abspath(output_path)


def _json_safe(obj):
    """JSON serialiser for numpy scalars / arrays."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# 4. Convenience bundle
# ---------------------------------------------------------------------------
def export_isro_gis_package(
    registered_img:   np.ndarray,
    pts_src:          np.ndarray,
    pts_ref:          np.ndarray,
    inlier_mask:      np.ndarray,
    H:                np.ndarray,
    run_meta:         dict,
    output_dir:       str,
    origin_lon:       float = 0.0,
    origin_lat:       float = 0.0,
    pixel_size_deg:   float = 1e-4,
    gsd_m:            float = 0.25,
) -> dict[str, str]:
    """
    Master export function — writes COG GeoTIFF, tie-point CSV, and
    telemetry JSON to *output_dir* in one call.

    Returns a dict of {artifact_name: absolute_path}.
    """
    os.makedirs(output_dir, exist_ok=True)

    cog_path = export_cog(
        registered_img,
        origin_lon=origin_lon, origin_lat=origin_lat,
        pixel_size_deg=pixel_size_deg,
        output_path=os.path.join(output_dir, "LunaProof_Registered.tif"),
    )

    csv_path = export_tiepoints_csv(
        pts_src, pts_ref, inlier_mask, H,
        origin_lon=origin_lon, origin_lat=origin_lat,
        pixel_size_deg=pixel_size_deg, gsd_m=gsd_m,
        output_path=os.path.join(output_dir, "tie_points_telemetry.csv"),
    )

    json_path = export_telemetry(
        run_meta,
        output_path=os.path.join(output_dir, "registration_metrics.json"),
    )

    paths = dict(cog=cog_path, csv=csv_path, json=json_path)
    print(f"[LunaProof GIS] Exported package to: {output_dir}")
    for k, v in paths.items():
        if v:
            print(f"  {k}: {v}")
    return paths
