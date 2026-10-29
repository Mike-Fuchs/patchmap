# SPDX-FileCopyrightText: 2026 Mike Devin Fuchs
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import tempfile

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.transform
from rasterio.warp import reproject, calculate_default_transform, Resampling
from skimage.measure import label, regionprops
from exactextract import exact_extract
from patchmap.utils import build_global_raster, shape_to_raster, detect_grid


# Internal calibration constant: area percentile separating micro from macro
# patches (0..1). Not exposed as a public parameter — change here to adjust
# package-wide behaviour.
_PERCENTILE_CUTOFF = 0.95


def analyze(field_gdf, weed_gdf, edge_width=10.0):
    """
    Analyze observed weed/infection patterns and extract spatial metrics.

    Args:
        field_gdf (GeoDataFrame): Field boundary polygon(s).
        weed_gdf (GeoDataFrame): Observed weed/infection polygons.
        edge_width (float): Edge zone width in meters (used for edge_bias computation).

    Returns:
        dict: Metrics ready for use as inputs to generate():
            - ir (float): Infection rate — fraction of field area infected.
            - mmr (float): Micro-to-macro ratio.
            - sr_micro (float): Area-weighted mean equivalent diameter of micro patches in meters.
            - sr_macro (float): Area-weighted mean equivalent diameter of macro patches in meters.
            - edge_bias (float): Ratio of infection rate in edge zone vs. core zone.

    Raises:
        TypeError: If field_gdf or weed_gdf is not a GeoDataFrame.
        ValueError: If either GeoDataFrame has no CRS defined.
    """
    for name, gdf in (("field_gdf", field_gdf), ("weed_gdf", weed_gdf)):
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError(f"{name} must be a GeoDataFrame, got {type(gdf).__name__}.")
        if gdf.crs is None:
            raise ValueError(f"{name} has no CRS; assign one before calling analyze().")

    ox, oy, resolution, case, target_crs = detect_grid(weed_gdf)

    if case == 1:
        field_mask, field_geom_utm, weed_binary, ir_raster, transform, width, height, crs, res_m = \
            _rasterize_case1(field_gdf, weed_gdf, ox, oy, resolution, target_crs)
    elif case == 2:
        field_mask, field_geom_utm, weed_binary, ir_raster, transform, width, height, crs, res_m = \
            _rasterize_case2(field_gdf, weed_gdf, ox, oy, resolution, target_crs)
    else:
        field_mask, field_geom_utm, weed_binary, ir_raster, transform, width, height, crs, res_m = \
            _rasterize_case3(field_gdf, weed_gdf, ox, oy, resolution, target_crs)

    infection_binary = (weed_binary == 1) & field_mask
    infection_ir = ir_raster * field_mask

    ir = _compute_ir(infection_ir, field_mask)
    edge_bias = _compute_edge_bias(field_geom_utm, infection_ir, field_mask, edge_width, transform, width, height)
    mmr, sr_micro, sr_macro = _compute_patch_metrics(infection_binary, res_m, _PERCENTILE_CUTOFF)

    return {
        "ir": ir,
        "mmr": mmr,
        "sr_micro": sr_micro,
        "sr_macro": sr_macro,
        "edge_bias": edge_bias,
    }


def _rasterize_case1(field_gdf, weed_gdf, ox, oy, res_deg, utm_crs):
    field_wgs = field_gdf.to_crs("EPSG:4326")
    field_geom_wgs = field_wgs.geometry.union_all()
    weed_geom_wgs = weed_gdf.to_crs("EPSG:4326").geometry.union_all()

    # Rasterize weed in WGS84 aligned to detected grid, covering the full field extent
    src_transform, src_w, src_h, _ = build_global_raster(field_geom_wgs, "EPSG:4326", res_deg, origin=(ox, oy))
    weed_arr_wgs = shape_to_raster(weed_geom_wgs, src_transform, src_w, src_h)

    # Reproject weed raster to UTM
    left = src_transform.c
    top = src_transform.f
    right = left + src_w * src_transform.a
    bottom = top + src_h * src_transform.e
    dst_transform, dst_w, dst_h = calculate_default_transform(
        "EPSG:4326", utm_crs, src_w, src_h, left=left, bottom=bottom, right=right, top=top
    )
    weed_arr_utm = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(
        source=weed_arr_wgs,
        destination=weed_arr_utm,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_transform,
        dst_crs=utm_crs,
        resampling=Resampling.nearest,
    )

    field_geom_utm = field_gdf.to_crs(utm_crs).geometry.union_all()
    field_arr = shape_to_raster(field_geom_utm, dst_transform, dst_w, dst_h).astype(bool)
    res_m = abs(dst_transform.a)

    return field_arr, field_geom_utm, weed_arr_utm, weed_arr_utm.astype(float), dst_transform, dst_w, dst_h, utm_crs, res_m


def _rasterize_case2(field_gdf, weed_gdf, ox, oy, res_m, utm_crs):
    field_geom = field_gdf.to_crs(utm_crs).geometry.union_all()
    weed_geom = weed_gdf.to_crs(utm_crs).geometry.union_all()

    transform, width, height, _ = build_global_raster(field_geom, utm_crs, res_m, origin=(ox, oy))
    field_arr = shape_to_raster(field_geom, transform, width, height).astype(bool)
    weed_arr = shape_to_raster(weed_geom, transform, width, height)

    return field_arr, field_geom, weed_arr, weed_arr.astype(float), transform, width, height, utm_crs, res_m


def _rasterize_case3(field_gdf, weed_gdf, ox, oy, res_m, utm_crs):
    field_geom = field_gdf.to_crs(utm_crs).geometry.union_all()
    weed_geom = weed_gdf.to_crs(utm_crs).geometry.union_all()

    transform, width, height, _ = build_global_raster(field_geom, utm_crs, res_m, origin=(ox, oy))
    field_arr = shape_to_raster(field_geom, transform, width, height).astype(bool)
    weed_binary = shape_to_raster(weed_geom, transform, width, height)

    # Proportional coverage via exactextract — exact per-cell coverage fractions
    weed_union_gdf = gpd.GeoDataFrame(geometry=[weed_geom], crs=utm_crs)
    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp.close()
    try:
        with rasterio.open(tmp.name, "w", driver="GTiff", height=height, width=width,
                           count=1, dtype="float32", crs=utm_crs, transform=transform) as ds:
            ds.write(np.ones((1, height, width), dtype=np.float32))

        result = exact_extract(tmp.name, weed_union_gdf,
                               ops=["cell_id", "coverage"], output="pandas")

        prop_flat = np.zeros(height * width, dtype=float)
        all_ids = np.concatenate(result["cell_id"].values).astype(int)
        all_covs = np.concatenate(result["coverage"].values).astype(float)
        np.add.at(prop_flat, all_ids, all_covs)
        weed_proportional = prop_flat.reshape(height, width)
    finally:
        os.unlink(tmp.name)

    return field_arr, field_geom, weed_binary, weed_proportional, transform, width, height, utm_crs, res_m


def _compute_ir(infection, field_mask):
    n_field = int(field_mask.sum())
    if n_field == 0:
        return np.nan
    return float(infection.sum() / n_field)


def _compute_edge_bias(field_geom, infection, field_mask, edge_width, transform, width, height):
    # Edge zone uses the full field (no half-pixel shrink) so every observed
    # cell counts. This intentionally differs from generate's build_zones, which
    # shrinks inward to avoid overspray. The asymmetry is by design.
    core_geom = field_geom.buffer(-edge_width)
    if core_geom.is_empty:
        return np.nan

    core_raster = shape_to_raster(core_geom, transform, width, height).astype(bool)
    edge_mask = field_mask & ~core_raster

    ir_edge = float(infection[edge_mask].sum() / edge_mask.sum()) if edge_mask.sum() > 0 else np.nan
    ir_core = float(infection[core_raster].sum() / core_raster.sum()) if core_raster.sum() > 0 else np.nan

    if np.isnan(ir_core) or ir_core <= 0 or np.isnan(ir_edge):
        return np.nan
    return ir_edge / ir_core


def _compute_patch_metrics(infection, resolution, percentile_cutoff):
    labeled = label(infection.astype(np.uint8), connectivity=2)
    props = regionprops(labeled)

    if not props:
        return np.nan, np.nan, np.nan

    areas_px = np.array([p.area for p in props])
    areas_m2 = areas_px * (resolution ** 2)

    cutoff = np.percentile(areas_m2, percentile_cutoff * 100)
    micro = areas_m2 <= cutoff
    macro = ~micro

    total_area = areas_m2.sum()
    mmr = float(areas_m2[micro].sum() / total_area) if total_area > 0 else np.nan

    # Area-weighted: a plain per-patch mean is dominated by the many 1-2 cell
    # fragments at patch fringes, which carry little area. Weighting by area
    # gives the size of the patch a typical infected cell belongs to.
    diameters_m = 2 * np.sqrt(areas_m2 / np.pi)
    sr_micro = float(np.average(diameters_m[micro], weights=areas_m2[micro])) if micro.any() else np.nan
    sr_macro = float(np.average(diameters_m[macro], weights=areas_m2[macro])) if macro.any() else np.nan

    return mmr, sr_micro, sr_macro
