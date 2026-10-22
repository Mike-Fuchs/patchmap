# SPDX-FileCopyrightText: 2026 Mike Devin Fuchs
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import rasterio.crs
import rasterio.io
from shapely.geometry.base import BaseGeometry
from patchmap.utils import (
    build_global_raster, build_zones, make_patches, meters_to_pixels, report_sr_adjustments
)


def generate(field_geom, ir, mmr, sr_micro, sr_macro,
             edge_bias=1.5, edge_width=10.0, smoothing_flag=2,
             rng_seed=None, resolution=1.0, crs=None, mode="patch", verbose=True):
    """
    Generate a synthetic infection patch raster for a field.

    The Gaussian-seed smoothing constants (k1–k4) are internal calibration
    values defined in patchmap.utils; they are not exposed as parameters.

    Args:
        field_geom: Shapely geometry of the field boundary (must be in a projected CRS).
        ir (float): Infection rate — fraction of the field area to infect (0..1).
        mmr (float): Micro-to-macro ratio — share of infected cells assigned to micro patches (0..1).
        sr_micro (float): Smoothing radius for micro patches in meters.
        sr_macro (float): Smoothing radius for macro patches in meters.
        edge_bias (float): Multiplier controlling preference for edge vs. core allocation (>1 = more edge).
        edge_width (float): Width of the edge zone in meters.
        smoothing_flag (int): 1 = focal (uniform) smoothing, 2 = Gaussian seed smoothing.
        rng_seed (int, optional): Random seed for reproducibility.
        resolution (float): Output raster resolution in meters.
        crs: Coordinate reference system — any value accepted by rasterio (e.g. 'EPSG:32632').
        mode (str): Application mode. Currently only "patch" is supported.
        verbose (bool): Print a note when the smoothing radii are rounded to odd pixels.

    Returns:
        rasterio.io.MemoryFile: In-memory raster dataset with:
            - 1  = infected
            - 0  = healthy
            - -99 = outside field (nodata)

    Raises:
        TypeError: If field_geom is not a shapely geometry.
        ValueError: If field_geom is empty, or crs is geographic (degrees) —
            generate requires a projected geometry in meters; or if the
            effective sr_micro exceeds the effective sr_macro (in pixels).
    """
    if mode != "patch":
        raise NotImplementedError(f"Mode '{mode}' is not yet implemented.")

    if not isinstance(field_geom, BaseGeometry):
        raise TypeError(
            f"field_geom must be a shapely geometry, got {type(field_geom).__name__}. "
            "Pass a (Multi)Polygon, e.g. gdf.geometry.union_all()."
        )
    if field_geom.is_empty:
        raise ValueError("field_geom is empty.")
    if field_geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise TypeError(f"field_geom must be a Polygon or MultiPolygon, got {field_geom.geom_type}.")
    if crs is not None and rasterio.crs.CRS.from_user_input(crs).is_geographic:
        raise ValueError(
            "crs is geographic (degrees); generate requires a projected CRS in meters "
            "because resolution, edge_width and sr_* are in meters. Reproject to UTM first."
        )

    if verbose:
        report_sr_adjustments(sr_micro, sr_macro, resolution)

    # sr_micro > sr_macro would make "micro" the larger patches and reverse the
    # meaning of mmr; equal values (uniform patch size) are allowed.
    sr_micro_px = meters_to_pixels(sr_micro, resolution)
    sr_macro_px = meters_to_pixels(sr_macro, resolution)
    if sr_micro_px > sr_macro_px:
        raise ValueError(
            f"sr_micro ({sr_micro}m -> {sr_micro_px}px) exceeds sr_macro "
            f"({sr_macro}m -> {sr_macro_px}px) after rounding to odd pixels; "
            "sr_micro must not exceed sr_macro."
        )

    transform, width, height, crs = build_global_raster(field_geom, crs, resolution)
    zones = build_zones(field_geom, transform, width, height, edge_width, resolution)
    patch = make_patches(
        zones, ir, mmr, sr_micro, sr_macro, edge_bias,
        smoothing_flag, rng_seed, resolution
    )

    memfile = rasterio.io.MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.int16,
        crs=crs,
        transform=transform,
        nodata=-99,
    ) as dataset:
        dataset.write(patch, 1)

    return memfile
