# SPDX-FileCopyrightText: 2026 Mike Devin Fuchs
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.features
import rasterio.transform
from scipy.ndimage import uniform_filter, gaussian_filter


# --- Internal calibration constants ---
# Gaussian-seed smoothing tuning (smoothing_flag=2 in make_patches). Calibrated to
# reproduce observed patch structure; not exposed as public parameters. Change here
# to adjust package-wide behaviour.
_K1 = 3.0   # macro seed density scaling (higher = denser seeds)
_K2 = 3.0   # macro blob core scaling (higher = tighter blobs)
_K3 = 0.2   # background noise level between macro patches
_K4 = 3.0   # final macro smoothing strength

# Smallest cell accepted as a genuine grid. Below this a "grid" is coordinate
# rounding or float noise, not an observation raster.
_MIN_CELL_M = 0.05
_M_PER_DEG = 111_320.0


# --- Grid detection ---

def detect_grid(gdf, n_sample=500, squareness_tol=0.05):
    """
    Detect whether weed observation polygons lie on a regular square grid.

    A grid only exists in the CRS it was created in, so the input's native CRS
    is tested first (geographic -> case 1, projected -> case 2) and the other
    one only as fallback.

    Args:
        gdf (GeoDataFrame): Weed observation polygons.
        n_sample (int): Number of polygons to sample for detection.
        squareness_tol (float): Relative tolerance for dx ≈ dy check (default 5%).

    Returns:
        tuple: (origin_x, origin_y, resolution, case, target_crs)
            - target_crs: the single metric CRS to use for all downstream
              analysis, chosen once here so every rasterize step shares it:
              the input CRS for case 2 if it is projected in meters, else the
              estimated UTM CRS
            - origin and resolution are in the *grid* CRS units: degrees for
              case 1 (rasterized in WGS84, then reprojected to target_crs),
              target_crs meters for cases 2 and 3
            - case: 1 (square in WGS84), 2 (square in a metric CRS), 3 (neither)
            - case 3: resolution defaults to 1.0m, origin from weed data bounds
    """
    utm_crs = gdf.estimate_utm_crs()
    sample = gdf.sample(min(n_sample, len(gdf)), random_state=42)

    native_geo = sample.crs.is_geographic
    native_metric = not native_geo and sample.crs.axis_info[0].unit_name == "metre"
    metric_crs = sample.crs if native_metric else utm_crs

    for case in ((1, 2) if native_geo else (2, 1)):
        if case == 1:
            # case 1: check in WGS84
            gdf_wgs = sample.to_crs("EPSG:4326") if sample.crs.to_epsg() != 4326 else sample
            result = _check_square_grid(gdf_wgs, squareness_tol, _MIN_CELL_M / _M_PER_DEG)
            target_crs = utm_crs
        else:
            # case 2: check in the metric CRS
            result = _check_square_grid(sample.to_crs(metric_crs), squareness_tol, _MIN_CELL_M)
            target_crs = metric_crs
        if result is not None:
            ox, oy, res = result
            return ox, oy, res, case, target_crs

    # case 3: default to 1m in UTM, origin from weed data bounds
    gdf_utm = gdf.to_crs(utm_crs)
    bounds = gdf_utm.total_bounds  # xmin, ymin, xmax, ymax
    return float(bounds[0]), float(bounds[1]), 1.0, 3, utm_crs


def _check_square_grid(gdf, squareness_tol, min_cell):
    """
    Extract vertex coordinates, fit a lattice per axis and check squareness.
    Returns (origin_x, origin_y, resolution) or None.
    """
    xs, ys = [], []
    for geom in gdf.geometry:
        parts = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
        for part in parts:
            for x, y in part.exterior.coords:
                xs.append(round(x, 8))
                ys.append(round(y, 8))

    xs = np.array(sorted(set(xs)))
    ys = np.array(sorted(set(ys)))

    fit_x = _fit_lattice(xs, min_cell)
    fit_y = _fit_lattice(ys, min_cell)

    if fit_x is None or fit_y is None:
        return None

    (res_x, phase_x), (res_y, phase_y) = fit_x, fit_y

    if abs(res_x - res_y) / max(res_x, res_y) > squareness_tol:
        return None

    resolution = (res_x + res_y) / 2

    # Snap the minimum coordinate to the nearest valid grid point.
    origin_x = phase_x + np.round((xs.min() - phase_x) / res_x) * res_x
    origin_y = phase_y + np.round((ys.min() - phase_y) / res_y) * res_y

    return float(origin_x), float(origin_y), float(resolution)


def _fit_lattice(coords, min_cell, on_grid_tol=0.25, min_on_grid=0.9):
    """
    Cell size and phase of the 1-D lattice the coordinates lie on, or None.

    Candidate cell size: the median of the gaps close to the smallest gap
    between sorted unique coordinates that is at least min_cell (smaller gaps
    are rounding or float noise, e.g. the rounding quantum exposed when
    projected data is reprojected to WGS84). The smallest gap alone can be a
    little short in rounded source data, and that error accumulates over many
    cells, so cell size and phase are then refined by a linear fit of the
    coordinates against their cell index.

    Accepted only if at least `min_on_grid` of the coordinates sit within
    `on_grid_tol` cells of phase + k * cell, i.e. the data lies on a grid by
    definition rather than by a gap statistic. A vertex within 0.25 cells is
    closer to its grid line than to the midpoint between two lines, so it
    snaps unambiguously; off-grid coordinates are spread evenly and only ~50%
    fall within that tolerance. Real grids drift from rounding in the source
    data (the case-1 fixture by up to ~0.16 cells), hence no tighter value.
    """
    diffs = np.diff(coords)
    diffs = diffs[diffs >= min_cell]
    if len(diffs) == 0:
        return None
    smallest = diffs.min()
    cell = float(np.median(diffs[diffs <= 1.5 * smallest]))

    ref = coords[len(coords) // 2]
    k = np.round((coords - ref) / cell)
    if np.ptp(k) > 0:
        cell, phase = np.polyfit(k, coords, 1)
    else:
        phase = ref
    resid = (coords - (phase + k * cell)) / cell  # offset from the lattice, in cells
    if np.mean(np.abs(resid) <= on_grid_tol) < min_on_grid:
        return None
    return float(cell), float(phase)


# --- SR conversion ---

def round_to_odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def meters_to_pixels(sr_m, resolution):
    return round_to_odd(sr_m / resolution)


def report_sr_adjustments(sr_micro_m, sr_macro_m, resolution):
    adjusted_micro = meters_to_pixels(sr_micro_m, resolution) * resolution
    adjusted_macro = meters_to_pixels(sr_macro_m, resolution) * resolution
    lines = []
    if adjusted_micro != sr_micro_m:
        lines.append(f"  sr_micro: {sr_micro_m}m -> {adjusted_micro}m ({meters_to_pixels(sr_micro_m, resolution)}px)")
    if adjusted_macro != sr_macro_m:
        lines.append(f"  sr_macro: {sr_macro_m}m -> {adjusted_macro}m ({meters_to_pixels(sr_macro_m, resolution)}px)")
    if lines:
        print(f"Based on your selected resolution ({resolution}m), smoothing radii have been adjusted:")
        for line in lines:
            print(line)


# --- Rasterization ---

def build_global_raster(field_geom, crs, resolution, origin=None):
    minx, miny, maxx, maxy = field_geom.bounds
    if origin is not None:
        ox, oy = origin
        minx = ox + np.floor((minx - ox) / resolution) * resolution
        miny = oy + np.floor((miny - oy) / resolution) * resolution
        maxx = ox + np.ceil((maxx - ox) / resolution) * resolution
        maxy = oy + np.ceil((maxy - oy) / resolution) * resolution
    else:
        minx = np.floor(minx / resolution) * resolution
        miny = np.floor(miny / resolution) * resolution
        maxx = np.ceil(maxx / resolution) * resolution
        maxy = np.ceil(maxy / resolution) * resolution
    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
    return transform, width, height, crs


def shape_to_raster(geom, transform, width, height, burn_value=1, dtype=np.uint8):
    out = np.zeros((height, width), dtype=dtype)
    rasterio.features.rasterize(
        [(geom, burn_value)],
        out=out,
        transform=transform,
        all_touched=False,
    )
    return out


def build_zones(field_geom, transform, width, height, edge_width, resolution):
    # Half-pixel inward shrink on the outer boundary avoids overspray: generate
    # must never paint outside the true field edge. This intentionally differs
    # from analyze's edge zone (_compute_edge_bias), which keeps the full field
    # to use every observed cell. The asymmetry is by design, not a bug.
    shrunk = field_geom.buffer(-0.5 * resolution)
    allowed = shape_to_raster(shrunk, transform, width, height) if not shrunk.is_empty else np.zeros((height, width), dtype=np.uint8)
    allowed_mask = allowed == 1

    core_geom = field_geom.buffer(-edge_width)
    if not core_geom.is_empty:
        core_arr = shape_to_raster(core_geom, transform, width, height)
        core_mask = (core_arr == 1) & allowed_mask
    else:
        core_mask = np.zeros((height, width), dtype=bool)

    edge_mask = allowed_mask & ~core_mask

    zones = np.zeros((height, width), dtype=np.uint8)
    zones[edge_mask] = 1
    zones[core_mask] = 2
    return zones


# --- Smoothing ---

def masked_smooth(noise, mask, radius):
    if radius <= 1:
        return noise.astype(float)
    w = mask.astype(float)
    num = uniform_filter(noise * w, size=radius, mode="reflect")
    den = uniform_filter(w, size=radius, mode="reflect")
    smooth = np.full_like(num, np.nan, dtype=float)
    np.divide(num, den, out=smooth, where=den > 0)
    return smooth


def allocate_from_noise(noise, mask, n_cells):
    sel = np.zeros_like(mask, dtype=np.uint8)
    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0 or n_cells <= 0:
        return sel
    n_cells = min(n_cells, idx.size)
    flat = noise.ravel()
    top_idx = idx[np.argpartition(flat[idx], -n_cells)[-n_cells:]]
    sel.ravel()[top_idx] = 1
    return sel


# --- Patch generation core ---

def make_patches(zones, ir, mmr, sr_micro, sr_macro, edge_bias,
                 smoothing_flag, rng_seed, resolution):
    k1, k2, k3, k4 = _K1, _K2, _K3, _K4
    H, W = zones.shape
    allowed_mask = zones > 0
    edge_mask = zones == 1
    core_mask = zones == 2

    edge_avail = int(edge_mask.sum())
    core_avail = int(core_mask.sum())
    n_allowed = edge_avail + core_avail

    if n_allowed == 0:
        return np.full_like(zones, -99, dtype=np.int16)

    if ir >= 1.0:
        out = np.full_like(zones, -99, dtype=np.int16)
        out[allowed_mask] = 1
        return out

    n_total = int(np.floor(ir * n_allowed))
    if n_total == 0:
        out = np.full_like(zones, -99, dtype=np.int16)
        out[allowed_mask] = 0
        return out

    # edge/core split with edge bias
    weight_edge = edge_bias * edge_avail
    weight_core = core_avail
    p_edge = weight_edge / (weight_edge + weight_core) if (weight_edge + weight_core) > 0 else 0
    n_edge_desired = int(round(p_edge * n_total))
    n_core_desired = n_total - n_edge_desired

    if n_edge_desired > edge_avail:
        n_edge = edge_avail
        n_core = min(core_avail, n_core_desired + (n_edge_desired - edge_avail))
        flag = "edge_full"
    elif n_core_desired > core_avail:
        n_core = core_avail
        n_edge = min(edge_avail, n_edge_desired + (n_core_desired - core_avail))
        flag = "core_full"
    else:
        n_edge, n_core = n_edge_desired, n_core_desired
        flag = "balanced"

    sr_micro_px = meters_to_pixels(sr_micro, resolution)
    sr_macro_px = meters_to_pixels(sr_macro, resolution)

    # Independent random streams per component: one seed gives the same random
    # fields for every parameter set, since the number of draws in one component
    # (e.g. num_seeds) never shifts the draws of another.
    rng_macro, rng_bg, rng_micro = (
        np.random.default_rng(s) for s in np.random.SeedSequence(rng_seed).spawn(3)
    )

    if smoothing_flag == 1:
        macro_field = masked_smooth(rng_macro.normal(0, 1, size=(H, W)), allowed_mask, sr_macro_px)
    else:
        seeds = np.zeros((H, W), dtype=float)
        num_seeds = max(1, int(int(n_total * (1 - mmr) / (sr_macro_px ** 2)) * k1))
        # Cannot seed more cells than exist; at the cap every cell is a seed, the
        # macro field is flat and allocation is driven by background noise alone.
        num_seeds = min(num_seeds, H * W)
        # Rank all cells once by a random priority and take the top num_seeds, so
        # seed positions are nested across seed counts (more seeds = same + extra).
        priority = rng_macro.random(H * W)
        seed_indices = np.argpartition(priority, -num_seeds)[-num_seeds:]
        seeds.ravel()[seed_indices] = 1.0
        macro_field = gaussian_filter(seeds, sigma=sr_macro_px / k2, mode="reflect")
        macro_field += rng_bg.uniform(0, k3, size=(H, W))
        macro_field = gaussian_filter(macro_field, sigma=sr_macro_px / k4, mode="reflect")

    micro_field = masked_smooth(rng_micro.normal(0, 1, size=(H, W)), allowed_mask, sr_micro_px)

    if flag == "edge_full":
        sel_edge = edge_mask.copy()
        n_core_macro = n_core - int(round(mmr * n_core))
        sel_macro = allocate_from_noise(macro_field, core_mask, n_core_macro)
        micro_field[sel_macro.astype(bool)] = -np.inf
        sel_micro = allocate_from_noise(micro_field, core_mask, int(round(mmr * n_core)))
        sel_core = (sel_macro | sel_micro).astype(bool)

    elif flag == "core_full":
        sel_core = core_mask.copy()
        n_edge_macro = n_edge - int(round(mmr * n_edge))
        sel_macro = allocate_from_noise(macro_field, edge_mask, n_edge_macro)
        micro_field[sel_macro.astype(bool)] = -np.inf
        sel_micro = allocate_from_noise(micro_field, edge_mask, int(round(mmr * n_edge)))
        sel_edge = (sel_macro | sel_micro).astype(bool)

    else:  # balanced
        n_edge_macro = n_edge - int(round(mmr * n_edge))
        sel_macro = allocate_from_noise(macro_field, edge_mask, n_edge_macro)
        micro_field[sel_macro.astype(bool)] = -np.inf
        sel_micro = allocate_from_noise(micro_field, edge_mask, int(round(mmr * n_edge)))
        sel_edge = (sel_macro | sel_micro).astype(bool)

        n_core_macro = n_core - int(round(mmr * n_core))
        sel_macro = allocate_from_noise(macro_field, core_mask, n_core_macro)
        micro_field[sel_macro.astype(bool)] = -np.inf
        sel_micro = allocate_from_noise(micro_field, core_mask, int(round(mmr * n_core)))
        sel_core = (sel_macro | sel_micro).astype(bool)

    out = np.full_like(zones, -99, dtype=np.int16)
    out[sel_edge | sel_core] = 1
    out[(~(sel_edge | sel_core)) & allowed_mask] = 0
    return out
