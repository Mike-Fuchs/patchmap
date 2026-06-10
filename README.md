# patchmap

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-GPL--3.0--or--later-green) ![Status](https://img.shields.io/badge/status-research-orange)

Weeds and infections rarely spread evenly across a field — they clump, crowd toward the edges, and form patches at very different scales. `patchmap` gives precision crop-protection research two tools for working with that spatial structure:

- **`analyze`** — measure how weeds are distributed in real observation data, reducing a whole field to a handful of comparable spatial metrics.
- **`generate`** — go the other way: turn those metrics into realistic synthetic patch rasters to feed into models and simulations.

The two are built to round-trip: `analyze` a field, hand the numbers straight to `generate`, and you get a synthetic field with the same spatial character.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Functions](#functions)
  - [`analyze`](#analyze)
  - [`generate`](#generate)
- [Concepts](#concepts)
- [Dependencies](#dependencies)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Installation

```bash
pip install git+https://github.com/Mike-Fuchs/patchmap.git
```

Requires Python ≥ 3.10.

---

## Quick start

**Analyze observed data** — characterize weed distribution:

```python
import geopandas as gpd
from patchmap import analyze

field = gpd.read_file("field_boundary.geojson")
weeds = gpd.read_file("weed_observations.geojson")

metrics = analyze(field, weeds)
# -> {'ir': 0.07, 'mmr': 0.54, 'sr_micro': 1.63, 'sr_macro': 6.35, 'edge_bias': 1.99}
```

**Generate a patch raster** — synthesize weed distribution:

```python
import rasterio
from patchmap import generate

# field_geom: a non-empty shapely polygon in a projected CRS (meters)
result = generate(field_geom, ir=0.07, mmr=0.54, sr_micro=1.63, sr_macro=6.35,
                  edge_bias=1.99, crs="EPSG:32632")

with result.open() as ds:
    patch_array = ds.read(1)   # 1 = infected, 0 = healthy, -99 = outside field

# ...or save it to disk
with result.open() as ds, rasterio.open("output.tif", "w", **ds.profile) as dst:
    dst.write(ds.read())
```

---

## Functions

### `analyze`

```python
analyze(field_gdf, weed_gdf, edge_width=10.0) -> dict
```

Extract spatial metrics from observed weed/infection data. Auto-detects the native
grid of the source data (see [Concepts](#concepts)), so there is no `resolution`
parameter; all lengths are returned in meters.

**Parameters**

| Name | Type | Unit | Default | Description |
|------|------|------|---------|-------------|
| `field_gdf` | GeoDataFrame | — | *required* | Field boundary polygon(s). |
| `weed_gdf` | GeoDataFrame | — | *required* | Observed weed/infection polygons. |
| `edge_width` | float | m | `10.0` | Edge-zone width used for `edge_bias`. Match the value used in `generate`. |

Both inputs must be GeoDataFrames with a defined CRS, else `TypeError` / `ValueError`.
They may be in different CRS — reconciled internally.

**Returns** — `dict`; lengths in meters, ratios dimensionless:

| Key | Unit | Description |
|-----|------|-------------|
| `ir` | — | Infection rate (0–1) — fraction of field area covered by infection. |
| `mmr` | — | Micro-to-macro ratio (0–1) — fraction of infected area in micro patches. |
| `sr_micro` | m | Mean equivalent diameter of micro patches, `2·√(area/π)` (> 0). |
| `sr_macro` | m | Mean equivalent diameter of macro patches (> 0). |
| `edge_bias` | — | Infection rate in the edge zone ÷ core zone (> 1 = more near edges). |

---

### `generate`

```python
generate(field_geom, ir, mmr, sr_micro, sr_macro,
         edge_bias=1.5, edge_width=10.0, smoothing_flag=2,
         rng_seed=None, resolution=1.0, crs=None, mode="patch") -> rasterio.io.MemoryFile
```

Synthesize a binary infection patch raster for a field.

**Parameters**

| Name | Type | Unit | Default | Description |
|------|------|------|---------|-------------|
| `field_geom` | shapely geometry | m | *required* | Field boundary. Non-empty `Polygon`/`MultiPolygon` in a projected CRS. |
| `ir` | float | — | *required* | Target infection rate (0–1) — fraction of field cells to infect. |
| `mmr` | float | — | *required* | Micro-to-macro ratio (0–1) — share of infected cells in micro patches. |
| `sr_micro` | float | m | *required* | Smoothing window for micro patches — fine-scale structure. |
| `sr_macro` | float | m | *required* | Smoothing window for macro patches — coarse-scale extent. |
| `edge_bias` | float | — | `1.5` | Edge-vs-core allocation multiplier (`1.0` = none, `> 1` = more near edges). |
| `edge_width` | float | m | `10.0` | Edge-zone width. |
| `smoothing_flag` | int | — | `2` | `1` = focal (uniform filter); `2` = Gaussian seed (more realistic blobs). |
| `rng_seed` | int / None | — | `None` | Random seed for reproducibility. `None` = non-deterministic. |
| `resolution` | float | m | `1.0` | Output raster cell size. |
| `crs` | any | — | `None` | Output CRS, e.g. `"EPSG:32632"`. Must be projected. |
| `mode` | str | — | `"patch"` | Application mode. Only `"patch"` is supported; `"strip"` is planned. |

`field_geom` must be a non-empty `Polygon`/`MultiPolygon` in a projected CRS (meters):
`resolution`, `edge_width` and `sr_*` are metric, so a geographic `crs` raises
`ValueError`, and an unsupported `mode` raises `NotImplementedError`.

**Returns** — `rasterio.io.MemoryFile`, a single-band `Int16` raster:

| Value | Meaning |
|-------|---------|
| `1` | Infected |
| `0` | Healthy |
| `-99` | Outside field (nodata) |

```python
with result.open() as ds:
    array     = ds.read(1)   # numpy Int16 array
    transform = ds.transform # affine transform
    crs       = ds.crs       # coordinate reference system
```

---

## Concepts

**Grid detection.** `analyze` has no fixed resolution — it detects the native grid of the observation data and rasterizes on it. Square grids (in WGS84 or UTM) give an exact binary mask at native resolution; irregular data falls back to a 1 m UTM grid and uses `exactextract` for proportional per-cell coverage. IR and `edge_bias` use the binary mask for grid data and proportional coverage for irregular data; `sr_micro`, `sr_macro` and `mmr` always use the binary mask.

**Smoothing radii.** `sr_micro` and `sr_macro` are always in meters, independent of `resolution`. `generate` converts them to an odd pixel count (`round_to_odd(sr / resolution)`, required by the kernel) and prints a note if rounding changes the effective value. At very coarse resolution a radius can collapse to 1 px, disabling smoothing — the note flags this.

**Internal constants.** The micro/macro split percentile and the Gaussian-seed tuning (`k1`–`k4`) are calibrated package-wide constants.

---

## Dependencies

[numpy](https://numpy.org/) ·
[geopandas](https://geopandas.org/) ·
[shapely](https://shapely.readthedocs.io/) ·
[rasterio](https://rasterio.readthedocs.io/) ·
[scipy](https://scipy.org/) ·
[scikit-image](https://scikit-image.org/) ·
[exactextract](https://github.com/isciences/exactextract)

---

## Disclaimer

`patchmap` is provided for **research purposes only** and distributed **"as is", without
warranty of any kind**, express or implied — including, without limitation, warranties of
merchantability, fitness for a particular purpose, and non-infringement. Results must be
independently validated before any operational use.

To the fullest extent permitted by applicable law, the author shall not be liable for any
direct, indirect, incidental, or consequential damage or loss arising from the use of, or
inability to use, this software.

This exclusion does **not** apply, and the author's liability remains unaffected, in respect
of: damage arising from injury to life, body, or health; damage caused by intent or gross
negligence; liability under the German Product Liability Act (*Produkthaftungsgesetz*); the
absence of a quality for which an express guarantee was given; and any other liability that
cannot be excluded or limited under mandatory statutory law.

See [LICENSE](LICENSE) for the full license terms.

---

## License

patchmap — Copyright (C) 2026 Mike Devin Fuchs

Licensed under the **GNU General Public License, version 3 or (at your option) any later version**. See [LICENSE](LICENSE) for full terms.
