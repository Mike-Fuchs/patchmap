import pytest
import numpy as np
import geopandas as gpd
from pathlib import Path
from patchmap import generate

FIXTURES = Path(__file__).parent / "fixtures"

# reproject fixture to UTM once for all tests
@pytest.fixture(scope="module")
def field_geom():
    gdf = gpd.read_file(FIXTURES / "field_boundary.geojson").to_crs("EPSG:32632")
    return gdf.geometry.iloc[0]

@pytest.fixture(scope="module")
def crs():
    return "EPSG:32632"

# shared default parameters
DEFAULTS = dict(
    mmr=0.5,
    sr_micro=3.0,
    sr_macro=10.0,
    edge_bias=1.5,
    edge_width=10.0,
    smoothing_flag=2,
    resolution=1.0,
    rng_seed=42,
)


class TestGenerateOutput:
    def test_returns_memfile(self, field_geom, crs):
        import rasterio.io
        result = generate(field_geom, ir=0.15, crs=crs, **DEFAULTS)
        assert isinstance(result, rasterio.io.MemoryFile)

    def test_output_only_valid_values(self, field_geom, crs):
        result = generate(field_geom, ir=0.15, crs=crs, **DEFAULTS)
        with result.open() as ds:
            arr = ds.read(1)
        assert set(np.unique(arr)).issubset({-99, 0, 1})

    def test_has_spatial_metadata(self, field_geom, crs):
        result = generate(field_geom, ir=0.15, crs=crs, **DEFAULTS)
        with result.open() as ds:
            assert ds.crs is not None
            assert ds.transform is not None
            assert ds.nodata == -99


class TestInfectionRate:
    @pytest.mark.parametrize("ir", [0.05, 0.15, 0.50])
    def test_ir_within_tolerance(self, field_geom, crs, ir):
        result = generate(field_geom, ir=ir, crs=crs, **DEFAULTS)
        with result.open() as ds:
            arr = ds.read(1)
        n_field = int((arr >= 0).sum())
        n_infected = int((arr == 1).sum())
        actual_ir = n_infected / n_field
        assert abs(actual_ir - ir) < 0.02, f"Expected IR ~{ir}, got {actual_ir:.4f}"

    def test_zero_ir_no_infection(self, field_geom, crs):
        result = generate(field_geom, ir=0.0, crs=crs, **DEFAULTS)
        with result.open() as ds:
            arr = ds.read(1)
        assert (arr == 1).sum() == 0

    def test_full_ir_all_infected(self, field_geom, crs):
        result = generate(field_geom, ir=1.0, crs=crs, **DEFAULTS)
        with result.open() as ds:
            arr = ds.read(1)
        assert (arr == 0).sum() == 0


class TestReproducibility:
    def test_same_seed_same_result(self, field_geom, crs):
        r1 = generate(field_geom, ir=0.15, crs=crs, **DEFAULTS)
        r2 = generate(field_geom, ir=0.15, crs=crs, **DEFAULTS)
        with r1.open() as ds1, r2.open() as ds2:
            assert np.array_equal(ds1.read(1), ds2.read(1))

    def test_different_seed_different_result(self, field_geom, crs):
        params = {**DEFAULTS, "rng_seed": 99}
        r1 = generate(field_geom, ir=0.15, crs=crs, **DEFAULTS)
        r2 = generate(field_geom, ir=0.15, crs=crs, **params)
        with r1.open() as ds1, r2.open() as ds2:
            assert not np.array_equal(ds1.read(1), ds2.read(1))


class TestSmoothingModes:
    def test_focal_smoothing(self, field_geom, crs):
        params = {**DEFAULTS, "smoothing_flag": 1}
        result = generate(field_geom, ir=0.15, crs=crs, **params)
        with result.open() as ds:
            arr = ds.read(1)
        assert set(np.unique(arr)).issubset({-99, 0, 1})

    def test_gaussian_smoothing(self, field_geom, crs):
        params = {**DEFAULTS, "smoothing_flag": 2}
        result = generate(field_geom, ir=0.15, crs=crs, **params)
        with result.open() as ds:
            arr = ds.read(1)
        assert set(np.unique(arr)).issubset({-99, 0, 1})


class TestMode:
    def test_unsupported_mode_raises(self, field_geom, crs):
        with pytest.raises(NotImplementedError):
            generate(field_geom, ir=0.15, crs=crs, mode="strip", **DEFAULTS)
