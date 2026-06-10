import pytest
import numpy as np
import geopandas as gpd
import shapely.geometry
import rasterio.features
from pathlib import Path
from patchmap import analyze, generate
from patchmap.utils import detect_grid

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def field_gdf():
    return gpd.read_file(FIXTURES / "field_boundary.geojson")


@pytest.fixture(scope="module")
def weed_gdf():
    return gpd.read_file(FIXTURES / "weed_observation.geojson")


@pytest.fixture(scope="module")
def metrics(field_gdf, weed_gdf):
    return analyze(field_gdf, weed_gdf)


class TestAnalyzeOutput:
    def test_returns_dict(self, metrics):
        assert isinstance(metrics, dict)

    def test_all_keys_present(self, metrics):
        assert set(metrics.keys()) == {"ir", "mmr", "sr_micro", "sr_macro", "edge_bias"}

    def test_all_values_finite(self, metrics):
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"


class TestAnalyzeValues:
    """Validation against known-good outputs from the fixture dataset."""

    def test_ir(self, metrics):
        assert abs(metrics["ir"] - 0.0704) < 0.001

    def test_mmr(self, metrics):
        assert abs(metrics["mmr"] - 0.5426) < 0.01

    def test_sr_micro(self, metrics):
        assert abs(metrics["sr_micro"] - 1.6254) < 0.05

    def test_sr_macro(self, metrics):
        assert abs(metrics["sr_macro"] - 6.3686) < 0.1

    def test_edge_bias(self, metrics):
        assert abs(metrics["edge_bias"] - 1.9862) < 0.05


class TestAnalyzeRanges:
    def test_ir_between_0_and_1(self, metrics):
        assert 0 <= metrics["ir"] <= 1

    def test_mmr_between_0_and_1(self, metrics):
        assert 0 <= metrics["mmr"] <= 1

    def test_sr_micro_positive(self, metrics):
        assert metrics["sr_micro"] > 0

    def test_sr_macro_positive(self, metrics):
        assert metrics["sr_macro"] > 0

    def test_sr_macro_larger_than_micro(self, metrics):
        assert metrics["sr_macro"] > metrics["sr_micro"]

    def test_edge_bias_positive(self, metrics):
        assert metrics["edge_bias"] > 0


class TestAnalyzeCrsHandling:
    def test_accepts_geographic_crs(self, field_gdf, weed_gdf):
        # fixture is EPSG:4326 — should auto-reproject without error
        result = analyze(field_gdf, weed_gdf)
        assert np.isfinite(result["ir"])

    def test_accepts_projected_crs(self, field_gdf, weed_gdf):
        field_proj = field_gdf.to_crs("EPSG:32632")
        weed_proj = weed_gdf.to_crs("EPSG:32632")
        result = analyze(field_proj, weed_proj)
        assert np.isfinite(result["ir"])

    def test_mismatched_crs_handled(self, field_gdf, weed_gdf):
        # field in UTM, weed still in geographic — should reproject weed automatically
        field_proj = field_gdf.to_crs("EPSG:32632")
        result = analyze(field_proj, weed_gdf)
        assert np.isfinite(result["ir"])


def _vectorize_infected(memfile, crs):
    """Convert infected pixels (value=1) in a generate() MemoryFile to individual cell polygons."""
    with memfile.open() as ds:
        arr = ds.read(1)
        transform = ds.transform
    rows, cols = np.where(arr == 1)
    polys = [
        shapely.geometry.box(
            transform.c + c * transform.a,
            transform.f + r * transform.e,
            transform.c + (c + 1) * transform.a,
            transform.f + (r + 1) * transform.e,
        )
        for r, c in zip(rows, cols)
    ]
    return gpd.GeoDataFrame(geometry=polys, crs=crs)


class TestRoundtrip:
    """Analyze → generate → vectorize → analyze. Validates case 2 and pipeline consistency."""

    @pytest.fixture(scope="class")
    def roundtrip(self, metrics, field_gdf):
        utm_crs = field_gdf.estimate_utm_crs()
        field_geom_utm = field_gdf.to_crs(utm_crs).geometry.union_all()

        memfile = generate(
            field_geom_utm,
            ir=metrics["ir"], mmr=metrics["mmr"],
            sr_micro=metrics["sr_micro"], sr_macro=metrics["sr_macro"],
            edge_bias=metrics["edge_bias"],
            resolution=0.25, crs=utm_crs, rng_seed=42,
        )

        synthetic_weed = _vectorize_infected(memfile, utm_crs)
        metrics2 = analyze(field_gdf.to_crs(utm_crs), synthetic_weed)
        return metrics, metrics2

    def test_case2_detected(self, field_gdf, metrics):
        utm_crs = field_gdf.estimate_utm_crs()
        field_geom_utm = field_gdf.to_crs(utm_crs).geometry.union_all()
        memfile = generate(
            field_geom_utm,
            ir=metrics["ir"], mmr=metrics["mmr"],
            sr_micro=metrics["sr_micro"], sr_macro=metrics["sr_macro"],
            edge_bias=metrics["edge_bias"],
            resolution=0.25, crs=utm_crs, rng_seed=42,
        )
        synthetic_weed = _vectorize_infected(memfile, utm_crs)
        _, _, _, case, _ = detect_grid(synthetic_weed)
        assert case == 2

    def test_all_metrics_finite(self, roundtrip):
        _, metrics2 = roundtrip
        for k, v in metrics2.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_ir_preserved(self, roundtrip):
        metrics1, metrics2 = roundtrip
        assert abs(metrics2["ir"] - metrics1["ir"]) < 0.02

    def test_mmr_in_range(self, roundtrip):
        _, metrics2 = roundtrip
        assert 0 <= metrics2["mmr"] <= 1

    def test_sr_macro_larger_than_micro(self, roundtrip):
        _, metrics2 = roundtrip
        assert metrics2["sr_macro"] > metrics2["sr_micro"]


class TestCase3:
    """Force the fixture through the case-3 (exactextract proportional) path.

    The fixture is natively case 1; here detect_grid is monkeypatched to return
    case 3 so the exactextract rasterization + metric path is exercised end to
    end. Case-3 classification itself is validated separately in test_utils.
    Values legitimately differ from the case-1 known-goods (1 m proportional vs
    native binary), so only IR — which proportional coverage approximates well —
    is anchored to the case-1 result; the rest get finite/range checks.
    """

    @pytest.fixture(scope="class")
    def case3_metrics(self, field_gdf, weed_gdf, request):
        mp = pytest.MonkeyPatch()

        def force_case3(gdf, *args, **kwargs):
            target = gdf.estimate_utm_crs()
            b = gdf.to_crs(target).total_bounds
            return float(b[0]), float(b[1]), 1.0, 3, target

        import importlib
        analyze_mod = importlib.import_module("patchmap.analyze")
        mp.setattr(analyze_mod, "detect_grid", force_case3)
        request.addfinalizer(mp.undo)
        return analyze(field_gdf, weed_gdf)

    def test_all_keys_present(self, case3_metrics):
        assert set(case3_metrics.keys()) == {"ir", "mmr", "sr_micro", "sr_macro", "edge_bias"}

    def test_all_values_finite(self, case3_metrics):
        for k, v in case3_metrics.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_ranges(self, case3_metrics):
        m = case3_metrics
        assert 0 <= m["ir"] <= 1
        assert 0 <= m["mmr"] <= 1
        assert m["sr_micro"] > 0
        assert m["sr_macro"] > 0
        assert m["sr_macro"] > m["sr_micro"]
        assert m["edge_bias"] > 0

    def test_ir_close_to_case1(self, case3_metrics, metrics):
        # Same field; proportional coverage at 1 m should approximate the
        # native-binary IR closely.
        assert abs(case3_metrics["ir"] - metrics["ir"]) < 0.02
