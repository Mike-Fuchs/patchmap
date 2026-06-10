import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon
from pathlib import Path
from patchmap.utils import round_to_odd, meters_to_pixels, report_sr_adjustments, detect_grid

FIXTURES = Path(__file__).parent / "fixtures"


class TestRoundToOdd:
    def test_odd_integer_unchanged(self):
        assert round_to_odd(3) == 3

    def test_even_integer_incremented(self):
        assert round_to_odd(4) == 5

    def test_rounds_up_to_odd(self):
        assert round_to_odd(2.5) == 3

    def test_rounds_down_to_odd(self):
        assert round_to_odd(2.1) == 3

    def test_one_is_odd(self):
        assert round_to_odd(1) == 1

    def test_zero_returns_one(self):
        assert round_to_odd(0) == 1


class TestMetersToPixels:
    def test_exact_odd_result(self):
        # 2.5m / 1.0m = 2.5 -> 3px
        assert meters_to_pixels(2.5, 1.0) == 3

    def test_coarse_resolution(self):
        # 2.5m / 3.0m = 0.83 -> 1px
        assert meters_to_pixels(2.5, 3.0) == 1

    def test_fine_resolution(self):
        # 2.5m / 0.25m = 10 -> 11px
        assert meters_to_pixels(2.5, 0.25) == 11

    def test_very_fine_resolution(self):
        # 2.5m / 0.1m = 25px (already odd)
        assert meters_to_pixels(2.5, 0.1) == 25

    def test_returns_odd(self):
        for sr in [1.0, 2.0, 3.0, 5.0, 10.0]:
            for res in [0.25, 0.5, 1.0, 2.0]:
                result = meters_to_pixels(sr, res)
                assert result % 2 == 1, f"meters_to_pixels({sr}, {res}) = {result} is not odd"


class TestReportSrAdjustments:
    def test_no_output_when_no_adjustment(self, capsys):
        # 3.0m / 1.0m = 3px -> 3px * 1.0m = 3.0m, no change
        report_sr_adjustments(3.0, 9.0, 1.0)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_prints_when_adjusted(self, capsys):
        # 2.5m / 1.0m = 3px -> 3.0m, changed from 2.5
        report_sr_adjustments(2.5, 10.0, 1.0)
        captured = capsys.readouterr()
        assert "sr_micro" in captured.out
        assert "2.5m" in captured.out
        assert "3.0m" in captured.out

    def test_prints_resolution_in_message(self, capsys):
        report_sr_adjustments(2.5, 10.0, 1.0)
        captured = capsys.readouterr()
        assert "1.0m" in captured.out


def _make_grid_gdf(origin_x, origin_y, cell_size, n=20, crs="EPSG:32632"):
    """Helper: create a regular grid of square polygons."""
    cells = []
    for i in range(n):
        for j in range(n):
            x0 = origin_x + i * cell_size
            y0 = origin_y + j * cell_size
            cells.append(box(x0, y0, x0 + cell_size, y0 + cell_size))
    return gpd.GeoDataFrame(geometry=cells, crs=crs)


class TestDetectGrid:
    def test_case1_fixture(self):
        weed = gpd.read_file(FIXTURES / "weed_observation.geojson")
        ox, oy, res, case, _ = detect_grid(weed)
        assert case == 1
        assert abs(res - 2.3e-6) < 1e-7
        assert abs(ox - 7.05182) < 0.0001
        assert abs(oy - 48.70641) < 0.0001

    def test_case2_utm_grid(self):
        gdf = _make_grid_gdf(400000.0, 5400000.0, cell_size=0.5)
        ox, oy, res, case, _ = detect_grid(gdf)
        assert case == 2
        assert abs(res - 0.5) < 0.01
        assert abs(ox - 400000.0) < 1.0
        assert abs(oy - 5400000.0) < 1.0

    def test_case3_irregular(self):
        # explicitly non-square cells: 0.3m wide x 0.9m tall — squareness fails in both CRS
        polys = [box(400000.0 + i * 0.5, 5400000.0, 400000.0 + i * 0.5 + 0.3, 5400000.9)
                 for i in range(20)]
        gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:32632")
        ox, oy, res, case, _ = detect_grid(gdf)
        assert case == 3
        assert res == 1.0
        assert ox is not None
        assert oy is not None

    def test_case3_resolution_is_1m(self):
        polys = [box(400000.0 + i * 0.5, 5400000.0, 400000.0 + i * 0.5 + 0.3, 5400000.9)
                 for i in range(20)]
        gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:32632")
        _, _, res, case, _ = detect_grid(gdf)
        assert case == 3
        assert res == 1.0
