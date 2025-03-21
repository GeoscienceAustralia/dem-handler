from dem_handler.dem.rema import get_rema_dem_for_bounds, BBox
from dem_handler.utils.spatial import resize_bounds, BoundingBox, transform_polygon
from dataclasses import dataclass
import rasterio as rio
from numpy.testing import assert_allclose
import pytest
import shutil
import os
from pathlib import Path
from shapely import box


CURRENT_DIR = Path(__file__).parent.resolve()
TEST_PATH = CURRENT_DIR.parent
GEOID_PATH = TEST_PATH / "data/geoid/egm_08_geoid.tif"
REMA_INDEX_PATH = CURRENT_DIR / "data/REMA_Mosaic_Index_v2_gpkg.gpkg"
TMP_PATH = CURRENT_DIR / "TMP"
TEST_DATA_PATH = CURRENT_DIR / "data"


@dataclass
class TestDem:
    requested_bounds: BBox
    bounds_array_file: str
    resolution: int
    num_tasks: int | None


dem_name = "38_48_1_2_32m_v2.0_dem.tif"
bbox = BoundingBox(67.45, -72.55, 67.55, -72.45)
test_single_tile = TestDem(
    bbox,
    os.path.join(TEST_DATA_PATH, dem_name),
    32,
    None,
)

dem_name = "rema_32m_three_tiles.tif"
test_three_tiles = TestDem(
    resize_bounds(bbox, 10.0),
    os.path.join(TEST_DATA_PATH, dem_name),
    32,
    -1,
)


dem_name = "rema_32m_two_tiles_ocean.tif"
ocean_bbox = BoundingBox(162.0, -70.95, 163.0, -69.83)
test_four_tiles_ocean = TestDem(
    ocean_bbox,
    os.path.join(TEST_DATA_PATH, dem_name),
    32,
    None,
)

test_dems = [
    test_single_tile,
    test_three_tiles,
    test_four_tiles_ocean,
]


@pytest.mark.parametrize("test_input", test_dems)
def test_rema_dem_for_bounds_ocean_and_land(test_input: TestDem):

    bounds = test_input.requested_bounds
    bounds_array_file = test_input.bounds_array_file
    resolution = test_input.resolution
    num_tasks = test_input.num_tasks

    if not TMP_PATH.exists():
        TMP_PATH.mkdir(parents=True, exist_ok=True)

    array, _, _ = get_rema_dem_for_bounds(
        bounds,
        save_path=TMP_PATH / Path("TMP.tif"),
        rema_index_path=REMA_INDEX_PATH,
        resolution=resolution,
        bounds_src_crs=4326,
        ellipsoid_heights=False,
        num_tasks=num_tasks,
    )

    with rio.open(bounds_array_file, "r") as src:
        expected_array = src.read(1)

    assert_allclose(array, expected_array)

    with rio.open(str(TMP_PATH / Path("TMP.tif"))) as src:
        array = src.read(1)

    assert_allclose(array, expected_array)

    # Once complete, remove the TMP files and directory
    shutil.rmtree(TMP_PATH)


def test_rema_dem_for_psg_bounds():
    psg_bbox = BoundingBox(*transform_polygon(box(*bbox.bounds), 4326, 3031).bounds)
    dem_name = "rema_32m_four_tiles.tif"
    bounds_array_file = os.path.join(TEST_DATA_PATH, dem_name)

    if not TMP_PATH.exists():
        TMP_PATH.mkdir(parents=True, exist_ok=True)

    array, _, _ = get_rema_dem_for_bounds(
        resize_bounds(psg_bbox, 10.0),
        save_path=TMP_PATH / Path("TMP.tif"),
        rema_index_path=REMA_INDEX_PATH,
        resolution=32,
        ellipsoid_heights=False,
    )

    with rio.open(bounds_array_file, "r") as src:
        expected_array = src.read(1)

    assert_allclose(array, expected_array)

    with rio.open(str(TMP_PATH / Path("TMP.tif"))) as src:
        array = src.read(1)

    assert_allclose(array, expected_array)

    # Once complete, remove the TMP files and directory
    shutil.rmtree(TMP_PATH)


def test_rema_dem_for_bounds_ocean_and_land_ellipsoid():

    dem_name = "38_48_1_2_32m_v2.0_dem_ellipsoid.tif"
    bbox = BoundingBox(67.45, -72.55, 67.55, -72.45)
    bounds_array_file = os.path.join(TEST_DATA_PATH, dem_name)

    if not TMP_PATH.exists():
        TMP_PATH.mkdir(parents=True, exist_ok=True)

    array, _, _ = get_rema_dem_for_bounds(
        bbox,
        save_path=TMP_PATH / Path("TMP.tif"),
        rema_index_path=REMA_INDEX_PATH,
        resolution=32,
        bounds_src_crs=4326,
        geoid_tif_path=GEOID_PATH,
    )

    with rio.open(bounds_array_file, "r") as src:
        expected_array = src.read(1)

    assert_allclose(array, expected_array)

    with rio.open(str(TMP_PATH / Path("TMP.tif"))) as src:
        array = src.read(1)

    assert_allclose(array, expected_array)

    # Once complete, remove the TMP files and directory
    shutil.rmtree(TMP_PATH)
