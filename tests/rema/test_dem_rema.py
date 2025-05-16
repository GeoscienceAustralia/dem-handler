from dem_handler.dem.rema import get_rema_dem_for_bounds, BBox
from dem_handler.utils.spatial import resize_bounds, BoundingBox, transform_polygon
from dataclasses import dataclass, replace
import rasterio as rio
from numpy.testing import assert_allclose
import pytest
import shutil
import os
from pathlib import Path
from shapely import box


CURRENT_DIR = Path(__file__).parent.resolve()
TEST_DATA_PATH = CURRENT_DIR.parent
REMA_INDEX_PATH = CURRENT_DIR / "data/REMA_Mosaic_Index_v2_gpkg.gpkg"
GEOID_DATA_PATH = TEST_DATA_PATH / "data" / "geoid"
TMP_PATH = CURRENT_DIR / "TMP"
TEST_DATA_PATH = CURRENT_DIR / "data"


# data for tests is downloaded with download_test_data.py script
@dataclass
class TestDem:
    requested_bounds: BBox
    dem_file: str
    resolution: int
    num_tasks: int | None
    geoid: Path
    ellipsoid_heights: bool


# A single source tile
bbox = BoundingBox(67.45, -72.55, 67.55, -72.45)
test_single_tile_ellipsoid_h = TestDem(
    bbox,
    os.path.join(TEST_DATA_PATH, "rema_38_48_1_2_32m_v2.0_dem_ellipsoid_h.tif"),
    32,
    None,
    os.path.join(GEOID_DATA_PATH, "egm_08_geoid_rema_38_48_1_2_32m_v2.0_dem.tif"),
    True,
)

test_four_tiles_ellipsoid_h = TestDem(
    resize_bounds(bbox, 10.0),
    os.path.join(TEST_DATA_PATH, "rema_32m_four_tiles_ellipsoid_h.tif"),
    32,
    -1,
    os.path.join(GEOID_DATA_PATH, "egm_08_geoid_rema_32m_four_tiles.tif"),
    True,
)

# over ocean where tile data exists
ocean_bbox = BoundingBox(162.0, -70.95, 163.0, -69.83)
test_two_tiles_ocean_ellipsoid_h = TestDem(
    ocean_bbox,
    os.path.join(TEST_DATA_PATH, "rema_32m_two_tiles_ocean_ellipsoid_h.tif"),
    32,
    None,
    os.path.join(GEOID_DATA_PATH, "egm_08_geoid_rema_32m_two_tiles_ocean.tif"),
    True,
)

# over land and ocean where tile data partially exists
ocean_no_data_bbox = BoundingBox(165.75, -77.07, 167.40, -76.53)
test_one_tile_and_no_tile_overlap_ellipsoid_h = TestDem(
    ocean_no_data_bbox,
    os.path.join(
        TEST_DATA_PATH, "rema_32m_one_tile_and_no_tile_overlap_ellipsoid_h.tif"
    ),
    32,
    None,
    os.path.join(
        GEOID_DATA_PATH, "egm_08_geoid_rema_32m_one_tile_and_no_tile_overlap.tif"
    ),
    True,
)

test_dems_ellipsoid = [
    test_single_tile_ellipsoid_h,
    test_four_tiles_ellipsoid_h,
    test_two_tiles_ocean_ellipsoid_h,
    test_one_tile_and_no_tile_overlap_ellipsoid_h,
]

# make the geoid test set
# reference the geoid files instead of the ellipsoid ones
test_single_tile_geoid_h = replace(
    test_single_tile_ellipsoid_h,
    dem_file=test_single_tile_ellipsoid_h.dem_file.replace("ellipsoid", "geoid"),
    ellipsoid_heights=False,
)
test_four_tiles_geoid_h = replace(
    test_four_tiles_ellipsoid_h,
    dem_file=test_four_tiles_ellipsoid_h.dem_file.replace("ellipsoid", "geoid"),
    ellipsoid_heights=False,
)
test_two_tiles_ocean_geoid_h = replace(
    test_two_tiles_ocean_ellipsoid_h,
    dem_file=test_two_tiles_ocean_ellipsoid_h.dem_file.replace("ellipsoid", "geoid"),
    ellipsoid_heights=False,
)
test_one_tile_and_no_tile_overlap_geoid_h = replace(
    test_one_tile_and_no_tile_overlap_ellipsoid_h,
    dem_file=test_one_tile_and_no_tile_overlap_ellipsoid_h.dem_file.replace(
        "ellipsoid", "geoid"
    ),
    ellipsoid_heights=False,
)

test_dems_geoid = [
    test_single_tile_geoid_h,
    test_four_tiles_geoid_h,
    test_two_tiles_ocean_geoid_h,
    test_one_tile_and_no_tile_overlap_geoid_h,
]


@pytest.mark.parametrize("test_input", test_dems_ellipsoid + test_dems_geoid)
def test_rema_dem_for_bounds(test_input: TestDem):

    bounds = test_input.requested_bounds
    dem_file = test_input.dem_file
    resolution = test_input.resolution
    num_tasks = test_input.num_tasks
    geoid_path = test_input.geoid
    ellipsoid_heights = test_input.ellipsoid_heights

    expected_dem_name = Path(dem_file).name

    if not TMP_PATH.exists():
        TMP_PATH.mkdir(parents=True, exist_ok=True)

    SAVE_PATH = TMP_PATH / Path(f"{expected_dem_name}")
    array, profile, _ = get_rema_dem_for_bounds(
        bounds,
        save_path=SAVE_PATH,
        rema_index_path=REMA_INDEX_PATH,
        resolution=resolution,
        bounds_src_crs=4326,
        ellipsoid_heights=ellipsoid_heights,
        num_tasks=num_tasks,
        geoid_tif_path=geoid_path,
        download_geoid=False,
    )

    with rio.open(dem_file, "r") as src:
        expected_array = src.read(1)

    assert_allclose(array, expected_array)

    with rio.open(SAVE_PATH) as src:
        array = src.read(1)

    assert_allclose(array, expected_array)

    # Once complete, remove the TMP files and directory
    shutil.rmtree(TMP_PATH)
