from pathlib import Path
from dem_handler.download.aio_aws import (
    bulk_download_dem_tiles,
)  # , bulk_upload_dem_tiles
from botocore.config import Config

CURRENT_DIR = Path(__file__).parent.resolve()
TMP_PATH = CURRENT_DIR / "TMP/Async_Test"

S3_BUCKET = "deant-data-public-dev"
REMOTE_DIR = "persistent/repositories/dem-handler/async_test/"
REMOTE_FILES = [
    "Copernicus_DSM_COG_10_N00_00_E009_00_DEM.tif",
    "Copernicus_DSM_COG_10_N00_00_E010_00_DEM.tif",
    "Copernicus_DSM_COG_10_N00_00_E011_00_DEM.tif",
    "Copernicus_DSM_COG_10_N00_00_E013_00_DEM.tif",
    "Copernicus_DSM_COG_10_N00_00_E014_00_DEM.tif",
    "Copernicus_DSM_COG_10_N00_00_E017_00_DEM.tif",
]
tile_objects = [REMOTE_DIR / Path(rf) for rf in REMOTE_FILES]

CONFIG = Config(
    region_name="ap-southeast-2",
    retries={"max_attempts": 3, "mode": "standard"},
)


def test_bulk_download():
    bulk_download_dem_tiles(tile_objects, TMP_PATH, S3_BUCKET, CONFIG, 2, 2)


# This needs AWS access keys, we should provid them if this test needs to run.
# def test_bulk_upload():
#     bulk_upload_dem_tiles(REMOTE_DIR, TMP_PATH, S3_BUCKET, CONFIG, 2, 2)
