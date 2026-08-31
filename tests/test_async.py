from pathlib import Path

from botocore.config import Config

from dem_handler.utils.aws import AsyncS3Util

CURRENT_DIR = Path(__file__).parent.resolve()
TMP_PATH = CURRENT_DIR / "TMP"

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

ASYNC_S3_UTIL = AsyncS3Util(retry_config=CONFIG, num_cpus=2, num_tasks=2)


def test_bulk_download():
    ASYNC_S3_UTIL.bulk_download_objects(
        tile_objects,
        TMP_PATH,
        S3_BUCKET,
        relative_to_s3_prefix="persistent/repositories/dem-handler",
    )
