import logging
from pathlib import Path

from dem_handler.utils.aws import AsyncS3Util

logging.basicConfig(level=logging.INFO)
CURRENT_DIR = Path(__file__).parent.resolve()
BUCKET = "deant-data-public-dev"
PREFIX = "persistent/repositories/dem-handler/test-data/"


def main():

    s3_util = AsyncS3Util()
    s3_objects = s3_util.get_objects_in_bucket(bucket_name=BUCKET, prefix=PREFIX, files_only=True)
    s3_util.bulk_download_objects(
        s3_objects=s3_objects,
        download_dir=CURRENT_DIR,
        bucket_name=BUCKET,
        relative_to_s3_prefix=PREFIX,
        skip_existing=False,
    )


if __name__ == "__main__":
    main()
