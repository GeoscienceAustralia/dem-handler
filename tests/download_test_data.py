from dem_handler.utils.aws import S3Util

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
CURRENT_DIR = Path(__file__).parent.resolve()
BUCKET = "deant-data-public-dev"
PREFIX = "persistent/repositories/dem-handler/test-data/"


def main():

    s3_util = S3Util()

    s3_util.download_files_in_bucket(BUCKET, PREFIX, CURRENT_DIR)


if __name__ == "__main__":
    main()
