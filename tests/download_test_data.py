from dem_handler.utils.aws import S3Util

import logging

logging.basicConfig(level=logging.INFO)


def main():
    BUCKET = "deant-data-public-dev"
    PREFIX = "persistent/repositories/dem-handler/test-data/"

    s3_util = S3Util()

    s3_util.download_files_in_bucket(BUCKET, PREFIX)


if __name__ == "__main__":
    main()
