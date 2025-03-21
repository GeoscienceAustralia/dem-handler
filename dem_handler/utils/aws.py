import boto3
import logging
from botocore import UNSIGNED
from botocore.client import Config
from pathlib import Path

logger = logging.getLogger(__name__)


class S3Util:
    def __init__(
        self,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None,
        region_name="ap-southeast-2",
    ):

        if not aws_access_key_id:
            logger.warning(
                f"No credentials provided. Attempting to use environment variables"
            )

        self.client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
            config=Config(signature_version=UNSIGNED),
        )

    def get_objects_in_bucket(self, s3_bucket: str, s3_prefix: str) -> list[str]:
        """Find all objects in an AWS S3 bucket for a given prefix

        Parameters
        ----------
        s3_bucket : str
            Name of the s3 bucket
        s3_prefix : str
            Name of the prefix in the s3 bucket

        Returns
        -------
        list[str]
            List of objects
        """
        object_list = []
        params = {"Bucket": s3_bucket, "Prefix": s3_prefix}
        objects = self.client.list_objects_v2(**params)
        if "Contents" in objects.keys():
            object_list.extend([x["Key"] for x in objects["Contents"]])

        return object_list

    def download_s3_file(self, s3_bucket: str, s3_key: str, local_file: Path):
        """Download a single S3 file

        Parameters
        ----------
        s3_bucket : str
            Name of the s3 bucket
        s3_key : str
            Name of the file on s3 (relative to the bucket)
        local_file : Path
            Desired path for the local file
        """

        local_directory = local_file.parent
        if not local_directory.exists():
            local_directory.mkdir(parents=True, exist_ok=True)

        if not local_file.exists():
            self.client.download_file(s3_bucket, s3_key, local_file)

    def download_files_in_bucket(
        self, s3_bucket: str, s3_prefix: str, local_prefix: Path
    ):
        """Identify and download all files in an AWS S3 bucket.
        Objects will be downloaded locally relative to the prefix.
        e.g. a file at <s3_prefix>/path/to/file will be downloaded to <local_prefix>/path/to/file

        Parameters
        ----------
        s3_bucket : str
            Name of the s3 bucket
        s3_prefix : str
            Name of the prefix in the s3 bucket
        local_directory : Path
            Path to the local directory in which to download the files
        """
        object_list = self.get_objects_in_bucket(s3_bucket, s3_prefix)

        file_list = [
            x
            for x in object_list
            if Path(x).suffix != "" and Path(x).suffix is not None
        ]

        for s3_file in file_list:
            local_path = local_prefix / Path(s3_file).relative_to(s3_prefix)
            if not local_path.exists():
                logger.info(f"downloading {local_path}")
                self.download_s3_file(s3_bucket, s3_file, local_path)
            else:
                logger.info(f"file found at {local_path}")
