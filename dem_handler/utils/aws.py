from __future__ import annotations

import asyncio
import glob
import logging
import os
from asyncio import gather
from pathlib import Path

import aioboto3
import boto3
import multiprocess as mp
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.client import Config

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


class AsyncS3Util:
    def __init__(
        self,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None,
        region_name="ap-southeast-2",
        retry_config: Config = Config(
            region_name="ap-southeast-2",
            retries={"max_attempts": 3, "mode": "standard"},
            max_pool_connections=50,
        ),
        transfer_config: TransferConfig = TransferConfig(
            multipart_threshold=1024 * 1024 * 50,  # 50 MB
            multipart_chunksize=1024 * 1024 * 25,  # 25 MB
            num_download_attempts=5,  # Retries per chunk
            max_concurrency=8,  # Number of threads to use for downloading
        ),
        num_cpus: int = 1,
        num_tasks: int = 8,
    ):

        if not aws_access_key_id:
            logger.warning(
                f"No credentials provided. Attempting to use environment variables"
            )

        retry_config.signature_version = UNSIGNED

        self.session = aioboto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
        )

        self.retry_config = retry_config
        self.transfer_config = transfer_config
        self.num_cpus = num_cpus
        self.num_tasks = num_tasks

    async def download_object(
        self,
        s3_object: Path,
        save_folder: Path,
        bucket: aioboto3.S3.Bucket,
    ):
        """Download a dem tile from AWS and save to specified folder

        Parameters
        ----------
        s3_object : Path
            DEM tile S3 object. e.g. Copernicus_DSM_COG_10_S78_00_E166_00_DEM/Copernicus_DSM_COG_10_S78_00_E166_00_DEM.tif
        save_folder : Path
            Folder to save the downloaded tif
        bucket : aioboto3.S3.Bucket
            S3 bucket object
        """

        save_path = save_folder / s3_object.name
        logger.info(
            f"Downloading dem tile : {s3_object.as_posix()}, save location : {save_path.as_posix()}"
        )
        return await bucket.download_file(
            s3_object.as_posix(), save_path.as_posix(), Config=self.transfer_config
        )

    async def upload_object(
        self,
        s3_object: Path,
        local_path: Path,
        bucket: aioboto3.S3.Bucket,
    ):
        """Upload a dem tile to AWS from local path and save to specified path

        Parameters
        ----------
        s3_object : Path
            DEM tile filename. e.g. Copernicus_DSM_COG_10_S78_00_E166_00_DEM.tif
        local_path : Path
            Local path to the file.
        bucket: aioboto3.S3.Bucket
            S3 bucket object
        """

        logger.info(
            f"Uploading dem tile : {local_path.as_posix()}, s3 location : {s3_object.as_posix()}"
        )
        return await bucket.upload_file(
            local_path.as_posix(),
            s3_object.as_posix(),
            Config=self.transfer_config,
        )

    def single_download_process(
        self,
        s3_objects: list[Path],
        save_folder: Path,
        bucket_name: str,
    ):
        """Single process for asynchronous download.

        Parameters
        ----------
        s3_objects : list[Path]
            List of S3 object paths
        save_folder : Path
            Local folder to save the files
        bucket_name : str
            Name of the S3 bucket
        """

        async def download(to, dir, bn, sess):
            async with sess.resource(
                "s3",
                config=self.retry_config,
            ) as sr:
                bucket = await sr.Bucket(bn)
                tasks = [self.download_object(i, dir, bucket) for i in to]
                results = await gather(*tasks, return_exceptions=True)

                for tile, result in zip(to, results):
                    if isinstance(result, Exception):
                        logger.error(
                            "Failed to download %s: %s",
                            tile,
                            result,
                        )

        asyncio.run(
            download(
                s3_objects,
                save_folder,
                bucket_name,
                self.session,
            )
        )

    def single_upload_process(
        self,
        s3_objects: list[Path],
        local_paths: list[Path],
        bucket_name: str,
    ):
        """Single process for asynchronous upload.

        Parameters
        ----------
        s3_objects : list[Path]
            List of s3 object paths to be created.
        local_paths : list[Path]
            List of local paths to tiles.
        bucket_name : str
            Name of the S3 bucket
        """

        async def upload(to, lp, bn, sess):
            async with sess.resource(
                "s3",
                config=self.retry_config,
            ) as sr:
                bucket = await sr.Bucket(bn)
                tasks = [self.upload_object(i, l, bucket) for i, l in zip(to, lp)]
                results = await gather(*tasks, return_exceptions=True)

                for tile, result in zip(to, results):
                    if isinstance(result, Exception):
                        logger.error(
                            "Failed to upload %s: %s",
                            tile,
                            result,
                        )

        asyncio.run(
            upload(
                s3_objects,
                local_paths,
                bucket_name,
                self.session,
            )
        )

    def bulk_download_objects(
        self,
        s3_objects: list[Path],
        download_dir: Path,
        bucket_name: str,
    ) -> list[Path]:
        """Asynchronous download of DEM objects from S3

        Parameters
        ----------
        s3_objects : list[Path]
            List of S3 object paths
        download_dir : Path
            Local folder to save the files
        bucket_name : str
            Name of S3 bucket

        Returns
        -------
        list[Path]
            List of local paths to the saved files.
        """

        os.makedirs(download_dir, exist_ok=True)
        download_list_chunk = (
            [s3_objects[i :: self.num_tasks] for i in range(self.num_tasks)]
            if self.num_tasks != -1
            else [s3_objects]
        )
        if self.num_cpus == 1:
            for ch in download_list_chunk:
                self.single_download_process(
                    ch,
                    download_dir,
                    bucket_name,
                )
        else:
            if self.num_cpus == -1:
                self.num_cpus = mp.cpu_count()
            with mp.Pool(self.num_cpus) as p:
                p.starmap(
                    self.single_download_process,
                    [
                        (
                            ch,
                            download_dir,
                            bucket_name,
                        )
                        for ch in download_list_chunk
                    ],
                )

        return [download_dir / t.name for t in s3_objects]

    def bulk_upload_objects(
        self,
        local_objects: list[Path],
        s3_dir: Path,
        bucket_name: str,
        keep_dir_structure: bool = True,
        remove_parent_dir: bool = False,
    ) -> list[Path]:
        """Asynchronous upload of DEM objects to S3

        Parameters
        ----------
        local_objects : list[Path]
            Local paths to files.
        s3_dir : Path
            S3 directory to upload files to
        bucket_name : str
            Name of the S3 bucket
        keep_dir_structure : bool, optional
            Whether to keep the directory structure of the local files when uploading to S3, by default True
        remove_parent_dir : bool, optional
            Whether to remove the parent directory of the local files when uploading to S3, by default False.
            This is useful when the local files are in a subdirectory and you want to upload them without including the parent directory.

        Returns
        -------
        list[Path]
            List of remote paths on S3.
        """

        if keep_dir_structure:
            # in case the local paths have ../ in them, we want to remove that for the s3 path
            object_dirs = [Path(str(tp).replace("../", "")) for tp in local_objects]
            if remove_parent_dir:
                object_dirs = [Path(*tp.parts[1:]) for tp in object_dirs]
        else:
            object_dirs = [tp.name for tp in local_objects]
        upload_objects = [s3_dir / od for od in object_dirs]

        upload_list_chunk = (
            [upload_objects[i :: self.num_tasks] for i in range(self.num_tasks)]
            if self.num_tasks != -1
            else [upload_objects]
        )
        local_list_chunk = (
            [local_objects[i :: self.num_tasks] for i in range(self.num_tasks)]
            if self.num_tasks != -1
            else [local_objects]
        )
        if self.num_cpus == 1:
            for ch, ll in zip(upload_list_chunk, local_list_chunk):
                self.single_upload_process(
                    ch,
                    ll,
                    bucket_name,
                )
        else:
            if self.num_cpus == -1:
                self.num_cpus = mp.cpu_count()
            with mp.Pool(self.num_cpus) as p:
                p.starmap(
                    self.single_upload_process,
                    [
                        (
                            el[0],
                            el[1],
                            bucket_name,
                        )
                        for el in list(zip(upload_list_chunk, local_list_chunk))
                    ],
                )

        return upload_objects
