from __future__ import annotations

import asyncio
import logging
import os
from asyncio import gather
from pathlib import Path

import aioboto3
import multiprocess as mp
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class AsyncS3Util:
    def __init__(
        self,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region_name: str = "ap-southeast-2",
        retry_config: Config = Config(
            region_name="ap-southeast-2",
            retries={"max_attempts": 3, "mode": "standard"}, # Retry configuration for S3 operations, Max attempts set to 3 and mode set to standard
            max_pool_connections=50, # Maximum number of simultaneous connections in the connection pool, set to 50.
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
        """Initialize the AWS S3 client with specified configurations.
        
        Parameters
        ----------
        aws_access_key_id : str | None, optional
            AWS access key ID, by default None
        aws_secret_access_key : str | None, optional
            AWS secret access key, by default None
        aws_session_token : str | None, optional
            AWS session token, by default None
        region_name : str, optional
            AWS region name, by default "ap-southeast-2"
        retry_config : Config, optional
            Retry configuration for S3 operations, 
            by default 
            ```
            Config(
                region_name="ap-southeast-2", 
                retries={"max_attempts": 3, "mode": "standard"}, 
                max_pool_connections=50)
            ```
            Maximum number of simultaneous connections in the connection pool is set to 50 to avoid connection issues during high concurrency operations.
        transfer_config : TransferConfig, optional
            Transfer configuration for S3 operations, 
            by default 
            ```
            TransferConfig(
                multipart_threshold=1024 * 1024 * 50, 
                multipart_chunksize=1024 * 1024 * 25, 
                num_download_attempts=5, 
                max_concurrency=8
            )
            ```
            `max_concurrency` is set to 8 to allow multiple threads to download/upload files concurrently, improving performance for large files.
            `num_download_attempts` is set to 5 to retry failed downloads for each chunk, improving reliability in case of transient network issues.
        num_cpus : int, optional
            Number of CPUs to use for multiprocessing, by default 1.
            Used for multiprocessing, if set to -1, it will use all available CPUs. 
            Chunks of work will be divided among the specified number of CPUs for parallel processing.
            Each CPU will handle a portion of the download/upload tasks, running the chunks asynchronously within each process.
        num_tasks : int, optional
            Number of tasks to divide the work into for asynchronous operations, by default 8.
            Total number of files will be divided into chunks using num_tasks, and each chunk will be processed asynchronously.
            if num_tasks is set to -1, all files will be processed in a single chunk.
        
        **NOTE**:
            If both num_cpus and num_tasks are passed, the file list will be divided into num_tasks chunks, and each chunk will be asynchronously processed by one of the num_cpus processes.
            For a small number of files, it is recommended to set num_cpus=1 and num_tasks>1, so that the files can be processed asynchronously in a single process

        """

        if not aws_access_key_id or not aws_secret_access_key or not aws_session_token:
            logger.warning(
                f"No or incomplete AWS credentials provided. Attempting to use environment variables."
            )
            # retry_config.signature_version = UNSIGNED # aioboto3 does not support unsigned requests, so we will use the default signature version
            self.session = aioboto3.Session(region_name=region_name)
        else:
            logger.info(f"Using provided AWS credentials for S3 access.")
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
        save_path: Path,
        bucket: aioboto3.S3.Bucket,
        skip_existing: bool = True,
    ):
        """Download a dem tile from AWS and save to specified folder

        Parameters
        ----------
        s3_object : Path
            DEM tile S3 object. e.g. Copernicus_DSM_COG_10_S78_00_E166_00_DEM/Copernicus_DSM_COG_10_S78_00_E166_00_DEM.tif
        save_path : Path
            Path to save the downloaded object
        bucket : aioboto3.S3.Bucket
            S3 bucket object
        skip_existing : bool, optional
            If True, skip downloading if the file already exists locally, by default True
        """

        local_directory = save_path.parent
        if not local_directory.exists():
            local_directory.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Downloading dem tile : {s3_object.as_posix()}, save location : {save_path.as_posix()}"
        )

        if skip_existing and save_path.exists():
            logger.info(
                f"Skipping download of {s3_object.as_posix()} as it already exists at {save_path.as_posix()}"
            )
            return
        
        return await bucket.download_file(
            s3_object.as_posix(), save_path.as_posix(), Config=self.transfer_config
        )

    async def upload_object(
        self,
        s3_object: Path,
        local_path: Path,
        bucket: aioboto3.S3.Bucket,
        skip_existing: bool = True,
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
        skip_existing : bool, optional
            If True, skip uploading if the file already exists in S3, by default True
        """

        logger.info(
            f"Uploading dem tile : {local_path.as_posix()}, s3 location : {s3_object.as_posix()}"
        )

        key_exists = False
        if skip_existing:
            try:
                await bucket.head_object(Key=s3_object.as_posix())
                key_exists = True
            except ClientError as e:
                # Extract the HTTP error code from the exception response
                error_code = e.response["Error"]["Code"]

                if error_code == "404":
                    # File definitely does not exist
                    key_exists = False
                elif error_code == "403":
                    # File may exist, but your IAM credentials lack s3:GetObject or s3:ListBucket permissions
                    logger.info(False, f"Access Denied to {s3_object.as_posix()}")
                    key_exists = False
                else:
                    # Re-raise for unexpected API failures
                    raise e

        if key_exists:
            logger.info(
                f"Skipping upload of {local_path.as_posix()} as it already exists at {s3_object.as_posix()}"
            )
            return
        return await bucket.upload_file(
            local_path.as_posix(),
            s3_object.as_posix(),
            Config=self.transfer_config,
        )

    def single_download_process(
        self,
        s3_objects: list[Path],
        save_paths: list[Path],
        bucket_name: str,
        skip_existing: bool = True,
    ):
        """Single process for asynchronous download.

        Parameters
        ----------
        s3_objects : list[Path]
            List of S3 object paths
        save_paths : list[Path]
            Local paths to save the files
        bucket_name : str
            Name of the S3 bucket
        """

        async def download(s3o, sps, bn, sess, skp):
            async with sess.resource(
                "s3",
                config=self.retry_config,
            ) as sr:
                bucket = await sr.Bucket(bn)
                tasks = [self.download_object(i, sp, bucket, skp) for i, sp in zip(s3o, sps)]
                results = await gather(*tasks, return_exceptions=True)

                for tile, result in zip(s3o, results):
                    if isinstance(result, Exception):
                        logger.error(
                            "Failed to download %s: %s",
                            tile,
                            result,
                        )

        asyncio.run(
            download(
                s3_objects,
                save_paths,
                bucket_name,
                self.session,
                skip_existing,
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
        relative_to_s3_prefix: str | None = None,
        skip_existing: bool = True,
    ) -> list[Path]:
        """Asynchronous download of S3 objects from S3 bucket to local folder.

        Parameters
        ----------
        s3_objects : list[Path]
            List of S3 object paths.
        download_dir : Path
            Local folder to save the files
        bucket_name : str
            Name of S3 bucket
        relative_to_s3_prefix : str | None, optional
            If provided, the local paths will be relative to this prefix in the S3 bucket.
            For example, if the S3 object is "prefix/subdir/file.tif" and relative_to_s3_prefix is "prefix", the local path will be "download_dir/subdir/file.tif". 
            If not provided, the local paths will be "download_dir/file.tif". By "download_dir/file.tif", it means the file will be saved directly under the download_dir with its original name.
        skip_existing : bool, optional
            If True, existing files will be skipped and not downloaded again, by default True

        Returns
        -------
        list[Path]
            List of local paths to the saved files.
        """

        def _relative_to_s3_prefix(s3_obj: Path, pref:str) -> Path:
            if pref is not None:
                return s3_obj.relative_to(pref)
            return s3_obj.name

        download_dir.mkdir(parents=True, exist_ok=True)

        save_paths = [download_dir / _relative_to_s3_prefix(t, relative_to_s3_prefix) for t in s3_objects]

        download_list_chunks, save_paths_chunks = (
            [s3_objects[i :: self.num_tasks] for i in range(self.num_tasks)],
            [save_paths[i :: self.num_tasks] for i in range(self.num_tasks)],
        ) if self.num_tasks != -1 else ([s3_objects], [save_paths])

        if self.num_cpus == 1:
            for ch, sp in zip(download_list_chunks, save_paths_chunks):
                self.single_download_process(
                    ch,
                    sp,
                    bucket_name,
                    skip_existing,
                )
        else:
            if self.num_cpus == -1:
                self.num_cpus = mp.cpu_count()
            with mp.Pool(self.num_cpus) as p:
                p.starmap(
                    self.single_download_process,
                    [
                        (
                            el[0],
                            el[1],
                            bucket_name,
                            skip_existing,
                        )
                        for el in list(zip(download_list_chunks, save_paths_chunks))
                    ],
                )

        return [download_dir / _relative_to_s3_prefix(t, relative_to_s3_prefix) for t in s3_objects]

    def bulk_upload_objects(
        self,
        local_objects: list[Path],
        s3_dir: Path,
        bucket_name: str,
        keep_dir_structure: bool = True,
        remove_parent_dir: bool = False,
    ) -> list[Path]:
        """Asynchronous upload of local objects to S3 bucket.

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

        upload_list_chunks, local_list_chunks = (
            [upload_objects[i :: self.num_tasks] for i in range(self.num_tasks)],
            [local_objects[i :: self.num_tasks] for i in range(self.num_tasks)]
        ) if self.num_tasks != -1 else ([upload_objects], [local_objects])

        if self.num_cpus == 1:
            for ch, ll in zip(upload_list_chunks, local_list_chunks):
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
                        for el in list(zip(upload_list_chunks, local_list_chunks))
                    ],
                )

        return upload_objects


    def get_objects_in_bucket(self, bucket_name: str, prefix: str, files_only: bool = True, full_s3_path: bool = False) -> list[str]:
        """Find all objects in an AWS S3 bucket for a given prefix

        Parameters
        ----------
        bucket_name : str
            Name of the s3 bucket
        prefix : str
            Name of the prefix in the s3 bucket
        files_only : bool, optional
            If True, return only files (objects with a suffix), by default True
        full_s3_path : bool, optional
            If True, return the full S3 path (s3://bucket_name/object_key),
            otherwise return just the object key, by default False

        Returns
        -------
        list[str]
            List of objects
        """

        async def _async_get_objects_in_bucket(bn: str, pfx: str, fo: bool = True, fp: bool = False) -> list[Path]:
            object_list = []
            params = {"Bucket": bn, "Prefix": pfx}

            async with self.session.client("s3", config=self.retry_config) as client:
                objects = await client.list_objects_v2(**params)

            if "Contents" in objects.keys():
                object_list.extend([x["Key"] for x in objects["Contents"]])

            if fo:
                object_list = [Path(x) for x in object_list if Path(x).suffix != "" and Path(x).suffix is not None]

            if fp:
                object_list = [Path(f"s3://{bn}/{str(x)}") for x in object_list]
            return object_list

        return asyncio.run(_async_get_objects_in_bucket(bucket_name, prefix, files_only, full_s3_path))
