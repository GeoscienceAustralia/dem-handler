from __future__ import annotations
import aioboto3
import asyncio
from asyncio import gather
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.config import Config
import os
from pathlib import Path
import multiprocess as mp
import glob


import logging

logger = logging.getLogger(__name__)


async def download_dem_tile(
    tile_object: Path,
    save_folder: Path,
    bucket: aioboto3.S3.Bucket,
    transfer_config: TransferConfig,
):
    """Download a dem tile from AWS and save to specified folder

    Parameters
    ----------
    tile_object : Path
        DEM tile S3 object. e.g. Copernicus_DSM_COG_10_S78_00_E166_00_DEM/Copernicus_DSM_COG_10_S78_00_E166_00_DEM.tif
    save_folder : Path
        Folder to save the downloaded tif
    bucket : aioboto3.S3.Bucket
        S3 bucket object
    transfer_config : TransferConfig
        TransferConfig for download
    """

    save_path = save_folder / tile_object.name
    logger.info(
        f"Downloading dem tile : {tile_object.as_posix()}, save location : {save_path.as_posix()}"
    )
    return await bucket.download_file(
        tile_object.as_posix(), save_path.as_posix(), Config=transfer_config
    )


async def upload_dem_tile(
    tile_object: Path,
    local_path: Path,
    bucket: aioboto3.S3.Bucket,
    transfer_config: TransferConfig,
):
    """Upload a dem tile to AWS from local path and save to specified path

    Parameters
    ----------
    tile_object : Path
        DEM tile filename. e.g. Copernicus_DSM_COG_10_S78_00_E166_00_DEM.tif
    local_path : Path
        Local path to the file.
    bucket: aioboto3.S3.Bucket
        S3 bucket object
    transfer_config: TransferConfig
    """

    logger.info(
        f"Uploading dem tile : {local_path.as_posix()}, s3 location : {tile_object.as_posix()}"
    )
    return await bucket.upload_file(
        local_path.as_posix(),
        tile_object.as_posix(),
        Config=transfer_config,
    )


def single_download_process(
    tile_objects: list[Path],
    save_folder: Path,
    retry_config: Config,
    bucket_name: str,
    session: aioboto3.Session,
    transfer_config: TransferConfig,
):
    """Single process for asynchronous download.

    Parameters
    ----------
    tile_objects : list[Path]
        List of S3 object paths
    save_folder : Path
        Local folder to save the files
    retry_config : botocore Config
    bucket_name : str
        Name of the S3 bucket
    session : aioboto3.Session
    transfer_config : TransferConfig
    """

    if retry_config.signature_version == "":
        retry_config.signature_version = UNSIGNED

    async def download(to, dir, rc, bn, sess, tc):
        async with sess.resource(
            "s3",
            config=rc,
        ) as sr:
            bucket = await sr.Bucket(bn)
            tasks = [download_dem_tile(i, dir, bucket, tc) for i in to]
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
            tile_objects,
            save_folder,
            retry_config,
            bucket_name,
            session,
            transfer_config,
        )
    )


def single_upload_process(
    tile_objects: list[Path],
    local_paths: list[Path],
    retry_config: Config,
    bucket_name: str,
    session: aioboto3.Session,
    transfer_config: TransferConfig,
):
    """Single process for asynchronous upload.

    Parameters
    ----------
    tile_objects : list[Path]
        List of s3 object paths to be created.
    local_paths : list[Path]
        List of local paths to tiles.
    retry_config : botocore Config
    bucket_name : str
        Name of the S3 bucket
    session : aioboto3.Session
    transfer_config : TransferConfig
    """

    if retry_config.signature_version == "":
        retry_config.signature_version = UNSIGNED

    async def upload(to, lp, rc, bn, sess, tc):
        async with sess.resource(
            "s3",
            config=rc,
        ) as sr:
            bucket = await sr.Bucket(bn)
            tasks = [upload_dem_tile(i, l, bucket, tc) for i, l in zip(to, lp)]
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
            tile_objects,
            local_paths,
            retry_config,
            bucket_name,
            session,
            transfer_config,
        )
    )


def bulk_download_dem_tiles(
    tile_objects: list[Path],
    save_folder: Path,
    bucket_name: str = "copernicus-dem-30m",
    retry_config: Config = Config(
        region_name="eu-central-1",
        retries={"max_attempts": 3, "mode": "standard"},
    ),
    num_cpus: int = 1,
    num_tasks: int = 8,
    session: aioboto3.Session | None = None,
    transfer_config: TransferConfig = TransferConfig(
        multipart_threshold=1024 * 1024 * 50,  # 50 MB
        multipart_chunksize=1024 * 1024 * 25,  # 25 MB
        num_download_attempts=5,  # Retries per chunk
    ),
) -> list[Path]:
    """Asynchronous download of DEM objects from S3

    Parameters
    ----------
    tile_objects : list[Path]
        List of S3 object paths
    save_folder : Path
        Local folder to save the files
    bucket_name : str, optional
        Name of S3 bucket, by default "copernicus-dem-30m"
    retry_config : Config, optional
        botocore Config, by default Config( signature_version="", region_name="eu-central-1", retries={"max_attempts": 3, "mode": "standard"}, )
    num_cpus : int, optional
        Number of cpus to be used for multi-processing, by default 1.
        Setting to -1 will use all available cpus
    num_tasks : int, optional
        Number of tasks to be run in async mode, by default 8
        If num_cpus > 1, each task will be assigned to a cpu and will run in async mode on that cpu (multiple threads).
        Setting to -1 will transfer all tiles in one task.
    session : aioboto3.Session | None, optional
        aioboto3.Session, by default None
    transfer_config : TransferConfig, optional
        TransferConfig for download, by default
        TransferConfig
        (
            multipart_threshold=1024 * 1024 * 50,  # 50 MB
            multipart_chunksize=1024 * 1024 * 25,  # 25 MB
            num_download_attempts=5,  # Retries per chunk
        )

    Returns
    -------
    list[Path]
        List of local paths to the saved files.
    """

    if not session:
        session = aioboto3.Session()
        retry_config.signature_version = ""

    os.makedirs(save_folder, exist_ok=True)
    download_list_chunk = (
        [tile_objects[i::num_tasks] for i in range(num_tasks)]
        if num_tasks != -1
        else [tile_objects]
    )
    if num_cpus == 1:
        for ch in download_list_chunk:
            single_download_process(
                ch, save_folder, retry_config, bucket_name, session, transfer_config
            )
    else:
        if num_cpus == -1:
            num_cpus = mp.cpu_count()
        with mp.Pool(num_cpus) as p:
            p.starmap(
                single_download_process,
                [
                    (
                        ch,
                        save_folder,
                        retry_config,
                        bucket_name,
                        session,
                        transfer_config,
                    )
                    for ch in download_list_chunk
                ],
            )

    return [save_folder / t.name for t in tile_objects]


def bulk_upload_dem_tiles(
    s3_dir: Path,
    local_dir: Path,
    bucket_name: str = "deant-data-public-dev",
    retry_config: Config = Config(
        region_name="ap-southeast-2",
        retries={"max_attempts": 3, "mode": "standard"},
        max_pool_connections=50,
    ),
    num_cpus: int = 1,
    num_tasks: int = 8,
    session: aioboto3.Session | None = None,
    transfer_config: TransferConfig = TransferConfig(
        multipart_threshold=1024 * 1024 * 50,  # 50 MB
        multipart_chunksize=1024 * 1024 * 25,  # 25 MB
    ),
) -> list[Path]:
    """Asynchronous upload of DEM objects to S3

    Parameters
    ----------
    s3_dir : Path
        S3 directory to upload files to
    local_dir : Path
        Local path to files.
    bucket_name : str, optional
        Name of the S3 bucket, by default "deant-data-public-dev"
    config : Config, optional
        botorcore Config, by default Config( region_name="ap-southeast-2", retries={"max_attempts": 3, "mode": "standard"}, )
    num_cpus : int, optional
        Number of cpus to be used for multi-processing, by default 1.
        Setting to -1 will use all available cpus
    num_tasks : int, optional
        Number of tasks to be run in async mode, by default 8
        If num_cpus > 1, each task will be assigned to a cpu and will run in async mode on that cpu (multiple threads).
        Setting to -1 will transfer all tiles in one task.
    session : aioboto3.Session | None, optional
        aioboto3.Session, by default None
    transfer_config : TransferConfig, optional
        TransferConfig for upload, by default
        TransferConfig
        (
            multipart_threshold=1024 * 1024 * 50,
            multipart_chunksize=1024 * 1024 * 25,
        )


    Returns
    -------
    list[Path]
        List of remote paths on S3.
    """

    if not session:
        session = aioboto3.Session()
        retry_config.signature_version = ""

    tile_paths = [
        Path(t)
        for t in list(
            filter(
                lambda f: f.endswith(".tif"),
                glob.glob(f"{local_dir}/**", recursive=True),
            )
        )
    ]
    tiles_dirs = [Path(*tp.parts[1:]) for tp in tile_paths]
    tile_objects = [s3_dir / td for td in tiles_dirs]

    upload_list_chunk = (
        [tile_objects[i::num_tasks] for i in range(num_tasks)]
        if num_tasks != -1
        else [tile_objects]
    )
    local_list_chunk = (
        [tile_paths[i::num_tasks] for i in range(num_tasks)]
        if num_tasks != -1
        else [tile_paths]
    )
    if num_cpus == 1:
        for ch, ll in zip(upload_list_chunk, local_list_chunk):
            single_upload_process(
                ch, ll, retry_config, bucket_name, session, transfer_config
            )
    else:
        if num_cpus == -1:
            num_cpus = mp.cpu_count()
        with mp.Pool(num_cpus) as p:
            p.starmap(
                single_upload_process,
                [
                    (el[0], el[1], retry_config, bucket_name, session, transfer_config)
                    for el in list(zip(upload_list_chunk, local_list_chunk))
                ],
            )

    return tile_objects
