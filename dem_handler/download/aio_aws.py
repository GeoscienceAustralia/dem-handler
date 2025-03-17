import aioboto3
import asyncio
from asyncio import gather
from botocore import UNSIGNED
from botocore.config import Config
import os
from pathlib import Path
import multiprocess as mp
import glob


async def download_dem_tile(
    tile_object: Path,
    save_folder: Path,
    bucket_name: str,
    config: Config,
    session: aioboto3.Session,
):
    """Download a dem tile from AWS and save to specified folder

    Parameters
    ----------
    tile_object : Path
        DEM tile S3 object. e.g. Copernicus_DSM_COG_10_S78_00_E166_00_DEM/Copernicus_DSM_COG_10_S78_00_E166_00_DEM.tif
    save_folder : Path
        Folder to save the downloaded tif
    bucket_name: str
        Name of the S3 bucket
    config: botocore Config
    session: aioboto3 Session
    """

    if not config.signature_version:
        config.signature_version = UNSIGNED

    async with session.resource("s3", config=config) as s3:
        bucket = await s3.Bucket(bucket_name)
        save_path = save_folder / tile_object.name
        print(
            f"Downloading dem tile : {tile_object.as_posix()}, save location : {save_path.as_posix()}"
        )
        try:
            return await bucket.download_file(tile_object.as_posix(), save_path)
        except Exception as e:
            raise (e)


async def upload_dem_tile(
    tile_object: Path,
    local_path: Path,
    bucket_name: str,
    config: Config,
    session: aioboto3.Session,
):
    """Upload a dem tile to AWS from local path and save to specified path

    Parameters
    ----------
    tile_object : Path
        DEM tile filename. e.g. Copernicus_DSM_COG_10_S78_00_E166_00_DEM.tif
    local_path : Path
        Local path to the file.
    bucket_name: str
        Name of the S3 bucket
    config: botocore Config
    session: aioboto3 Session
    """

    async with session.resource(
        "s3",
        config=config,
    ) as s3:
        bucket = await s3.Bucket(bucket_name)
        print(
            f"Uploading dem tile : {local_path.as_posix()}, s3 location : {tile_object.as_posix()}"
        )
        try:
            return await bucket.upload_file(
                local_path,
                tile_object.as_posix(),
            )
        except Exception as e:
            raise (e)


def single_download_process(
    tile_objects: list[Path],
    save_folder: Path,
    config: Config,
    bucket_name: str,
    session: aioboto3.Session,
):
    """Single process for asynchronous download.

    Parameters
    ----------
    tile_objects : list[Path]
        List of S3 object paths
    save_folder : Path
        Local folder to save the files
    config : botocore Config
    bucket_name : str
        Name of the S3 bucket
    session : aioboto3.Session
    """

    async def download(to, dir, cf, bn, sess):
        tasks = [download_dem_tile(i, dir, bn, cf, sess) for i in to]
        await gather(*tasks)

    asyncio.run(download(tile_objects, save_folder, config, bucket_name, session))


def single_upload_process(
    tile_objects: list[Path],
    local_paths: list[Path],
    config: Config,
    bucket_name: str,
    session: aioboto3.Session,
):
    """Single process for asynchronous upload.

    Parameters
    ----------
    tile_objects : list[Path]
        List of s3 object paths to be created.
    local_paths : list[Path]
        List of local paths to tiles.
    config : botocore Config
    bucket_name : str
        Name of the S3 bucket
    session : aioboto3.Session
    """

    async def upload(to, lp, cf, bn, sess):
        tasks = [upload_dem_tile(i, l, bn, cf, sess) for i, l in zip(to, lp)]
        await gather(*tasks)

    asyncio.run(upload(tile_objects, local_paths, config, bucket_name, session))


def download_dem_tiles_async(
    tile_objects: list[Path],
    save_folder: Path,
    bucket_name: str = "copernicus-dem-30m",
    config: Config = Config(
        signature_version="",
        region_name="eu-central-1",
        retries={"max_attempts": 3, "mode": "standard"},
    ),
    num_cpu: int = 4,
    num_tasks: int = 4,
    session: aioboto3.Session | None = None,
):
    """Asynchronous download of objects from S3

    Parameters
    ----------
    tile_objects : list[Path]
        List of S3 object paths
    save_folder : Path
        Local folder to save the files
    bucket_name : str, optional
        Name of S3 bucket, by default "copernicus-dem-30m"
    config : Config, optional
        botorcore Config, by default Config( signature_version="", region_name="eu-central-1", retries={"max_attempts": 3, "mode": "standard"}, )
    num_cpu : int, optional
        Number of cpus to be used for multi-processing, by default 4
    num_tasks : int, optional
        Number of tasks to be run in async mode, by default 4
        If num_cps > 1, each task will be assigned to a cpu and will run in async mode on that cpu (multiple threads)
    session : aioboto3.Session | None, optional
        aioboto3.Session, by default None
    """

    if not session:
        session = aioboto3.Session()

    os.makedirs(save_folder, exist_ok=True)
    download_list_chunk = [tile_objects[i::num_tasks] for i in range(num_tasks)]
    if num_cpu == 1:
        for ch in download_list_chunk:
            single_download_process(ch, save_folder, config, bucket_name, session)
    else:
        with mp.Pool(num_cpu) as p:
            p.starmap(
                single_download_process,
                [
                    (ch, save_folder, config, bucket_name, session)
                    for ch in download_list_chunk
                ],
            )


def upload_dem_tiles_async(
    s3_dir: Path,
    local_dir: Path,
    bucket_name: str = "deant-data-public-dev",
    config: Config = Config(
        region_name="ap-southeast-2",
        retries={"max_attempts": 3, "mode": "standard"},
    ),
    num_cpu: int = 4,
    num_tasks: int = 4,
    session: aioboto3.Session | None = None,
):
    """Asynchronous upload of objects to S3

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
    num_cpu : int, optional
        Number of cpus to be used for multi-processing, by default 4
    num_tasks : int, optional
        Number of tasks to be run in async mode, by default 4
        If num_cps > 1, each task will be assigned to a cpu and will run in async mode on that cpu (multiple threads)
    session : aioboto3.Session | None, optional
        aioboto3.Session, by default None
    """

    if not session:
        session = aioboto3.Session()

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

    upload_list_chunk = [tile_objects[i::num_tasks] for i in range(num_tasks)]
    local_list_chunk = [tile_paths[i::num_tasks] for i in range(num_tasks)]
    if num_cpu == 1:
        for ch, ll in zip(upload_list_chunk, local_list_chunk):
            single_upload_process(ch, ll, config, bucket_name, session)
    else:
        with mp.Pool(num_cpu) as p:
            p.starmap(
                single_upload_process,
                [
                    (el[0], el[1], config, bucket_name, session)
                    for el in list(zip(upload_list_chunk, local_list_chunk))
                ],
            )
