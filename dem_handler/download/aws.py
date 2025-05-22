import os

import rasterio.profiles
import rasterio.session
from rasterio.session import AWSSession
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
import rasterio
from rasterio.windows import from_bounds
from rasterio.mask import mask
from shapely.geometry import box
import shapely
import numpy as np

import xml.etree.ElementTree as ET
import os
import tempfile


from dem_handler.download.aio_aws import bulk_download_dem_tiles
from dem_handler.utils.spatial import BoundingBox

import logging

logger = logging.getLogger(__name__)

EGM_08_URL = (
    "https://aria-geoid.s3.us-west-2.amazonaws.com/us_nga_egm2008_1_4326__agisoft.tif"
)


def download_cop_glo30_tiles(
    tile_filenames: list[Path],
    save_folder: Path | list[Path],
    make_folders=True,
    num_cpus: int = 1,
    num_tasks: int | None = None,
) -> None:
    """Download a dem tile from AWS and save to specified folder

    Parameters
    ----------
    tile_filename : list[Path]
        Copernicus 30m tile filename. e.g. Copernicus_DSM_COG_10_S78_00_E166_00_DEM.tif
    save_folder : Path | list[Path]
        Folder(s) to save the downloaded tifs. If using async mode (i.e. num_tasks is not None), save folder should be a single path.
    make_folders: bool
        Make the save folder if it does not exist
    """
    config = Config(
        signature_version="",
        region_name="eu-central-1",
        retries={"max_attempts": 3, "mode": "standard"},
    )
    bucket_name = "copernicus-dem-30m"

    if num_tasks:
        assert (
            type(save_folder) is not list
        ), "Save folder should be a single path in async mode."
        tile_objects = [tn.stem / tn for tn in tile_filenames]
        bulk_download_dem_tiles(
            tile_objects, save_folder, bucket_name, config, num_cpus, num_tasks
        )
    else:
        config.signature_version = UNSIGNED
        s3 = boto3.resource(
            "s3",
            config=config,
        )
        bucket = s3.Bucket(bucket_name)
        for i, tile_filename in enumerate(tile_filenames):
            s3_path = (Path(tile_filename).stem / Path(tile_filename)).as_posix()
            save_path = save_folder[i] / Path(tile_filename)
            logger.info(
                f"Downloading cop30m tile : {s3_path}, save location : {save_path}"
            )

            if make_folders:
                os.makedirs(save_folder[i], exist_ok=True)

            try:
                bucket.download_file(s3_path, save_path)
            except Exception as e:
                raise (e)


def download_egm_08_geoid(
    save_path: Path, bounds: BoundingBox, geoid_url: str = EGM_08_URL
):
    """Download the egm_2008 geoid for AWS for the specified bounds.

    Parameters
    ----------
    save_path : Path
        Where to save tif. e.g. my/geoid/folder/geoid.tif
    bounds : BoundingBox
        Bounding box to download data
    geoid_url : str, optional
        URL, by default EGM_08_URL=
        https://aria-geoid.s3.us-west-2.amazonaws.com/us_nga_egm2008_1_4326__agisoft.tif

    Returns
    -------
    tuple(np.array, dict)
        geoid array and geoid rasterio profile
    """

    logger.info(f"Downloading egm_08 geoid for bounds {bounds} from {geoid_url}")

    if bounds is None:
        with rasterio.open(geoid_url) as ds:
            geoid_arr = ds.read()
            geoid_profile = ds.profile

    else:
        with rasterio.open(geoid_url) as ds:
            geom = [box(*bounds)]

            # Clip the raster to the bounding box
            geoid_arr, clipped_transform = mask(ds, geom, crop=True, all_touched=True)
            geoid_profile = ds.profile.copy()
            geoid_profile.update(
                {
                    "height": geoid_arr.shape[1],  # Rows
                    "width": geoid_arr.shape[2],  # Columns
                    "transform": clipped_transform,
                }
            )

    # Transform nodata to nan
    geoid_arr = geoid_arr.astype("float32")
    geoid_arr[geoid_profile["nodata"] == geoid_arr] = np.nan
    geoid_profile["nodata"] = np.nan

    # Write to file
    with rasterio.open(save_path, "w", **geoid_profile) as dst:
        dst.write(geoid_arr)

    return geoid_arr, geoid_profile


import os
import requests
from urllib.request import urlretrieve


def find_files(folder, contains):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            if contains in name:
                filename = os.path.join(root, name)
                paths.append(filename)
    return paths


def extract_s3_path(url: str) -> str:
    """Extracts AWS S3 path from a long URL

    Parameters
    ----------
    url : Path
        URL containing the S3 path.

    Returns
    -------
    Path
        Extracted S3 path.
    """
    json_url = f'https://{url.split("external/")[-1]}'
    # Make a GET request to fetch the raw JSON content
    response = requests.get(json_url)
    # Check if the request was successful
    if response.status_code != 200:
        # Parse JSON content into a Python dictionary
        logger.info(
            f"Failed to retrieve data for {os.path.splitext(os.path.basename(json_url))[0]}. Status code: {response.status_code}"
        )
        return ""

    return json_url.replace(".json", "_dem.tif")


def download_rema_tiles(
    s3_url_list: list[Path],
    save_folder: Path,
    num_cpus: int = 1,
    num_tasks: int | None = None,
) -> list[Path]:
    """Downloads rema tiles from AWS S3.

    Parameters
    ----------
    s3_url_list : list[Path]
        List od S3 URLs.
    save_folder : Path
        Local directory to save the files to.
    num_cpus : int, optional
        Number of cpus to be used for parallel download, by default 1.
        Setting to -1 will use all available cpus
    num_tasks : int | None, optional
        Number of tasks to be run in async mode, by default None which does not use async or parallel downloads
        If num_cpus > 1, each task will be assigned to a cpu and will run in async mode on that cpu (multiple threads).
        Setting to -1 will transfer all tiles in one task.

    Returns
    -------
    list[Path]
        List of local paths to the saved files.
    """

    REMA_BUCKET_NAME = "pgc-opendata-dems"
    REMA_REGION = "us-west-2"
    REMA_CONFIG = Config(
        region_name=REMA_REGION,
        retries={"max_attempts": 3, "mode": "standard"},
    )

    # download individual dems
    dem_urls = [extract_s3_path(url.as_posix()) for url in s3_url_list]

    if num_tasks:
        tile_objects = [Path(*Path(url).parts[2:]) for url in dem_urls]
        dem_paths = bulk_download_dem_tiles(
            tile_objects,
            save_folder,
            REMA_BUCKET_NAME,
            REMA_CONFIG,
            num_cpus,
            num_tasks,
            None,
        )
    else:
        dem_paths = []
        for i, dem_url in enumerate(dem_urls):
            # get the raw json url
            if not dem_url:
                continue
            local_path = (
                save_folder / dem_url.split("amazonaws.com")[1][1:]
            )  # extracts the S3 object path of the full url
            local_folder = local_path.parent
            # check if the dem.tif already exists
            if local_path.is_file():
                logger.info(f"{local_path} already exists, skipping download")
                dem_paths.append(local_path)
                continue
            local_folder.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"downloading {i+1} of {len(dem_urls)}: src: {dem_url} dst: {local_path}"
            )
            urlretrieve(dem_url, local_path)
            dem_paths.append(local_path)

    return dem_paths


def read_and_process_vrt(local_vrt_path, bounds, save_path=None):

    with rasterio.open(local_vrt_path) as src:
        print(src.profile)
        dem_array, dem_transform = rasterio.mask.mask(
            src,
            [shapely.geometry.box(*bounds.bounds)],
            all_touched=True,
            crop=True,
        )
        # Using the masking adds an extra dimension from the read
        # Remove this by squeezing before writing
        dem_array = dem_array.squeeze()

        # replace nodata with np.nan
        dem_array[dem_array == src.nodata] = np.nan

        dem_profile = src.profile
        dem_profile.update(
            {
                "driver": "GTiff",
                "height": dem_array.shape[0],
                "width": dem_array.shape[1],
                "transform": dem_transform,
                "count": 1,
                "nodata": np.nan,
            }
        )

        if save_path:
            with rasterio.open(save_path, "w", **dem_profile) as dst:
                dst.write(dem_array, 1)

    return dem_array, dem_profile


def read_rema_timeseries_vrt(
    year,
    bounds,
    save_path,
    local_vrt_folder="/home/rtc_user/working/rema-timeseries-tiles/mosaics/thwaites_local",
    cloud=False,
    aws_access_key_id=None,
    aws_secret_access_key=None,
):

    local_vrt_path = save_path.with_suffix(".vrt")

    if cloud:
        logging.info(f"Reading DEM .vrt from remote datastore")
        if not aws_access_key_id:
            if os.environ.get("REMA_AWS_ACCESS_KEY_ID") is None:
                wrn_msg = "REMA_AWS_ACCESS_KEY_ID is not set in environment variables."
                raise ValueError(wrn_msg)
            else:
                aws_access_key_id = os.environ.get("REMA_AWS_ACCESS_KEY_ID")
        if not aws_secret_access_key:
            if os.environ.get("REMA_AWS_SECRET_ACCESS_KEY") is None:
                wrn_msg = (
                    "REMA_AWS_SECRET_ACCESS_KEY is not set in environment variables."
                )
                raise ValueError(wrn_msg)
            else:
                aws_secret_access_key = os.environ.get("REMA_AWS_SECRET_ACCESS_KEY")

        # S3-compatible endpoint
        endpoint = "umn1.osn.mghpcc.org"
        endpoint_url = "https://umn1.osn.mghpcc.org"

        # File path inside the bucket
        s3_bucket = "cse-pgc-test"
        vrt_path = f"digital_earth_antarctica/v0.1/mosaics/thwaites_remote/thwaites_remote_z_{year}.vrt"
        logging.info(f"Downloading .vrt from : {endpoint_url}/{vrt_path}")

        # Start boto3 session
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        # download and edit the vrt as the remote vrt references browse images and not the data
        s3 = session.client("s3", endpoint_url=endpoint_url)
        response = s3.get_object(Bucket=s3_bucket, Key=vrt_path)
        vrt_content = response["Body"].read()

        with open(local_vrt_path, "w") as temp_vrt:
            temp_vrt.write(vrt_content.decode("utf-8"))

    else:
        logging.info(f"Checking local folder for REMA tmeseries vrt: {local_vrt_folder}")
        local_vrt_path = f"{local_vrt_folder}/thwaites_local_z_{year}.vrt"
        logging.info(f"Reading DEM .vrt from local path: {local_vrt_path}")

    # GDAL config
    if cloud:
        local_vrt_folder
        with rasterio.Env(
            AWSSession(session),
            AWS_S3_ENDPOINT=endpoint,
            AWS_VIRTUAL_HOSTING=False,  # critical for non-AWS S3 providers
        ):
            dem_array, dem_profile = read_and_process_vrt(
                local_vrt_path, bounds, save_path
            )
    else:
        dem_array, dem_profile = read_and_process_vrt(local_vrt_path, bounds, save_path)

    # Optional: Clean up
    # os.remove(local_vrt_path)

    return dem_array, dem_profile
