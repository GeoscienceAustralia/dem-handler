import os

import rasterio.profiles
import rasterio.session
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import numpy as np

from dem_handler.utils.spatial import BoundingBox

import logging

logger = logging.getLogger(__name__)

EGM_08_URL = (
    "https://aria-geoid.s3.us-west-2.amazonaws.com/us_nga_egm2008_1_4326__agisoft.tif"
)


def download_cop_glo30_tiles(
    tile_filename: str, save_folder: Path, make_folders=True
) -> None:
    """Download a dem tile from AWS and save to specified folder

    Parameters
    ----------
    tile_filename : str
        Copernicus 30m tile filename. e.g. Copernicus_DSM_COG_10_S78_00_E166_00_DEM.tif
    save_folder : Path
        Folder to save the downloaded tif
    make_folders: bool
        Make the save folder if it does not exist
    """
    s3 = boto3.resource(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            region_name="eu-central-1",
        ),
    )
    bucket_name = "copernicus-dem-30m"
    bucket = s3.Bucket(bucket_name)
    s3_path = str(Path(tile_filename).stem / Path(tile_filename))
    save_path = save_folder / Path(tile_filename)
    logger.info(f"Downloading cop30m tile : {s3_path}, save location : {save_path}")

    if make_folders:
        os.makedirs(save_folder, exist_ok=True)

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
        print(
            f"Failed to retrieve data for {os.path.splitext(os.path.basename(json_url))[0]}. Status code: {response.status_code}"
        )
        return ""

    return json_url.replace(".json", "_dem.tif")


def download_rema_tiles(s3_url_list: list[Path], save_folder: Path) -> list[Path]:
    """Downloads rema tiles from AWS S3.

    Parameters
    ----------
    s3_url_list : list[Path]
        List od S3 URLs.
    save_folder : Path
        Local directory to save the files to.

    Returns
    -------
    list[Path]
        List of local paths to the saved files.
    """

    # download individual dems
    dem_paths = []
    for i, s3_file_url in enumerate(s3_url_list):
        # get the raw json url
        dem_url = extract_s3_path(s3_file_url.as_posix())
        if not dem_url:
            continue
        local_path = (
            save_folder / dem_url.split("amazonaws.com")[1][1:]
        )  # extracts the S3 object path of the full url
        local_folder = local_path.parent
        # check if the dem.tif already exists
        if local_path.is_file():
            print(f"{local_path} already exists, skipping download")
            dem_paths.append(local_path)
            continue
        local_folder.mkdir(parents=True, exist_ok=True)
        print(
            f"downloading {i+1} of {len(s3_url_list)}: src: {dem_url} dst: {local_path}"
        )
        urlretrieve(dem_url, local_path)
        dem_paths.append(local_path)

    return dem_paths
