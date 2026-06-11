from __future__ import annotations
from pathlib import Path
import shapely
from shapely import box
import geopandas as gpd
import rasterio
from rasterio.profiles import Profile
import numpy as np
import math
import logging

from dem_handler.utils.spatial import (
    BoundingBox,
    transform_polygon,
    crop_datasets_to_bounds,
)
from dem_handler.utils.general import log_timing
from dem_handler.download.aws import download_rema_tiles, extract_s3_path

from dem_handler.dem.geoid import apply_geoid
from dem_handler.download.aws import download_egm_08_geoid

# Create a custom type that allows use of BoundingBox or tuple(left, bottom, right, top)
BBox = BoundingBox | tuple[float | int, float | int, float | int, float | int]

from dem_handler import REMA_GPKG_PATH, REMA_VALID_RESOLUTIONS, REMAResolutions


@log_timing
def get_rema_dem_for_bounds(
    bounds: BBox,
    rema_year: int | None = None,
    save_path: Path | str = "",
    rema_index_path: Path | str = REMA_GPKG_PATH,
    local_dem_dir: Path | str | None = None,
    resolution: REMAResolutions = 2,
    bounds_src_crs: int = 3031,
    buffer_metres: int = 0,
    buffer_pixels: int = 0,
    ellipsoid_heights: bool = True,
    geoid_tif_path: Path | str = "egm_08_geoid.tif",
    download_geoid: bool = False,
    num_cpus: int = 1,
    num_tasks: int | None = None,
    return_paths: bool = False,
    download_dir: Path | str = "rema_dems_temp_folder",
) -> tuple[np.ndarray, Profile | list[Path]] | list[Path] | tuple[None, None, None]:
    """Finds the REMA DEM tiles in a given bounding box and merges them into a single tile.

    Parameters
    ----------
    bounds : BBox
        BoundingBox object or tuple of coordinates
    rema_year : int | None
        The year for the DEM if the timeseries product is required. If None, the static
        REMA dem product is used.
    save_path : Path | str, optional
        Local path to save the output tile, by default ""
    rema_index_path : Path | str, optional
        Path to the index files with the list of REMA tiles in it, by default REMA_GPKG_PATH
    local_dem_dir: Path | str | None, optional
        Path to existing local DEM directory, by default None
    resolution : int, optional
        Resolution of the required tiles, by default 2
    bounds_src_crs : int, optional
        CRS of the provided bounding box, by default 3031
    buffer_metres: int, optional
        buffer to add to the dem in metres.
    buffer_pixels: int, optional
        buffer to add to the dem in pixels.
    ellipsoid_heights : bool, optional
        Subtracts the geoid height from the tiles to get the ellipsoid height, by default True
    geoid_tif_path : Path | str, optional
        Path to the existing ellipsoid file, by default "egm_08_geoid.tif"
    download_geoid : bool, optional
        Flag to download the ellipsoid file, by default False
    num_cpus : int, optional
        Number of cpus to be used for parallel download, by default 1.
        Setting to -1 will use all available cpus
    num_tasks : int | None, optional
        Number of tasks to be run in async mode, by default None which does not use async or parallel downloads
        If num_cpus > 1, each task will be assigned to a cpu and will run in async mode on that cpu (multiple threads).
        Setting to -1 will transfer all tiles in one task.
    return_paths: bool, optional
        Flag to return the local paths for downloaded DEMs only, by default False
    download_dir: Path | str , optional
        Directory to download the REMA DEMs to, by default "rema_dems_temp_folder"
    Returns
    -------
    tuple[np.ndarray, Profile | list[Path]] | list[Path]
        Tuple of the output tile array and its profile, and file paths. Only file paths if `return_paths` is true.

    Raises
    ------
    FileExistsError
        If `ellipsoid_heights` is True, it will raise an error if the ellipsoid file does not exist and `download_geoid` is set to False.
    """

    GEOID_CRS = 4326
    REMA_CRS = 3031

    assert (
        resolution in REMA_VALID_RESOLUTIONS
    ), f"resolution must be in {REMA_VALID_RESOLUTIONS}"

    if type(bounds) != BoundingBox:
        bounds = BoundingBox(*bounds)

    if rema_year:
        logging.warning(
            f"The REMA timeseries product will be used for year : {rema_year}. "
            "This may have increased artifacts compared to the static product. "
            "Note, an error will be raised if a timeseries dem does not exist for the requested year."
        )

    # convert paths from str to path type for ease of handling
    save_path = Path(save_path) if isinstance(save_path, str) else save_path
    rema_index_path = (
        Path(rema_index_path) if isinstance(rema_index_path, str) else rema_index_path
    )
    local_dem_dir = (
        Path(local_dem_dir) if isinstance(local_dem_dir, str) else local_dem_dir
    )
    geoid_tif_path = (
        Path(geoid_tif_path) if isinstance(geoid_tif_path, str) else geoid_tif_path
    )
    download_dir = Path(download_dir) if isinstance(download_dir, str) else download_dir

    # Log the requested bounds
    logging.info(f"Getting REMA DEM for bounds: {bounds.bounds}")

    if bounds_src_crs != REMA_CRS:
        logging.warning(
            f"Transforming bounds from {bounds_src_crs} to {REMA_CRS}. This may return data beyond the requested bounds. If this is not desired, provide the bounds in EPSG:{REMA_CRS}."
        )
        bounds = BoundingBox(
            *transform_polygon(box(*bounds.bounds), bounds_src_crs, REMA_CRS).bounds
        )
        bounds_src_crs = REMA_CRS
        logging.info(f"Adjusted bounds in 3031 : {bounds}")

    bounds_poly = box(*bounds.bounds)

    # buffer in 3031 based on user input
    if buffer_metres and buffer_pixels:
        logging.warning(
            "Both pixel and metre buffer provided. Metre buffer will be used."
        )
        buffer_pixels = None
    if buffer_pixels:
        logging.info(
            f"Buffering bounds by {buffer_pixels} pixels, converting buffer to metres first."
        )
        # convert from pixels to metres
        buffer_metres = buffer_pixels * resolution
    if buffer_metres:
        logging.info(f"Buffering bounds by {buffer_metres} metres.")
        bounds = BoundingBox(*bounds_poly.buffer(buffer_metres).bounds)
        bounds_poly = box(*bounds.bounds)
        logging.info(f"Buffered bounds : {bounds}")

    if not rema_year:

        rema_layer = f"REMA_Mosaic_Index_v2_{resolution}m"
        rema_index_df = gpd.read_file(rema_index_path, layer=rema_layer)

        intersecting_rema_files = rema_index_df[
            rema_index_df.geometry.intersects(bounds_poly)
        ]
        if len(intersecting_rema_files.s3url) == 0:
            logging.info("No REMA tiles found for this bounding box")
            return None, None, None
        logging.info(f"{len(intersecting_rema_files.s3url)} intersecting tiles found")

        s3_url_list = [Path(url) for url in intersecting_rema_files["s3url"].to_list()]
        raster_paths = []
        if local_dem_dir:
            raster_paths = list(local_dem_dir.rglob("*.tif"))
            raster_names = [r.stem.replace("_dem", "") for r in raster_paths]
            s3_url_list = [url for url in s3_url_list if url.stem not in raster_names]

        if return_paths:
            if num_tasks:
                raster_paths.extend(
                    [
                        download_dir / u.name.replace(".json", "_dem.tif")
                        for u in s3_url_list
                    ]
                )
            else:
                dem_urls = [extract_s3_path(url.as_posix()) for url in s3_url_list]
                raster_paths.extend(
                    [
                        download_dir / dem_url.split("amazonaws.com")[1][1:]
                        for dem_url in dem_urls
                    ]
                )
            return raster_paths

        raster_paths.extend(
            download_rema_tiles(s3_url_list, download_dir, num_cpus, num_tasks)
        )

        # adjust the bounds to include whole dem pixels and stop offsets being introduced
        logging.info("Adjusting bounds to include whole dem pixels")
        bounds = adjust_bounds_for_rema_profile(bounds, raster_paths)
        logging.info(f"Adjusted bounds : {bounds}")

        logging.info("combining found DEMs")
        dem_array, dem_profile = crop_datasets_to_bounds(
            raster_paths, bounds, save_path
        )

    else:
        # Read in the rema timeseries dem from appropriate vrt
        logging.info("Reading in REMA timeseries .vrt")
        dem_array, dem_profile = read_rema_timeseries_vrt(
            year=rema_year,
            bounds=bounds,
            save_path=save_path,
            resolution=resolution,
        )
        raster_paths = []

    # return the dem in either ellipsoid or geoid referenced heights. The REMA dem is already
    # referenced to the ellipsoid. Values are set to zero where no tile data exists
    # create a mask for this area so we can apply the geoid here to convert to ellipsoid heights
    dem_novalues_mask = dem_array == 0
    dem_novalues_count = np.count_nonzero(dem_novalues_mask)
    dem_values_mask = dem_array != 0

    if dem_novalues_count == 0 and ellipsoid_heights:
        # we have data everywhere and the values are already ellipsoid referenced
        if save_path:
            logging.info(f"DEM saved to : {save_path}")
        logging.info(f"Dem array shape = {dem_array.shape}")
        return dem_array, dem_profile, raster_paths
    else:
        geoid_bounds = bounds
        if bounds_src_crs != GEOID_CRS:
            geoid_bounds = transform_polygon(
                box(*bounds.bounds), bounds_src_crs, GEOID_CRS
            ).bounds
        if not download_geoid and not geoid_tif_path.exists():
            raise FileExistsError(
                f"Geoid file does not exist: {geoid_tif_path}. "
                "correct path or set download_geoid = True"
            )
        elif download_geoid and not geoid_tif_path.exists():
            logging.info(f"Downloading the egm_08 geoid")
            download_egm_08_geoid(geoid_tif_path, bounds=geoid_bounds)
        elif download_geoid and geoid_tif_path.exists():
            # Check that the existing geiod covers the dem
            with rasterio.open(geoid_tif_path) as src:
                existing_geoid_bounds = shapely.geometry.box(*src.bounds)
            if existing_geoid_bounds.covers(shapely.geometry.box(*bounds.bounds)):
                logging.info(
                    f"Skipping geoid download. The existing geoid file covers the DEM bounds. Existing geoid file: {geoid_tif_path}."
                )
            else:
                logging.info(
                    f"The existing geoid file does not cover the DEM bounds. A new geoid file covering the bounds will be downloaded, overwriting the existing geiod file: {geoid_tif_path}."
                )
                download_egm_08_geoid(geoid_tif_path, bounds=geoid_bounds)

        logging.info(f"Using geoid file: {geoid_tif_path}")

    if ellipsoid_heights:
        logging.info(f"Returning DEM referenced to ellipsoidal heights")
        dem_array = apply_geoid(
            dem_array=dem_array,
            dem_profile=dem_profile,
            geoid_path=geoid_tif_path,
            buffer_pixels=2,
            save_path=save_path,
            mask_array=dem_novalues_mask,
            method="add",
        )
    else:
        # heights are not referenced to the geoid, therefore we must
        # convert ellipsoid referenced heights to geoid referenced heights as the
        # rema dem is by default referenced to the ellipsoid. We do this only in
        # areas with data, leaving the nodata areas at zero values
        logging.info(f"Returning DEM referenced to geoid heights")
        dem_array = apply_geoid(
            dem_array=dem_array,
            dem_profile=dem_profile,
            geoid_path=geoid_tif_path,
            buffer_pixels=2,
            save_path=save_path,
            mask_array=dem_values_mask,
            method="subtract",
        )

    if save_path:
        logging.info(f"DEM saved to : {save_path}")
    logging.info(f"Dem array shape = {dem_array.shape}")
    return dem_array, dem_profile, raster_paths


def adjust_bounds_for_rema_profile(bounds: BBox, raster_paths: list[str]):
    """
    Adjust the bounds to consider whole pixels of the source dem.
    If a fraction of a pixel is requested, it can cause small offsets
    in the mosaicked dem. This ensures the origin of the mosaic aligns
    with the source dem tiles.

    Parameters
    ----------
    bounds : BBox
        Requested bounding box in 3031
    raster_paths : list(str)
        List of paths to intersecting rema tiles

    Returns
    -------
    BoundingBox
        Original bounding box adjusted to include full rema dem pixels

    """
    if type(bounds) == BoundingBox:
        bounds = bounds.bounds

    sample_dem_tile_path = raster_paths[0]
    with rasterio.open(sample_dem_tile_path) as src:
        transform = src.transform

    x_origin = transform.c
    y_origin = transform.f
    resolution = abs(transform.a)

    # adjust the bounds to be nearest pixel multiples of the origin
    def _round_down_to_step(coord, origin, res):
        n = math.floor((coord - origin) / res)
        return origin + n * res

    def _round_up_to_step(coord, origin, res):
        n = math.ceil((coord - origin) / res)
        return origin + n * res

    xmin = _round_down_to_step(bounds[0], x_origin, resolution)
    ymin = _round_down_to_step(bounds[1], y_origin, resolution)
    xmax = _round_up_to_step(bounds[2], x_origin, resolution)
    ymax = _round_up_to_step(bounds[3], y_origin, resolution)
    adjusted_bounds = (xmin, ymin, xmax, ymax)

    return BoundingBox(*adjusted_bounds)


import os
import re
import tempfile
import boto3
from rasterio.session import AWSSession
from dem_handler.utils.raster import read_raster_from_vrt


def read_rema_timeseries_vrt(
    year: int,
    bounds: BBox,
    save_path: str,
    resolution: int,
    aoi_name: (
        str | None
    ) = "thwaites",  # Reading data using indexing file doesn't need the aoi name.
    endpoint: str = "umn1.osn.mghpcc.org",
    s3_bucket: str = "cse-pgc-test",
    remote_folder: str = "digital_earth_antarctica/v0.4/mosaics",  # or "digital_earth_antarctica/v0.5" for version 0.5
    indexing_file: (
        str | None
    ) = None,  # or "MultiTemporalREMAIndex.parquet" for version 0.5
):
    """Reads the REMA timeseries .vrt or a provided indexing file for a given year and bounds, merging the intersecting tiles into one raster.

    Parameters
    ----------
    year : int
        The year for the DEM if the timeseries product is required.
    bounds : BBox
        Bounding box for the area of interest in 3031.
    save_path : str
        Local path to save the output tile.
    resolution : int
        Resolution of the required tiles, for example 10.
    aoi_name : str | None, optional
        Name of the area of interest, by default "thwaites". This is only used
        when reading data without an indexing file.
    endpoint : str, optional
        Endpoint for the s3 datastore, by default "umn1.osn.mghpcc.org"
    s3_bucket : str, optional
        S3 bucket for the datastore, by default "cse-pgc-test"
    remote_folder : str, optional
        Remote folder in the s3 bucket where the data is stored, by default "digital_earth_antarctica/v0.4/mosaics"
    indexing_file : str | None, optional
        Name of the indexing file in the remote datastore.
        If None, it will attempt to read directly from the vrt, which is the case for version 0.4 of the REMA timeseries product

    Returns
    -------
    tuple[np.ndarray, Profile]
        Tuple of the output tile array and its profile
    """

    logging.info(f"Reading DEM .vrt from remote datastore")
    if not os.environ.get("REMA_AWS_ACCESS_KEY_ID"):
        raise ValueError("REMA_AWS_ACCESS_KEY_ID is not set in environment variables.")
    else:
        aws_access_key_id = os.environ.get("REMA_AWS_ACCESS_KEY_ID")
    if not os.environ.get("REMA_AWS_SECRET_ACCESS_KEY"):
        raise ValueError(
            "REMA_AWS_SECRET_ACCESS_KEY is not set in environment variables."
        )
    else:
        aws_secret_access_key = os.environ.get("REMA_AWS_SECRET_ACCESS_KEY")

    # Start boto3 session
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    endpoint_url = f"https://{endpoint}"

    if indexing_file:
        indexing_file_prefix = f"{remote_folder}/{indexing_file}"
        print(f"Looking for indexing file at s3://{s3_bucket}/{indexing_file_prefix}")
        storage_options = {
            "key": aws_access_key_id,
            "secret": aws_secret_access_key,
            "client_kwargs": {"endpoint_url": endpoint_url},
        }
        indexing_file_path = f"s3://{s3_bucket}/{indexing_file_prefix}"
        indexing_gdf = gpd.read_parquet(
            indexing_file_path, storage_options=storage_options
        )
        indexing_gdf = indexing_gdf[indexing_gdf.year == str(year)]
        bounds_poly = box(*bounds)
        indexing_gdf = indexing_gdf[indexing_gdf.intersects(bounds_poly)]
        indexing_gdf = indexing_gdf[
            indexing_gdf.dem_url.apply(lambda x: os.path.basename(x).split("_")[2])
            == str(resolution) + "m"
        ]
        dem_urls = indexing_gdf.dem_url.to_list()

        if len(dem_urls) == 0:
            raise ValueError(
                f"No DEM tiles found for the requested year {year}, "
                f"resolution {resolution}m, and bounds {bounds}. "
                f"Please check the indexing file and ensure there are tiles that intersect the requested bounds."
            )

        with rasterio.Env(
            AWSSession(session),
            AWS_S3_ENDPOINT=endpoint,
            AWS_VIRTUAL_HOSTING=False,  # critical for non-AWS S3 providers
        ):
            remote_datasets = []
            for url in dem_urls:
                logging.info(f"Reading raster for bounds from remote vrt : {url}")
                dataset = rasterio.open(url)
                remote_datasets.append(dataset)

            dem_array, dem_transform = rasterio.merge.merge(
                remote_datasets, bounds=bounds
            )
            dem_profile = remote_datasets[0].profile.copy()
            dem_array[dem_array == remote_datasets[0].nodata] = np.nan
            dem_array = np.squeeze(dem_array)  # remove extra dimension added by merge
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
            with rasterio.open(save_path, "w", **dem_profile) as dst:
                dst.write(dem_array, 1)
    else:
        # download and edit the vrt as the remote vrt references browse images and not the data
        s3 = session.client("s3", endpoint_url=endpoint_url)

        # check to ensure the year is there
        vrt_pattern = re.compile(rf"{aoi_name}_.*?{resolution}m_dem_(\d+)\.vrt$")
        # List all objects under prefix
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=s3_bucket, Prefix=remote_folder)
        valid_years = set()
        valid_files = {}

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                match = vrt_pattern.search(key)
                if match:
                    year_found = int(match.group(1))
                    valid_years.add(year_found)
                    valid_files[year_found] = key

        years_list = sorted(valid_years)

        if year not in valid_years:
            raise ValueError(
                f'The requested year "{year}" does not have a rema timeseries. '
                f"valid years are : {years_list}"
            )

        vrt_path = valid_files[year]
        logging.info(f"Downloading .vrt from : {endpoint_url}/{vrt_path}")
        response = s3.get_object(Bucket=s3_bucket, Key=vrt_path)
        vrt_content = response["Body"].read()

        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(vrt_content.decode("utf-8"))
            temp_vrt = f.name

        logging.info("Adjusting rema bounds to stop small offsets")
        logging.info(f"original bounds : {bounds}")
        bounds = adjust_bounds_for_rema_profile(bounds, [temp_vrt])
        logging.info(f"adjusted bounds : {bounds}")

        with rasterio.Env(
            AWSSession(session),
            AWS_S3_ENDPOINT=endpoint,
            AWS_VIRTUAL_HOSTING=False,  # critical for non-AWS S3 providers
        ):
            logging.info(f"Reading raster for bounds from remote vrt")
            dem_array, dem_profile = read_raster_from_vrt(temp_vrt, bounds, save_path)

    return dem_array, dem_profile
