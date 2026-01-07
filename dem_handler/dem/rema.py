from __future__ import annotations
from pathlib import Path
import shapely
from shapely import box
import geopandas as gpd
import rasterio
from rasterio.profiles import Profile
from rasterio.crs import CRS
import numpy as np
import math
import logging
from affine import Affine

from dem_handler.utils.spatial import (
    BoundingBox,
    transform_polygon,
    crop_datasets_to_bounds,
)
from dem_handler.utils.general import log_timing
from dem_handler.download.aws import download_rema_tiles, extract_s3_path

from dem_handler.dem.geoid import apply_geoid
from dem_handler.download.aws import download_egm_08_geoid
from dem_handler.utils.spatial import (
    check_bounds_likely_cross_antimeridian,
    split_antimeridian_shape_into_east_west_bounds,
)
from dem_handler.utils.raster import reproject_and_merge_rasters

# Create a custom type that allows use of BoundingBox or tuple(left, bottom, right, top)
BBox = BoundingBox | tuple[float | int, float | int, float | int, float | int]

from dem_handler import REMA_GPKG_PATH, REMA_VALID_RESOLUTIONS, REMAResolutions


@log_timing
def get_rema_dem_for_bounds(
    bounds: BBox,
    save_path: Path | str = "",
    rema_index_path: Path | str = REMA_GPKG_PATH,
    local_dem_dir: Path | str | None = None,
    resolution: REMAResolutions = 2,
    bounds_src_crs: int = 3031,
    buffer_metres: int = 0,
    buffer_pixels: int = 0,
    ellipsoid_heights: bool = True,
    return_over_ocean: bool = True,
    geoid_tif_path: Path | str = "egm_08_geoid.tif",
    download_geoid: bool = False,
    check_geoid_crosses_antimeridian: bool = True,
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
    return_over_ocean: bool, optional
        If no tile intersections are found with the dem, return elevation heights. i.e.
        ellipsoid height if ellipsoid_heights = True, else DEM of zero values.
    geoid_tif_path : Path | str, optional
        Path to the existing ellipsoid file, by default "egm_08_geoid.tif"
    download_geoid : bool, optional
        Flag to download the ellipsoid file, by default False
    check_geoid_crosses_antimeridian : bool, optional
        Check if the geoid crosses the antimeridian. Set to False if it is known
        The requested bounds do not cross the antimeridian to stop false positives.
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

    # Check if bounds cross the antimeridian
    if bounds.left > bounds.right:
        logging.warning(
            f"left longitude value ({bounds[0]}) is greater than the right longitude value {({bounds[2]})} "
            f"for the bounds provided. Assuming the bounds cross the antimeridian : {bounds}"
        )
        dem_crosses_antimeridian = True
    else:
        dem_crosses_antimeridian = False
        # run a basic to check if the bounds likely cross the antimeridian but
        # are just formatted wrong. If so, warn the user.
        if check_bounds_likely_cross_antimeridian(bounds):
            logging.warning(
                "Provided bounds have very large longitude extent. If the shape crosses the "
                f"antimeridian, reformat the bounds as : ({bounds[2]}, {bounds[1]}, {bounds[0]}, {bounds[3]}). "
                "For large areas, provide the inputs bounds in 3031 to avoid transform errors between "
                "coordinate systems."
            )

    if bounds_src_crs != REMA_CRS:
        logging.warning(
            f"Transforming bounds from {bounds_src_crs} to {REMA_CRS}. This may return data beyond the requested bounds. If this is not desired, provide the bounds in EPSG:{REMA_CRS}."
        )
        bounds = BoundingBox(
            *transform_polygon(box(*bounds.bounds), bounds_src_crs, REMA_CRS).bounds
        )
        logging.info(f"Adjusted bounds in 3031 : {bounds}")

    # bounds crs is now 3031
    bounds_crs = REMA_CRS
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

    rema_layer = f"REMA_Mosaic_Index_v2_{resolution}m"
    rema_index_df = gpd.read_file(rema_index_path, layer=rema_layer)

    # adjust the bounds to include whole dem pixels and stop offsets being introduced
    logging.info("Adjusting bounds to include whole dem pixels")
    sample_rema_tile = extract_s3_path(rema_index_df.iloc[0].s3url)
    bounds = adjust_bounds_for_rema_profile(bounds, [sample_rema_tile])
    logging.info(f"Adjusted bounds : {bounds}")

    intersecting_rema_files = rema_index_df[
        rema_index_df.geometry.intersects(bounds_poly)
    ]
    s3_url_list = [Path(url) for url in intersecting_rema_files["s3url"].to_list()]
    raster_paths = []  # list to store paths to rasters

    if len(intersecting_rema_files.s3url) == 0:
        logging.info("No REMA tiles found for this bounding box")
        if not return_over_ocean:
            logging.info(
                "Exiting process. To return zero valued DEM or ellipsoid heights, set return_over_ocean=True."
            )
            return None, None, None
        else:
            logging.info("Returning sea-level DEM")
            # Generate a zero's out DEM. If ellipsoid heights
            # Are required, zero values will be replaced with logic below
            dem_profile = make_empty_rema_profile_for_bounds(bounds, resolution)
            dem_array = 0 * np.ones((dem_profile["height"], dem_profile["width"]))

    else:
        logging.info(f"{len(intersecting_rema_files.s3url)} intersecting tiles found")
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

        logging.info("combining found DEMs")
        dem_array, dem_profile = crop_datasets_to_bounds(
            raster_paths, bounds, save_path
        )

    # return the dem in either ellipsoid or geoid referenced heights. The REMA dem is already
    # referenced to the ellipsoid. Values are set to zero where no tile data exists
    # create a mask for this area so we can apply the geoid here to convert to ellipsoid heights
    dem_novalues_mask = dem_array == 0
    dem_novalues_count = np.count_nonzero(dem_novalues_mask)
    dem_values_mask = dem_array != 0

    if dem_novalues_count == 0 and ellipsoid_heights:
        # we have data everywhere and the values are already ellipsoid referenced
        logging.info(f"All DEM values already referenced to ellipsoid")
        logging.info(f"Returning DEM referenced to ellipsoidal heights")
        if save_path:
            logging.info(f"DEM saved to : {save_path}")
        logging.info(f"Dem array shape = {dem_array.shape}")
        return dem_array, dem_profile, raster_paths
    else:
        # we need to apply the geoid to the DEM
        if bounds_crs == GEOID_CRS:
            # bounds already in correct CRS (4326 for egm_08)
            geoid_bounds = bounds
            geoid_poly = box(*bounds.bounds)
        else:
            geoid_poly = transform_polygon(box(*bounds.bounds), bounds_crs, GEOID_CRS)
            geoid_bounds = geoid_poly.bounds

        # check if the bounds likely the antimeridian
        if check_geoid_crosses_antimeridian:
            logging.info(
                "Checking if geoid likely crosses the antimeridian. "
                "If this is not desired, set check_geoid_crosses_antimeridian = False"
            )
            geoid_crosses_antimeridian = check_bounds_likely_cross_antimeridian(
                geoid_bounds
            )
        else:
            geoid_crosses_antimeridian = False
        if geoid_crosses_antimeridian or dem_crosses_antimeridian:
            logging.info(
                "Geoid crosses antimeridian, splitting into east and west hemispheres"
            )
            # separate the geoid into east and west sections
            east_geoid_tif_path = geoid_tif_path.with_stem(
                geoid_tif_path.stem + "_eastern"
            )
            west_geoid_tif_path = geoid_tif_path.with_stem(
                geoid_tif_path.stem + "_western"
            )
            # construct the bounds for east and west hemisphere
            east_geoid_bounds, west_geoid_bounds = (
                split_antimeridian_shape_into_east_west_bounds(
                    geoid_poly, buffer_degrees=0.1
                )
            )
            logging.info(f"East geoid bounds : {east_geoid_bounds.bounds}")
            logging.info(f"West geoid bounds : {west_geoid_bounds.bounds}")
            geoid_tifs_to_apply = [east_geoid_tif_path, west_geoid_tif_path]
            geoid_bounds_to_apply = [
                east_geoid_bounds.bounds,
                west_geoid_bounds.bounds,
            ]
        else:
            geoid_tifs_to_apply = [geoid_tif_path]
            geoid_bounds_to_apply = [geoid_bounds]  # bounds already tuple

        if not download_geoid and not all(p.exists() for p in geoid_tifs_to_apply):
            raise FileExistsError(
                f"Required geoid files do not exist: {geoid_tifs_to_apply}. "
                "correct path or set download_geoid = True"
            )
        elif download_geoid and not all(p.exists() for p in geoid_tifs_to_apply):
            logging.info(f"Downloading the egm_08 geoid")
            for geoid_path, geoid_bounds in zip(
                geoid_tifs_to_apply, geoid_bounds_to_apply
            ):
                download_egm_08_geoid(geoid_path, bounds=geoid_bounds)
        elif download_geoid and all(p.exists() for p in geoid_tifs_to_apply):
            # Check that the existing geoid covers the dem
            for geoid_path, geoid_bounds in zip(
                geoid_tifs_to_apply, geoid_bounds_to_apply
            ):
                with rasterio.open(geoid_path) as src:
                    existing_geoid_bounds = shapely.geometry.box(*src.bounds)
                if existing_geoid_bounds.covers(shapely.geometry.box(*bounds.bounds)):
                    logging.info(
                        f"Skipping geoid download. The existing geoid file covers the DEM bounds. Existing geoid file: {geoid_path}."
                    )
                else:
                    logging.info(
                        f"The existing geoid file does not cover the DEM bounds. A new geoid file covering the bounds will be downloaded, overwriting the existing geiod file: {geoid_tif_path}."
                    )
                    download_egm_08_geoid(geoid_path, bounds=geoid_bounds)

    if geoid_crosses_antimeridian:
        logging.info(
            f"Reproject and merge east and west hemisphere geoid rasters to EPSG:{REMA_CRS}"
        )
        reproject_and_merge_rasters(
            geoid_tifs_to_apply, REMA_CRS, save_path=geoid_tif_path
        )
        # add a larger buffer to ensure geoid is applied correctly to all of dem
        dem_mask_buffer = 5_000
    else:
        # no buffer is required
        dem_mask_buffer = 0

    if ellipsoid_heights:
        logging.info(f"Returning DEM referenced to ellipsoidal heights")
        logging.info(f"Applying geoid to DEM : {geoid_tif_path}")
        # Apply the geoid only on areas where no rema data was found
        # i.e. these values were set to zero and will be replaced with
        # ellipsoid heights
        dem_array = apply_geoid(
            dem_array=dem_array,
            dem_profile=dem_profile,
            geoid_path=geoid_tif_path,
            buffer_pixels=2,
            dem_mask_buffer=dem_mask_buffer,
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
        logging.info(f"Applying geoid to DEM : {geoid_tif_path}")
        dem_array = apply_geoid(
            dem_array=dem_array,
            dem_profile=dem_profile,
            geoid_path=geoid_tif_path,
            buffer_pixels=2,
            dem_mask_buffer=dem_mask_buffer,
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


def make_empty_rema_profile_for_bounds(
    bounds: BoundingBox | tuple[float, float, float, float],
    resolution: int,
) -> dict:
    """Make an empty REMA DEM rasterio profile for given bounds.

    Parameters
    ----------
    bounds : BoundingBox | tuple
        (left, bottom, right, top) in EPSG:3031
    resolution : int
        Pixel size in metres

    Returns
    -------
    dict
        Rasterio profile
    """
    if isinstance(bounds, tuple):
        bounds = BoundingBox(*bounds)

    width = math.ceil((bounds.right - bounds.left) / resolution)
    height = math.ceil((bounds.top - bounds.bottom) / resolution)

    transform = Affine.translation(bounds.left, bounds.top) * Affine.scale(
        resolution, -resolution
    )

    dem_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": np.nan,
        "width": width,
        "height": height,
        "count": 1,
        "crs": CRS.from_epsg(3031),
        "transform": transform,
    }

    return dem_profile
