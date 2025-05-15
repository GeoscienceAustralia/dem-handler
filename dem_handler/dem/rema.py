from pathlib import Path
from shapely import box
import geopandas as gpd
from rasterio.profiles import Profile
import numpy as np
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


# Create a custom type that allows use of BoundingBox or tuple(xmin, ymin, xmax, ymax)
BBox = BoundingBox | tuple[float | int, float | int, float | int, float | int]

DATA_DIR = Path(__file__).parents[1] / Path("data")
REMA_GPKG_PATH = DATA_DIR / Path("REMA_Mosaic_Index_v2.gpkg")
REMA_VALID_RESOLUTIONS = [
    2,
    10,
    32,
]  # [2, 10, 32, 100, 500, 1000] It seems there are no higher resolutions in the new index


@log_timing
def get_rema_dem_for_bounds(
    bounds: BBox,
    save_path: Path | str = "",
    rema_index_path: Path | str = REMA_GPKG_PATH,
    local_dem_dir: Path | str | None = None,
    resolution: int = 2,
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
        raster_paths = list(Path(local_dem_dir).rglob("*.tif"))
        raster_names = [r.stem.replace("_dem", "") for r in raster_paths]
        s3_url_list = [url for url in s3_url_list if url.stem not in raster_names]

    if return_paths:
        if num_tasks:
            raster_paths.extend(
                [
                    Path(download_dir) / u.name.replace(".json", "_dem.tif")
                    for u in s3_url_list
                ]
            )
        else:
            dem_urls = [extract_s3_path(url.as_posix()) for url in s3_url_list]
            raster_paths.extend(
                [
                    Path(download_dir) / dem_url.split("amazonaws.com")[1][1:]
                    for dem_url in dem_urls
                ]
            )
        return raster_paths

    raster_paths.extend(
        download_rema_tiles(s3_url_list, Path(download_dir), num_cpus, num_tasks)
    )

    logging.info("combining found DEMs")
    dem_array, dem_profile = crop_datasets_to_bounds(
        raster_paths, bounds, Path(save_path)
    )

    # return the dem in either ellipsoid or geoid referenced heights. The REMA dem is already
    # referenced to the ellipsoid. Values are set to zero where no tile data exists
    # create a mask for this area so we can apply the geoid here to convert to ellipsoid heights
    dem_novalues_mask = dem_array == 0
    dem_novalues_count = np.count_nonzero(dem_novalues_mask)
    dem_values_mask = dem_array != 0

    if dem_novalues_count == 0 and ellipsoid_heights:
        # we have data everywhere and the values are already ellipsoid referenced
        logging.info(f"Dem array shape = {dem_array.shape}")
        return dem_array, dem_profile, raster_paths
    else:
        # get the geoid file
        if not download_geoid and not Path(geoid_tif_path).exists():
            raise FileExistsError(
                f"Geoid file does not exist: {geoid_tif_path}. "
                "correct path or set download_geoid = True"
            )
        elif download_geoid and not Path(geoid_tif_path).exists():
            logging.info(f"Downloading the egm_08 geoid")
            geoid_bounds = bounds
            if bounds_src_crs != GEOID_CRS:
                geoid_bounds = transform_polygon(
                    box(*bounds.bounds), bounds_src_crs, GEOID_CRS
                ).bounds

            download_egm_08_geoid(Path(geoid_tif_path), geoid_bounds)

        logging.info(f"Using geoid file: {geoid_tif_path}")

    if ellipsoid_heights:
        logging.info(f"Returning DEM referenced to ellipsoidal heights")
        dem_array = apply_geoid(
            dem_array=dem_array,
            dem_profile=dem_profile,
            geoid_path=Path(geoid_tif_path),
            buffer_pixels=2,
            save_path=Path(save_path),
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
            geoid_path=Path(geoid_tif_path),
            buffer_pixels=2,
            save_path=Path(save_path),
            mask_array=dem_values_mask,
            method="subtract",
        )

    logging.info(f"Dem array shape = {dem_array.shape}")
    return dem_array, dem_profile, raster_paths
