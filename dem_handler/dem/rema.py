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
from dem_handler.download.aws import download_rema_tiles

from dem_handler.dem.geoid import remove_geoid
from dem_handler.download.aws import download_egm_08_geoid
from dem_handler.utils.rio_tools import reproject_profile_to_new_crs


# Create a custom type that allows use of BoundingBox or tuple(xmin, ymin, xmax, ymax)
BBox = BoundingBox | tuple[float | int, float | int, float | int, float | int]

DATA_DIR = Path(__file__).parents[1] / Path("data")
REMA_GPKG_PATH = DATA_DIR / Path("REMA_Mosaic_Index_v2.gpkg")
REMA_VALID_RESOLUTIONS = [
    2,
    10,
    32,
]  # [2, 10, 32, 100, 500, 1000] It seems there are no higher resolutions in the new index


def get_rema_dem_for_bounds(
    bounds: BBox,
    save_path: Path = "",
    rema_index_path: Path = REMA_GPKG_PATH,
    resolution: int = 2,
    bounds_src_crs: int = 3031,
    ellipsoid_heights: bool = True,
    geoid_tif_path: Path = "egm_08_geoid.tif",
    download_geoid: bool = False,
) -> tuple[np.ndarray, Profile]:
    """Finds the REMA DEM tiles in a given bounding box and merges them into a single tile.

    Parameters
    ----------
    bounds : BBox
        BoundingBox object or tuple of coordinates
    save_path : Path, optional
        Local path to save the output tile, by default ""
    rema_index_path : Path, optional
        Path to the index files with the list of REMA tiles in it, by default REMA_GPKG_PATH
    resolution : int, optional
        Resolution of the required tiles, by default 2
    bounds_src_crs : int, optional
        CRS of the provided bounding box, by default 3031
    ellipsoid_heights : bool, optional
        Subtracts the geoid height from the tiles to get the ellipsoid height, by default True
    geoid_tif_path : Path, optional
        Path to the existing ellipsoid file, by default "egm_08_geoid.tif"
    download_geoid : bool, optional
        Flag to download the ellipsoid file, by default False

    Returns
    -------
    tuple[np.ndarray, Profile]
        Tuple of the output tile array and its profile

    Raises
    ------
    FileExistsError
        If `ellipsoid_heights` is True, it will raise an error if the ellipsoid file does not exist and `download_geoid` is set to False.
    """

    TEMP_SAVE_FOLDER = Path("rema_dems_temp_folder")
    GEOID_CRS = 4326
    REMA_CRS = 3031

    assert (
        resolution in REMA_VALID_RESOLUTIONS
    ), f"resolution must be in {REMA_VALID_RESOLUTIONS}"

    if type(bounds) != BoundingBox:
        bounds = BoundingBox(*bounds)

    if bounds_src_crs != REMA_CRS:
        bounds_poly = transform_polygon(box(*bounds.bounds), bounds_src_crs, REMA_CRS)
        bounds = BoundingBox(
            *transform_polygon(box(*bounds.bounds), bounds_src_crs, REMA_CRS).bounds
        )
        bounds_src_crs = REMA_CRS
    else:
        bounds_poly = box(*bounds.bounds)

    rema_layer = f"REMA_Mosaic_Index_v2_{resolution}m"
    rema_index_df = gpd.read_file(rema_index_path, layer=rema_layer)

    intersecting_rema_files = rema_index_df[
        rema_index_df.geometry.intersects(bounds_poly)
    ]
    s3_url_list = intersecting_rema_files["s3url"].to_list()
    logging.info(f"{len(s3_url_list)} intersecting tiles found")

    logging.info("combining found DEMS")
    rasters = download_rema_tiles(s3_url_list[0:], TEMP_SAVE_FOLDER)
    dem_array, dem_profile = crop_datasets_to_bounds(rasters, bounds, save_path)

    if ellipsoid_heights:
        logging.info(f"Subtracting the geoid from the DEM to return ellipsoid heights")
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

            download_egm_08_geoid(geoid_tif_path, geoid_bounds)

        logging.info(f"Using geoid file: {geoid_tif_path}")
        dem_array = remove_geoid(
            dem_array=dem_array,
            dem_profile=dem_profile,
            geoid_path=geoid_tif_path,
            buffer_pixels=2,
            save_path=save_path,
        )
        dem_array = np.squeeze(dem_array)

    return dem_array, dem_profile
