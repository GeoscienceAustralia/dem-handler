from affine import Affine
import math
import numpy as np
from rasterio.crs import CRS
import rasterio
import rasterio.mask
import geopandas as gpd
import numpy as np
from osgeo import gdal
from pathlib import Path
import shapely.geometry
import logging

from dem_handler.utils.spatial import (
    BoundingBox,
    check_s1_bounds_cross_antimeridian,
    get_target_antimeridian_projection,
    split_s1_bounds_at_am_crossing,
    adjust_bounds_at_high_lat,
    crop_datasets_to_bounds,
)
from dem_handler.utils.raster import (
    reproject_raster,
    merge_arrays_with_geometadata,
    adjust_pixel_coordinate_from_point_to_area,
    expand_bounding_box_to_pixel_edges,
)
from dem_handler.dem.geoid import remove_geoid
from dem_handler.download.aws import download_cop_glo30_tiles, download_egm_08_geoid

logger = logging.getLogger(__name__)

# Create a custom type that allows use of BoundingBox or tuple(xmin, ymin, xmax, ymax)
BBox = BoundingBox | tuple[float | int, float | int, float | int, float | int]

DATA_DIR = Path(__file__).parents[1] / Path("data")
COP30_GPKG_PATH = DATA_DIR / Path("copdem_tindex_filename.gpkg")


def get_cop30_dem_for_bounds(
    bounds: BBox,
    save_path: Path,
    ellipsoid_heights: bool = True,
    adjust_at_high_lat: bool = True,
    buffer_pixels: int | None = None,
    buffer_degrees: int | float | None = None,
    cop30_index_path: Path = COP30_GPKG_PATH,
    cop30_folder_path: Path = ".",
    geoid_tif_path: Path = "egm_08_geoid.tif",
    download_dem_tiles: bool = False,
    download_geoid: bool = False,
):

    # Convert bounding box to built-in bounding box type
    if isinstance(bounds, tuple):
        bounds = BoundingBox(*bounds)

    # Check if bounds cross the antimeridian
    antimeridian_crossing = check_s1_bounds_cross_antimeridian(
        bounds, max_scene_width=8
    )

    if antimeridian_crossing:
        logger.warning(
            "DEM crosses the dateline/antimeridian. Bounds will be split and processed."
        )

        target_crs = get_target_antimeridian_projection(bounds)

        logger.info(f"Splitting bounds into left and right side of antimeridian")
        bounds_eastern, bounds_western = split_s1_bounds_at_am_crossing(bounds)

        # Use recursion to process each side of the AM. The function is rerun
        # This time, antimeridian_crossing will be False enabling each side to be
        # independantly processed
        logger.info("Producing raster for Eastern Hemisphere bounds")
        eastern_save_path = save_path.parent.joinpath(
            save_path.stem + "_eastern" + save_path.suffix
        )
        get_cop30_dem_for_bounds(
            bounds_eastern,
            eastern_save_path,
            ellipsoid_heights,
            adjust_at_high_lat=True,
            buffer_pixels=buffer_pixels,
            cop30_index_path=cop30_index_path,
            cop30_folder_path=cop30_folder_path,
            geoid_tif_path=geoid_tif_path,
        )

        logger.info("Producing raster for Western Hemisphere bounds")
        western_save_path = save_path.parent.joinpath(
            save_path.stem + "_western" + save_path.suffix
        )
        get_cop30_dem_for_bounds(
            bounds_western,
            western_save_path,
            ellipsoid_heights,
            adjust_at_high_lat=True,
            buffer_pixels=buffer_pixels,
            cop30_index_path=cop30_index_path,
            cop30_folder_path=cop30_folder_path,
            geoid_tif_path=geoid_tif_path,
        )

        # reproject to 3031 and merge
        logging.info(
            f"Reprojecting Eastern and Western hemisphere rasters to EPGS:{target_crs}"
        )
        eastern_dem, eastern_profile = reproject_raster(eastern_save_path, target_crs)
        western_dem, western_profile = reproject_raster(western_save_path, target_crs)

        logging.info(f"Merging across antimeridian")
        dem_array, dem_profile = merge_arrays_with_geometadata(
            arrays=[eastern_dem, western_dem],
            profiles=[eastern_profile, western_profile],
            method="max",
            output_path=save_path,
        )

        return dem_array, dem_profile

    else:
        logger.info(f"Getting cop30m dem for bounds: {bounds.bounds}")

        # Adjust bounds at high latitude if requested
        if adjust_at_high_lat:
            adjusted_bounds = adjust_bounds_at_high_lat(bounds)
            logger.info(
                f"Getting cop30m dem for adjusted bounds: {adjusted_bounds.bounds}"
            )
        else:
            adjusted_bounds = bounds

        # Buffer bounds if reqeuested
        if buffer_pixels or buffer_degrees:
            logger.info(f"Buffering bounds by requested value")
            adjusted_bounds = buffer_bounds_cop_glo30(
                adjusted_bounds,
                pixel_buffer=buffer_pixels,
                degree_buffer=buffer_degrees,
            )

        # Before continuing, check that the new bounds for the dem cover the original bounds
        adjusted_bounds_polygon = shapely.geometry.box(*adjusted_bounds.bounds)
        bounds_polygon = shapely.geometry.box(*bounds.bounds)
        bounds_filled_by_dem = bounds_polygon.within(adjusted_bounds_polygon)
        print(bounds_polygon.bounds)
        if not bounds_filled_by_dem:
            warn_msg = (
                "The Cop30 DEM bounds do not fully cover the requested bounds. "
                "Try increasing the 'buffer_pixels' value. Note at the antimeridian "
                "This is expected, with bounds being slighly smaller on +ve side. "
                "e.g. max_lon is 179.9999 < 180."
            )
            logging.warning(warn_msg)

        # Adjust bounds further to be at full resolution pixel values
        # This function will expand the requested bounds to produce an integer number of pixels,
        # aligned with the cop glo30 pixel grid, in area-convention (top-left of pixel) coordinates.
        adjusted_bounds, adjusted_bounds_profile = (
            make_empty_cop_glo30_profile_for_bounds(adjusted_bounds)
        )
        print(adjusted_bounds.bounds)
        # Find cop glo30 paths for bounds
        logger.info(f"Finding intersecting DEM files from: {cop30_index_path}")
        dem_paths = find_required_dem_paths_from_index(
            adjusted_bounds,
            cop30_folder_path=cop30_folder_path,
            dem_index_path=cop30_index_path,
            tifs_in_subfolder=True,
            download_missing=download_dem_tiles,
        )

        # Display dem tiles to the user
        logger.info(f"{len(dem_paths)} tiles found in bounds")
        for p in dem_paths:
            logger.info(p)

        # Produce raster of zeros if no tiles are found
        if len(dem_paths) == 0:
            logger.warning(
                "No DEM tiles found. Assuming that the bounds are over water and creating a DEM containing all zeros."
            )

            dem_profile = adjusted_bounds_profile
            # Construct an array of zeros the same shape as the adjusted bounds profile
            dem_array = 0 * np.ones((dem_profile["height"], dem_profile["width"]))

            if save_path:
                with rasterio.open(save_path, "w", **dem_profile) as dst:
                    dst.write(dem_array, 1)
        # Create and read from VRT if tiles are found
        else:
            dem_array, dem_profile = crop_datasets_to_bounds(
                dem_paths, adjusted_bounds, save_path
            )

        if ellipsoid_heights:
            logging.info(
                f"Subtracting the geoid from the DEM to return ellipsoid heights"
            )
            if not download_geoid and not Path(geoid_tif_path).exists():
                raise FileExistsError(
                    f"Geoid file does not exist: {geoid_tif_path}. "
                    "correct path or set download_geoid = True"
                )
            elif download_geoid and not Path(geoid_tif_path).exists():
                logging.info(f"Downloading the egm_08 geoid")
                download_egm_08_geoid(geoid_tif_path, bounds=adjusted_bounds.bounds)

            logging.info(f"Using geoid file: {geoid_tif_path}")
            dem_array = remove_geoid(
                dem_array=dem_array,
                dem_profile=dem_profile,
                geoid_path=geoid_tif_path,
                buffer_pixels=2,
                save_path=save_path,
            )

        return dem_array, dem_profile


def find_required_dem_paths_from_index(
    bounds: BBox,
    cop30_folder_path: Path | None,
    dem_index_path=COP30_GPKG_PATH,
    search_buffer=0.3,
    tifs_in_subfolder=True,
    download_missing=False,
) -> list[Path]:

    if isinstance(bounds, tuple):
        bounds = BoundingBox(*bounds)

    gdf = gpd.read_file(dem_index_path)
    bounding_box = shapely.geometry.box(*bounds.bounds).buffer(search_buffer)
    print(bounding_box)

    if gdf.crs is not None:
        # ensure same crs
        bounding_box = (
            gpd.GeoSeries([bounding_box], crs="EPSG:4326").to_crs(gdf.crs).iloc[0]
        )
    # Find rows that intersect with the bounding box
    intersecting_tiles = gdf[gdf.intersects(bounding_box)]
    logger.info(
        f"Number of cop30 files found intersecting bounds : {len(intersecting_tiles)}"
    )
    if len(intersecting_tiles) == 0:
        # no intersecting tiles
        return []
    else:
        dem_tiles = sorted(intersecting_tiles.location.tolist())
        local_dem_paths = []
        missing_dems = []
        for i, t_filename in enumerate(dem_tiles):
            t_folder = (
                Path(cop30_folder_path)
                if not tifs_in_subfolder
                else Path(cop30_folder_path) / Path(t_filename).stem
            )
            t_path = t_folder / t_filename
            (
                local_dem_paths.append(t_path)
                if t_path.exists()
                else missing_dems.append(t_path)
            )
        logger.info(f"Local cop30m directory: {cop30_folder_path}")
        logger.info(f"Number of tiles existing locally : {len(local_dem_paths)}")
        logger.info(f"Number of tiles missing locally : {len(missing_dems)}")
        if download_missing and len(missing_dems) > 0:
            for t_path in missing_dems:
                download_cop_glo30_tiles(
                    tile_filename=t_path.name, save_folder=t_path.parent
                )
                local_dem_paths.append(t_path)
        local_dem_paths.append(t_path)

    return local_dem_paths


def buffer_bounds_cop_glo30(
    bounds: BoundingBox | tuple[float | int, float | int, float | int, float | int],
    pixel_buffer: int | None = None,
    degree_buffer: float | int | None = None,
) -> BoundingBox:
    """Buffer a bounding box by a fixed number of pixels or distance in decimal degrees

    Parameters
    ----------
    bounds : BoundingBox | tuple[float  |  int, float  |  int, float  |  int, float  |  int]
        The set of bounds (min_lon, min_lat, max_lon, max_lat)
    pixel_buffer : int | None, optional
        Number of pixels to buffer, by default None
    degree_buffer : float | int | None, optional
        Distance (in decimal degrees) to buffer by, by default None

    Returns
    -------
    BoundingBox
        Buffered bounds
    """

    if isinstance(bounds, tuple):
        bounds = BoundingBox(*bounds)

    if not pixel_buffer and not degree_buffer:
        logger.warning("No buffer has been provided.")
        return bounds

    if degree_buffer and pixel_buffer:
        logger.warning(
            "Both pixel and degree buffer provided. Degree buffer will be used."
        )
        pixel_buffer = None

    if pixel_buffer:
        lon_spacing, lat_spacing = get_cop_glo30_spacing(bounds)
        buffer = (pixel_buffer * lon_spacing, pixel_buffer * lat_spacing)

    if degree_buffer:
        buffer = (degree_buffer, degree_buffer)

    new_xmin = max(bounds.xmin - buffer[0], -180)
    new_ymin = max(bounds.ymin - buffer[1], -90)
    new_xmax = min(bounds.xmax + buffer[0], 180)
    new_ymax = min(bounds.ymax + buffer[1], 90)

    return BoundingBox(new_xmin, new_ymin, new_xmax, new_ymax)


def get_cop_glo30_spacing(
    bounds: BoundingBox | tuple[float | int, float | int, float | int, float | int],
) -> tuple[float, float]:
    """Get the longitude and latitude spacing for the Copernicus GLO30 DEM at the centre of the bounds

    Parameters
    ----------
    bounds : BoundingBox | tuple[float  |  int, float  |  int, float  |  int, float  |  int]
        The set of bounds (min_lon, min_lat, max_lon, max_lat)

    Returns
    -------
    tuple[float, float]
        A tuple of the longitude and latitude spacing

    Raises
    ------
    ValueError
        If the absolute latitude of bounds does not fall within expected range (<90)
    """

    if isinstance(bounds, tuple):
        bounds = BoundingBox(*bounds)

    mean_latitude = abs((bounds.ymin + bounds.ymax) / 2)

    minimum_pixel_spacing = 0.0002777777777777778

    # Latitude spacing
    latitude_spacing = minimum_pixel_spacing

    # Longitude spacing
    if mean_latitude < 50:
        longitude_spacing = minimum_pixel_spacing
    elif mean_latitude < 60:
        longitude_spacing = minimum_pixel_spacing * 1.5
    elif mean_latitude < 70:
        longitude_spacing = minimum_pixel_spacing * 2
    elif mean_latitude < 80:
        longitude_spacing = minimum_pixel_spacing * 3
    elif mean_latitude < 85:
        longitude_spacing = minimum_pixel_spacing * 5
    elif mean_latitude < 90:
        longitude_spacing = minimum_pixel_spacing * 10
    else:
        raise ValueError("cannot resolve cop30m lattitude")

    return (longitude_spacing, latitude_spacing)


def get_cop_glo30_tile_transform(
    origin_lon: float, origin_lat: float, spacing_lon: float, spacing_lat: float
) -> Affine:
    """Generates an Affine transform with the origin in the top-left of the Copernicus GLO30 DEM
    containing the provided origin.

    Parameters
    ----------
    origin_lon : float
        Origin longitude
    origin_lat : float
        Origin latitude
    spacing_lon : float
        Pixel spacing in longitude
    spacing_lat : float
        Pixel spacing in latitude

    Returns
    -------
    Affine
        An Affine transform with the origin at the top-left pixel of the tile containing the supplied origin
    """

    # Find whole degree value containing the origin
    whole_degree_origin_lon = math.floor(origin_lon)
    whole_degree_origin_lat = math.ceil(origin_lat)

    # Create the scaling from spacing
    scaling = (spacing_lon, -spacing_lat)

    # Adjust to the required 0.5 pixel offset
    adjusted_origin = adjust_pixel_coordinate_from_point_to_area(
        (whole_degree_origin_lon, whole_degree_origin_lat), scaling
    )

    transfrom = Affine.translation(*adjusted_origin) * Affine.scale(*scaling)

    return transfrom


def make_empty_cop_glo30_profile_for_bounds(
    bounds: BoundingBox | tuple[float | int, float | int, float | int, float | int],
) -> tuple[tuple, dict]:
    """make an empty cop30m dem rasterio profile based on a set of bounds.
    The desired pixel spacing changes based on lattitude
    see : https://copernicus-dem-30m.s3.amazonaws.com/readme.html

    Parameters
    ----------
    bounds : BoundingBox | tuple[float | int, float | int, float | int, float | int]
        The set of bounds (min_lon, min_lat, max_lon, max_lat)
    pixel_buffer | int
        The number of pixels to add as a buffer to the profile

    Returns
    -------
    dict
        A rasterio profile

    Raises
    ------
    ValueError
        If the latitude of the supplied bounds cannot be
        associated with a target pixel size
    """
    if isinstance(bounds, tuple):
        bounds = BoundingBox(*bounds)

    spacing_lon, spacing_lat = get_cop_glo30_spacing(bounds)

    glo30_transform = get_cop_glo30_tile_transform(
        bounds.xmin, bounds.ymax, spacing_lon, spacing_lat
    )

    # Expand the bounds to the edges of pixels
    expanded_bounds, expanded_transform = expand_bounding_box_to_pixel_edges(
        bounds.bounds, glo30_transform
    )
    if isinstance(expanded_bounds, tuple):
        expanded_bounds = BoundingBox(*expanded_bounds)

    # Convert bounds from world to pixel to get width and height
    left_px, top_px = ~expanded_transform * expanded_bounds.top_left
    right_px, bottom_px = ~expanded_transform * expanded_bounds.bottom_right

    width = abs(round(right_px) - round(left_px))
    height = abs(round(bottom_px) - round(top_px))

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": np.nan,
        "width": width,
        "height": height,
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": expanded_transform,
        "blockysize": 1,
        "tiled": False,
        "interleave": "band",
    }

    return (expanded_bounds, profile)
