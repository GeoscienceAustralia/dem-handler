from __future__ import annotations
from affine import Affine
import math
import numpy as np
from rasterio.crs import CRS
import rasterio
import rasterio.mask
import geopandas as gpd
import numpy as np
from osgeo import gdal
from pathlib import Path, PurePath
import shapely.geometry
import logging

from dem_handler.utils.spatial import (
    BoundingBox,
    check_bounds_likely_cross_antimeridian,
    get_target_antimeridian_projection,
    split_bounds_at_antimeridian,
    adjust_bounds_at_high_lat,
    crop_datasets_to_bounds,
)
from dem_handler.utils.raster import (
    reproject_raster,
    merge_arrays_with_geometadata,
    adjust_pixel_coordinate_from_point_to_area,
    expand_bounding_box_to_pixel_edges,
)
from dem_handler.utils.general import log_timing
from dem_handler.dem.geoid import apply_geoid
from dem_handler.download.aws import download_cop_glo30_tiles, download_egm_08_geoid

logger = logging.getLogger(__name__)

# Create a custom type that allows use of BoundingBox or tuple(left, bottom, right, top)
BBox = BoundingBox | tuple[float | int, float | int, float | int, float | int]

from dem_handler import COP30_GPKG_PATH


@log_timing
def get_cop30_dem_for_bounds(
    bounds: BBox,
    save_path: Path,
    ellipsoid_heights: bool = True,
    adjust_at_high_lat: bool = False,
    buffer_pixels: int | None = None,
    buffer_degrees: int | float | None = None,
    cop30_index_path: Path = COP30_GPKG_PATH,
    cop30_folder_path: Path = ".",
    geoid_tif_path: Path = "egm_08_geoid.tif",
    download_dem_tiles: bool = False,
    download_geoid: bool = False,
    num_cpus: int = 1,
    num_tasks: int | None = None,
    return_paths: bool = False,
    download_dir: Path | None = None,
):
    """
    Retrieve and mosaic COPDEM GLO-30 (Copernicus 30 m DEM) tiles covering a specified geographic bounding box.

    This function locates and optionally downloads the Copernicus GLO-30 Digital Elevation Model (DEM)
    tiles that intersect the requested bounding box. It handles high-latitude adjustments, buffering,
    ellipsoidal height conversion using a geoid model, and special cases where the bounds cross
    the antimeridian (±180° longitude).

    The function can return either a merged DEM array with metadata or a list of intersecting DEM tile paths.

    Parameters
    ----------
    bounds : BBox or tuple
        The geographic bounding box of interest, either as a `BBox` object or a tuple
        `(min_x, min_y, max_x, max_y)` in degrees.
    save_path : Path
        File path where the output DEM (GeoTIFF) will be saved.
    ellipsoid_heights : bool, optional
        If True, converts DEM heights from the geoid reference to ellipsoidal heights
        using the EGM08 geoid model. Default is True.
    adjust_at_high_lat : bool, optional
        If True, expands the bounds near the poles to ensure adequate DEM coverage.
        Default is False.
    buffer_pixels : int or None, optional
        Optional pixel buffer applied around the requested bounds to include a margin of DEM data.
        Default is None.
    buffer_degrees : int or float or None, optional
        Optional geographic buffer in degrees around the bounds. Default is None.
    cop30_index_path : Path, optional
        Path to the COPDEM GLO-30 index GeoPackage (`.gpkg`) used to locate intersecting DEM tiles.
        Default is `COP30_GPKG_PATH`.
    cop30_folder_path : Path, optional
        Directory containing the COPDEM GLO-30 tiles. Default is the current directory.
    geoid_tif_path : Path, optional
        Path to the local geoid model GeoTIFF (e.g., `egm_08_geoid.tif`) used for height conversion.
        Default is `"egm_08_geoid.tif"`.
    download_dem_tiles : bool, optional
        If True, automatically downloads any missing DEM tiles required to cover the requested bounds.
        Default is False.
    download_geoid : bool, optional
        If True, downloads the EGM08 geoid model if it does not exist locally.
        Default is False.
    num_cpus : int, optional
        Number of CPU cores to use for parallel tasks such as downloading or merging tiles.
        Default is 1.
    num_tasks : int or None, optional
        Number of parallel tasks to execute when searching or downloading tiles.
        Default is None (serial execution).
    return_paths : bool, optional
        If True, returns only a list of file paths to intersecting DEM tiles rather than reading or merging them.
        Default is False.
    download_dir : Path or None, optional
        Directory where downloaded DEM tiles or geoid files should be saved. Default is None (current directory).

    Returns
    -------
    tuple or list
        If `return_paths=True`, returns:
            list of Path
                File paths to the intersecting COPDEM GLO-30 tiles.
        Otherwise, returns:
            tuple (dem_array, dem_profile, dem_paths)
                - dem_array : numpy.ndarray
                  The merged DEM raster data covering the requested bounds.
                - dem_profile : dict
                  Raster metadata/profile dictionary compatible with `rasterio`.
                - dem_paths : list of Path
                  The DEM tile paths used to produce the merged output.

    Raises
    ------
    FileExistsError
        If the geoid model file does not exist locally and `download_geoid=False`.
    ValueError
        If the input bounds are invalid or cannot be processed.
    RuntimeError
        If DEM merging or reprojection fails during processing.

    Notes
    -----
    - If the bounds cross the antimeridian the function recursively processes each
    side of the antimeridian, reprojects them into a common coordinate reference system,
    and merges them into a continuous DEM.
    - The DEM heights are geoid-referenced by default (EGM08 model). Set `ellipsoid_heights=True`
      to obtain ellipsoidal heights (WGS84).
    - At high latitudes or near the poles, `adjust_at_high_lat=True` can help include complete DEM coverage.
    """

    # Convert bounding box to built-in bounding box type
    if isinstance(bounds, tuple):
        bounds = BoundingBox(*bounds)

    # Log the requested bounds
    logger.info(f"Getting cop30m dem for bounds: {bounds.bounds}")

    # Check if bounds cross the antimeridian
    if bounds.left > bounds.right:
        logger.warning(
            f"left longitude value ({bounds[0]}) is greater than the right longitude value {({bounds[2]})} "
            f"for the bounds provided. Assuming the bounds cross the antimeridian : {bounds}"
        )
        antimeridian_crossing = True
    else:
        antimeridian_crossing = False
        # run a basic to check if the bounds likely cross the antimeridian but
        # are just formatted wrong. If so, warn the user.
        if check_bounds_likely_cross_antimeridian(bounds):
            logger.warning(
                "Provided bounds have very large longitude extent. If the shape crosses the "
                f"antimeridian, reformat the bounds as : ({bounds[2]}, {bounds[1]}, {bounds[0]}, {bounds[3]})"
            )

    if antimeridian_crossing:
        logger.warning(
            "DEM crosses the dateline/antimeridian. Bounds will be split and processed."
        )

        target_crs = get_target_antimeridian_projection(bounds)

        logger.info(f"Splitting bounds into left and right side of antimeridian")
        bounds_eastern, bounds_western = split_bounds_at_antimeridian(bounds)

        # Use recursion to process each side of the AM. The function is rerun
        # This time, antimeridian_crossing will be False enabling each side to be
        # independently processed
        logger.info("Producing raster for Eastern Hemisphere bounds")
        save_path = Path(save_path)
        geoid_tif_path = Path(geoid_tif_path)
        eastern_dem_save_path = save_path.parent.joinpath(
            save_path.stem + "_eastern" + save_path.suffix
        )
        if download_geoid:
            # When downloading, this will save the eastern hemisphere geoid to a unique file.
            eastern_geoid_tif_path = geoid_tif_path.parent.joinpath(
                geoid_tif_path.stem + "_eastern" + geoid_tif_path.suffix
            )
        else:
            # This assumes that you have one local geoid file that covers 180 degrees to -180 degrees.
            # The program logic would need to be rewritten if dealing with multiple geoid files.
            eastern_geoid_tif_path = geoid_tif_path
        eastern_output = get_cop30_dem_for_bounds(
            bounds_eastern,
            eastern_dem_save_path,
            ellipsoid_heights,
            adjust_at_high_lat=adjust_at_high_lat,
            buffer_pixels=buffer_pixels,
            buffer_degrees=buffer_degrees,
            cop30_index_path=cop30_index_path,
            cop30_folder_path=cop30_folder_path,
            geoid_tif_path=eastern_geoid_tif_path,
            download_dem_tiles=download_dem_tiles,
            download_geoid=download_geoid,
            num_cpus=num_cpus,
            num_tasks=num_tasks,
            return_paths=return_paths,
            download_dir=download_dir,
        )

        logger.info("Producing raster for Western Hemisphere bounds")
        western_dem_save_path = save_path.parent.joinpath(
            save_path.stem + "_western" + save_path.suffix
        )
        if download_geoid:
            # When downloading, this will save the western hemisphere geoid to a unique file.
            western_geoid_tif_path = geoid_tif_path.parent.joinpath(
                geoid_tif_path.stem + "_western" + geoid_tif_path.suffix
            )
        else:
            # This assumes that you have one local geoid file that covers 180 degrees to -180 degrees.
            # The program logic would need to be rewritten if dealing with multiple geoid files.
            western_geoid_tif_path = geoid_tif_path
        western_output = get_cop30_dem_for_bounds(
            bounds_western,
            western_dem_save_path,
            ellipsoid_heights,
            adjust_at_high_lat=adjust_at_high_lat,
            buffer_pixels=buffer_pixels,
            buffer_degrees=buffer_degrees,
            cop30_index_path=cop30_index_path,
            cop30_folder_path=cop30_folder_path,
            geoid_tif_path=western_geoid_tif_path,
            download_dem_tiles=download_dem_tiles,
            download_geoid=download_geoid,
            num_cpus=num_cpus,
            num_tasks=num_tasks,
            return_paths=return_paths,
            download_dir=download_dir,
        )

        if return_paths:
            return eastern_output + western_output

        # reproject to 3031 and merge
        logging.info(
            f"Reprojecting Eastern and Western hemisphere rasters to EPSG:{target_crs}"
        )
        eastern_dem, eastern_profile = reproject_raster(
            eastern_dem_save_path, target_crs
        )
        western_dem, western_profile = reproject_raster(
            western_dem_save_path, target_crs
        )

        logging.info(f"Merging across antimeridian")
        dem_array, dem_profile = merge_arrays_with_geometadata(
            arrays=[western_dem, eastern_dem],
            profiles=[western_profile, eastern_profile],
            method="max",
            output_path=save_path,
        )

        return dem_array, dem_profile, eastern_output[2] + western_output[2]

    else:
        # Adjust bounds at high latitude if requested
        if adjust_at_high_lat:
            logger.info(
                f"Adjusting bounds at high latitude, this may return additional data than requested"
            )
            adjusted_bounds = adjust_bounds_at_high_lat(bounds)
            logger.info(
                f"Getting cop30m dem for adjusted bounds: {adjusted_bounds.bounds}"
            )
        else:
            adjusted_bounds = bounds

        # Buffer bounds if requested
        if buffer_pixels or buffer_degrees:
            logger.info(f"Buffering bounds by requested value")
            adjusted_bounds = buffer_bounds_cop_glo30(
                adjusted_bounds,
                pixel_buffer=buffer_pixels,
                degree_buffer=buffer_degrees,
            )
            logger.info(f"Getting cop30m dem for buffered bounds : {adjusted_bounds}")

        # Before continuing, check that the new bounds for the dem cover the original bounds
        adjusted_bounds_polygon = shapely.geometry.box(*adjusted_bounds.bounds)
        bounds_polygon = shapely.geometry.box(*bounds.bounds)
        bounds_filled_by_dem = bounds_polygon.within(adjusted_bounds_polygon)
        if not bounds_filled_by_dem:
            warn_msg = (
                "The Cop30 DEM bounds do not fully cover the requested bounds. "
                "Try increasing the 'buffer_pixels' value. Note at the antimeridian "
                "This is expected, with bounds being slightly smaller on +ve side. "
                "e.g. right is 179.9999 < 180."
            )
            logging.warning(warn_msg)

        # Adjust bounds further to be at full resolution pixel values
        # This function will expand the requested bounds to produce an integer number of pixels,
        # aligned with the cop glo30 pixel grid, in area-convention (top-left of pixel) coordinates.
        adjusted_bounds, adjusted_bounds_profile = (
            make_empty_cop_glo30_profile_for_bounds(adjusted_bounds)
        )
        # Find cop glo30 paths for bounds
        logger.info(f"Finding intersecting DEM files from: {cop30_index_path}")
        dem_paths = find_required_dem_paths_from_index(
            adjusted_bounds,
            cop30_folder_path=cop30_folder_path,
            dem_index_path=cop30_index_path,
            tifs_in_subfolder=False if num_tasks else True,
            download_missing=download_dem_tiles,
            num_cpus=num_cpus,
            num_tasks=num_tasks,
            download_dir=download_dir,
            return_paths=return_paths,
        )

        # Display dem tiles to the user
        logger.info(f"{len(dem_paths)} tiles found in bounds")
        for p in dem_paths:
            logger.info(p)

        if return_paths:
            return dem_paths

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
            logging.info(f"Returning DEM referenced to ellipsoidal heights")
            if not download_geoid and not Path(geoid_tif_path).exists():
                raise FileExistsError(
                    f"Geoid file does not exist: {geoid_tif_path}. "
                    "correct path or set download_geoid = True"
                )
            elif download_geoid and not Path(geoid_tif_path).exists():
                logging.info(f"Downloading the egm_08 geoid")
                download_egm_08_geoid(geoid_tif_path, bounds=adjusted_bounds.bounds)
            elif download_geoid and Path(geoid_tif_path).exists():
                # Check that the existing geiod covers the dem
                with rasterio.open(geoid_tif_path) as src:
                    existing_geoid_bounds = shapely.geometry.box(*src.bounds)
                if existing_geoid_bounds.covers(
                    shapely.geometry.box(*adjusted_bounds.bounds)
                ):
                    logging.info(
                        f"Skipping geoid download. The existing geoid file covers the DEM bounds. Existing geoid file: {geoid_tif_path}."
                    )
                else:
                    logging.info(
                        f"The existing geoid file does not cover the DEM bounds. A new geoid file covering the bounds will be downloaded, overwriting the existing geiod file: {geoid_tif_path}."
                    )
                    download_egm_08_geoid(geoid_tif_path, bounds=adjusted_bounds.bounds)

            logging.info(f"Using geoid file: {geoid_tif_path}")
            dem_array = apply_geoid(
                dem_array=dem_array,
                dem_profile=dem_profile,
                geoid_path=geoid_tif_path,
                buffer_pixels=2,
                save_path=save_path,
                method="add",
            )

        else:
            logging.info(f"Returning DEM referenced to geoid heights")

        logging.info(f"Dem array shape = {dem_array.shape}")
        return dem_array, dem_profile, dem_paths


def find_required_dem_paths_from_index(
    bounds: BBox,
    cop30_folder_path: Path | None,
    dem_index_path=COP30_GPKG_PATH,
    search_buffer=0.0,
    tifs_in_subfolder=True,
    download_missing=False,
    num_cpus: int = 1,
    num_tasks: int | None = None,
    download_dir: Path | None = None,
    return_paths: bool = False,
) -> list[Path]:

    logger.info(f"Requested folder for tiles: {cop30_folder_path}")

    if isinstance(bounds, tuple):
        bounds = BoundingBox(*bounds)

    gdf = gpd.read_file(dem_index_path)
    bounding_box = shapely.geometry.box(*bounds.bounds).buffer(search_buffer)

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
            t_tif_name = Path(t_filename).name
            t_folder = (
                Path(cop30_folder_path)
                if not tifs_in_subfolder
                else Path(cop30_folder_path) / PurePath(t_tif_name).stem
            )
            t_path = t_folder / t_tif_name
            (
                local_dem_paths.append(t_path)
                if t_path.exists()
                else missing_dems.append(t_path)
            )
        logger.info(f"Local cop30m directory: {cop30_folder_path}")
        logger.info(f"Number of tiles existing locally : {len(local_dem_paths)}")
        logger.info(f"Number of tiles missing locally : {len(missing_dems)}")
        if download_missing and len(missing_dems) > 0:
            if not download_dir:
                if num_tasks:
                    download_dir = cop30_folder_path
                else:
                    download_dir = Path("")
            if not return_paths:
                download_cop_glo30_tiles(
                    tile_filenames=[
                        Path(missed_path.name) for missed_path in missing_dems
                    ],
                    save_folder=(
                        download_dir
                        if num_tasks
                        else [
                            download_dir / missed_path.parent
                            for missed_path in missing_dems
                        ]
                    ),
                    num_cpus=num_cpus,
                    num_tasks=num_tasks,
                )
            local_dem_paths.extend(
                [download_dir / missed_path.name for missed_path in missing_dems]
                if num_tasks
                else [download_dir / missed_path for missed_path in missing_dems]
            )

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
        The set of bounds (left, bottom, right, top)
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

    new_left = max(bounds.left - buffer[0], -180)
    new_bottom = max(bounds.bottom - buffer[1], -90)
    new_right = min(bounds.right + buffer[0], 180)
    new_top = min(bounds.top + buffer[1], 90)

    return BoundingBox(new_left, new_bottom, new_right, new_top)


def get_cop_glo30_spacing(
    bounds: BoundingBox | tuple[float | int, float | int, float | int, float | int],
) -> tuple[float, float]:
    """Get the longitude and latitude spacing for the Copernicus GLO30 DEM at the centre of the bounds

    Parameters
    ----------
    bounds : BoundingBox | tuple[float  |  int, float  |  int, float  |  int, float  |  int]
        The set of bounds (left, bottom, right, top)

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

    mean_latitude = abs((bounds.bottom + bounds.top) / 2)

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
        raise ValueError("cannot resolve cop30m latitude")

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

    transform = Affine.translation(*adjusted_origin) * Affine.scale(*scaling)

    return transform


def make_empty_cop_glo30_profile_for_bounds(
    bounds: BoundingBox | tuple[float | int, float | int, float | int, float | int],
) -> tuple[tuple, dict]:
    """make an empty cop30m dem rasterio profile based on a set of bounds.
    The desired pixel spacing changes based on latitude
    see : https://copernicus-dem-30m.s3.amazonaws.com/readme.html

    Parameters
    ----------
    bounds : BoundingBox | tuple[float | int, float | int, float | int, float | int]
        The set of bounds (left, bottom, right, top)
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
        bounds.left, bounds.top, spacing_lon, spacing_lat
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
