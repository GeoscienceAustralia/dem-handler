from __future__ import annotations
from dataclasses import dataclass
import geopandas as gpd
import pyproj
from shapely import segmentize
from shapely.geometry import Polygon, box
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest
from pyproj import CRS
from osgeo import gdal
import shapely
import rasterio
from rasterio.io import DatasetReader
from rasterio.profiles import Profile
import json
import logging
import numpy as np
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

from dem_handler import (
    REMA_GPKG_PATH,
    COP30_GPKG_PATH,
    REMA_VALID_RESOLUTIONS,
    ValidDEMResolutions,
    COP_VALID_RESOLUTIONS,
)


# Construct a dataclass for bounding boxes
@dataclass
class BoundingBox:
    left: float | int
    bottom: float | int
    right: float | int
    top: float | int

    @property
    def bounds(self) -> tuple[float | int, float | int, float | int, float | int]:
        return (self.left, self.bottom, self.right, self.top)

    @property
    def top_left(self) -> tuple[float | int, float | int]:
        return (self.left, self.top)

    @property
    def bottom_right(self) -> tuple[float | int, float | int]:
        return (self.right, self.bottom)

    def __getitem__(self, index: int) -> float | int:
        """Allow list-style indexing for bounding box values."""
        try:
            return self.bounds[index]
        except IndexError:
            raise IndexError("BoundingBox index out of range (valid indices: 0–3)")

    # Run checks on the bounding box values
    def __post_init__(self):
        if self.bottom >= self.top:
            raise ValueError(
                "The bounding box's bottom value is greater than or equal to the top value. Check ordering"
            )
        if self.left >= self.right:
            logger.warning(
                "The bounding box's left value is greater than or equal to the right value. "
                "Assuming the bounds cross the antimeridian. Refactor the bounds if this is not correct."
            )


# Create a custom type that allows use of BoundingBox or tuple(left, bottom, right, top)
BBox = BoundingBox | tuple[float | int, float | int, float | int, float | int]


def transform_polygon(
    geometry: Polygon, src_crs: int, dst_crs: int, always_xy: bool = True
):
    src_crs = pyproj.CRS(f"EPSG:{src_crs}")
    dst_crs = pyproj.CRS(f"EPSG:{dst_crs}")
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=always_xy)
    # Transform the polygon's coordinates
    if isinstance(geometry, Polygon):
        # Transform exterior
        exterior_coords = [
            transformer.transform(x, y) for x, y in geometry.exterior.coords
        ]
        # Transform interiors (holes)
        interiors_coords = [
            [transformer.transform(x, y) for x, y in interior.coords]
            for interior in geometry.interiors
        ]
        # Create the transformed polygon
        return Polygon(exterior_coords, interiors_coords)

    # Handle other geometry types as needed
    raise ValueError("Only Polygon geometries are supported for transformation.")


def adjust_bounds(
    bounds: BoundingBox | tuple[float | int, float | int, float | int, float | int],
    src_crs: int,
    ref_crs: int,
    segment_length: float = 0.1,
) -> tuple:
    """_summary_

    Parameters
    ----------
    bounds : BoundingBox | tuple[float | int, float | int, float | int, float | int],
        Bounds to adjust.
    src_crs : int
        Source EPSG. e.g. 4326
    ref_crs : int
        Reference crs to create the true bbox. i.e. 3031 in southern
        hemisphere and 3995 in northern (polar stereographic)
    segment_length : float, optional
        distance between generation points along the bounding box sides in
        src_crs. e.g. 0.1 degrees in lat/lon, by default 0.1

    Returns
    -------
    BoundingBox
        A polygon bounding box expanded to the true min max
    """
    if isinstance(bounds, tuple):
        bounds = BoundingBox(*bounds)

    geometry = box(*bounds.bounds)
    segmentized_geometry = segmentize(geometry, max_segment_length=segment_length)
    transformed_geometry = transform_polygon(segmentized_geometry, src_crs, ref_crs)
    transformed_box = box(*transformed_geometry.bounds)
    corrected_geometry = transform_polygon(transformed_box, ref_crs, src_crs)
    return BoundingBox(*corrected_geometry.bounds)


def get_local_utm(
    bounds: BoundingBox | tuple[float | int, float | int, float | int, float | int],
    antimeridian: bool = False,
) -> int:
    """_summary_

    Parameters
    ----------
    bounds : BoundingBox | tuple[float  |  int, float  |  int, float  |  int, float  |  int]
        The set of bounds (left, bottom, top, right)
    antimeridian : bool, optional
        Whether the bounds cross the antimeridian, by default False

    Returns
    -------
    int
        The CRS in integer form (e.g. 32749 for WGS 84 / UTM zone 49S)
    """
    if bounds.isinstance(tuple):
        bounds = BoundingBox(*bounds)

    logger.info("Finding best crs for area")
    centre_lat = (bounds.bottom + bounds.top) / 2
    centre_lon = (bounds.left + bounds.right) / 2
    if antimeridian:
        # force the lon to be next to antimeridian on the side with the scene centre.
        # e.g. (-177 + 178)/2 = 1, this is > 0 more data on -'ve side
        centre_lon = 179.9 if centre_lon < 0 else -179.9
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=centre_lon - 0.01,
            south_lat_degree=centre_lat - 0.01,
            east_lon_degree=centre_lon + 0.01,
            north_lat_degree=centre_lat + 0.01,
        ),
    )
    crs = CRS.from_epsg(utm_crs_list[0].code)
    crs = str(crs).split(":")[-1]  # get the EPSG integer
    return int(crs)


def bounds_to_polygon(bounds, save_path=""):
    """
    Convert a bounding box to a GeoJSON Polygon.

    Parameters:
        bounds (tuple): A tuple of (left, bottom, right, top).
        save_path (str): path to save the geojson

    Returns:
        dict: A GeoJSON FeatureCollection with a single Polygon feature.
    """
    left, bottom, right, top = bounds

    # Define the polygon coordinates
    coordinates = [
        [
            [left, bottom],
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],  # Closing the polygon
        ]
    ]

    # Create the GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": coordinates},
                "properties": {},
            }
        ],
    }

    if save_path:
        with open(save_path, "w") as f:
            f.write(json.dumps(geojson))

    return geojson


def check_bounds_likely_cross_antimeridian(
    bounds: BBox, max_crossing_width: int = 20
) -> bool:
    """Check if the bounds likely cross the antimeridian and may not be correctly
    formatted. Can occur if the bounds are provided as (xmin, ymin, xmax, ymax)
    rather than (left, bottom, right, top). For example, the bounds (-178, -60, 179, -57)
    are valid, but stretch 357 degrees across the earth. It is therefore likely
    the scene crosses the antimeridian with correct bounds (179, -60, -178, -57) representing
    the left, bottom, right and top values respectively. The function works based on
    allowable antimeridian crossing width (max_crossing_width)

    Parameters
    ----------
    bounds : BoundingBox
        the set of bounds (xmin, ymin, xmax, ymax) / (left, bottom, right, top)
    max_crossing_width : int, optional
        maximum allowable width of bounds degrees, by default 20

    Returns
    -------
    bool
        if the bounds likely cross the antimeridian
    """

    if isinstance(bounds, tuple):
        bounds = BoundingBox(*bounds)

    antimeridian_xmin = -180
    bounding_xmin = antimeridian_xmin + max_crossing_width  # -160 by default

    antimeridian_xmax = 180
    bounding_xmax = antimeridian_xmax - max_crossing_width  # 160 by default

    if (bounds.left < bounding_xmin) and (bounds.left > antimeridian_xmin):
        if bounds.right > bounding_xmax and bounds.right < antimeridian_xmax:
            return True
    return False


def get_target_antimeridian_projection(bounds: BoundingBox) -> int:
    """depending where were are on the earth, the desired
    crs at the antimeridian will change. e.g. polar stereo
    is desired at high and low lats, local utm zone elsewhere
    (e.g. at the equator).

    Parameters
    ----------
    bounds : BoundingBox
        The set of bounds (left, bottom, right, top)

    Returns
    -------
    int
        The CRS in integer form (e.g. 3031 for Polar Stereographic)
    """
    bottom = min(bounds.bottom, bounds.top)
    target_crs = (
        3031
        if bottom < -50
        else 3995 if bottom > 50 else get_local_utm(bounds.bounds, antimeridian=True)
    )
    logger.warning(f"Data will be returned in EPSG:{target_crs} projection")
    return target_crs


def split_bounds_at_antimeridian(
    bounds: BBox, lat_buff: float = 0
) -> tuple[BoundingBox]:
    """Split the bounds at the antimeridian, producing one set of bounds for the
    Eastern Hemisphere (left of the antimeridian) and one set for the Western
    Hemisphere (right of the antimeridian)

    Parameters
    ----------
    bounds : BBox (BoundingBox | tuple[float | int, float | int, float | int, float | int])
        The set of bounds (left, bottom, right, top)
    lat_buff : float, optional
        An additional buffer to subtract from lat, by default 0.

    Returns
    -------
    tuple[BoundingBox]
        A tuple containing two sets of bounds, one for the Eastern Hemisphere, one for
        the Western Hemisphere.
    """
    if isinstance(bounds, tuple):
        bounds = BoundingBox(*bounds)

    min_y = max(-90, bounds.bottom - lat_buff)
    max_y = min(90, bounds.top + lat_buff)

    bounds_western_hemisphere = BoundingBox(-180, min_y, bounds[2], max_y)
    bounds_eastern_hemisphere = BoundingBox(bounds[0], min_y, 180, max_y)

    logger.info(f"Eastern Hemisphere bounds: {bounds_eastern_hemisphere.bounds}")
    logger.info(f"Western Hemisphere bounds: {bounds_western_hemisphere.bounds}")

    return (bounds_eastern_hemisphere, bounds_western_hemisphere)


def get_all_lat_lon_coords(
    geom: shapely.geometry.Polygon | shapely.geometry.MultiPolygon,
) -> tuple[list, list]:
    """
    Extract all longitude (x) and latitude (y) coordinates from a Shapely Polygon or MultiPolygon.
    This function gathers coordinates from both the exterior boundary and any interior rings
    (holes) of each polygon in the input geometry.

    Parameters
    ----------
    geom : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        The input geometry to extract coordinates from.

    Returns
    -------
    longitudes : list
        list of x-coordinates (longitude values).
    latitudes : list
        list array of y-coordinates (latitude values).

    Raises
    ------
    TypeError
        If the input geometry is not a Polygon or MultiPolygon.
    """

    def coords_from_polygon(polygon):
        exterior = list(polygon.exterior.coords)
        interiors = [pt for interior in polygon.interiors for pt in interior.coords]
        return exterior + interiors

    if isinstance(geom, shapely.geometry.Polygon):
        coords = coords_from_polygon(geom)
    elif isinstance(geom, shapely.geometry.MultiPolygon):
        coords = [pt for poly in geom.geoms for pt in coords_from_polygon(poly)]
    else:
        raise TypeError("Geometry must be a Polygon or MultiPolygon")

    longitudes, latitudes = zip(*coords) if coords else ([], [])
    return list(longitudes), list(latitudes)


def adjust_bounds_at_high_lat(bounds: BBox) -> tuple:
    """Expand the bounds for high latitudes. The
    provided bounds sometimes do not contain the full scene due to
    warping at high latitudes. Solve this by converting bounds to polar
    stereographic, getting bounds, converting back to 4326. At high
    latitudes this will increase the longitude range.

    Parameters
    ----------
    bounds : BBox (BoundingBox | tuple[float | int, float | int, float | int, float | int])
        The set of bounds (left, bottom, right, top)

    Returns
    -------
    BoundingBox
        The expanded bounds (left, bottom, right, top)
    """
    if isinstance(bounds, tuple):
        bounds = BoundingBox(*bounds)

    if bounds.bottom < -50:
        logging.info(f"Adjusting bounds at high southern latitudes")
        bounds = adjust_bounds(bounds, src_crs=4326, ref_crs=3031)
    if bounds.bottom > 50:
        logging.info(f"Adjusting bounds at high northern latitudes")
        bounds = adjust_bounds(bounds, src_crs=4326, ref_crs=3995)

    return bounds


def resize_bounds(bounds: BoundingBox, scale_factor: float = 1.0) -> BoundingBox:
    """Resizes a bounding box

    Parameters
    ----------
    bounds : BoundingBox
        BoundingBox object.
    scale_factor : float, optional
        Factor to scale up or down the bounding box, by default 1.0

    Returns
    -------
    BoundingBox
        Resized bounding box.
    """
    x_dim = bounds.right - bounds.left
    y_dim = bounds.top - bounds.bottom

    dx = ((scale_factor - 1) * x_dim) / 2
    dy = ((scale_factor - 1) * y_dim) / 2

    return BoundingBox(
        bounds.left - dx, bounds.bottom - dy, bounds.right + dx, bounds.top + dy
    )


def crop_datasets_to_bounds(
    dem_rasters: list[Path], bounds: BBox, save_path: Path | None = None
) -> tuple[np.ndarray, Profile]:
    """Merges a list of datasets and crops the merged tiles to a given bounding box.

    Parameters
    ----------
    dem_rasters : list[Path]
        List of dataset paths.
    bounds : BBox
        BoundingBox object or tuple of coordinates.
    save_path : Path | None, optional
        Local path to save the output merged tile, by default None

    Returns
    -------
    tuple[np.ndarray, Profile]
        tuple of array of merged tile and its profile.
    """

    if type(bounds) == BoundingBox:
        bounds = bounds.bounds

    logger.info(f"Creating VRT")
    vrt_path = (
        str(save_path).replace(".tif", ".vrt") if save_path else "temp_dem_file.vrt"
    )  # Temporary VRT file path
    logger.info(f"VRT path = {vrt_path}")
    VRT_options = gdal.BuildVRTOptions(
        resolution="highest",
        outputBounds=bounds,
        VRTNodata=0,
    )
    ds = gdal.BuildVRT(vrt_path, dem_rasters, options=VRT_options)
    ds.FlushCache()

    with rasterio.open(vrt_path, "r", count=1) as src:
        dem_array, dem_transform = rasterio.mask.mask(
            src,
            [shapely.geometry.box(*bounds)],
            all_touched=True,
            crop=True,
        )
        # Using the masking adds an extra dimension from the read
        # Remove this by squeezing before writing
        dem_array = dem_array.squeeze()

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
        os.remove(vrt_path)

        if save_path:
            with rasterio.open(save_path, "w", **dem_profile) as dst:
                dst.write(dem_array, 1)
            # shutil.rmtree(
            #     dem_rasters[0].parts[0], ignore_errors=True
            # )

    return dem_array, dem_profile


def check_dem_type_in_bounds(
    dem_type: str, resolution: ValidDEMResolutions, bounds: BBox
) -> bool:
    """Check if the specified dem has data within the provided bounds. The provided dem_type is matched to either the
    Copernicus Global 30m DEM or REMA DEM (currently implemented options). True is returned if the provided bounds
    intersect with any tiles of the specified DEM.

    Parameters
    ----------
    dem_type : str
        dem type to check. Can be variations of REMA and COP. e.g. REMA_32, REMA_10,
        cop30m, cop_30m, cop_glo30.
    resolution: ValidDEMResolutions
        resolution of the dem. Required to read the correct layer of the GPKG.
    bounds : BBox
        Bounds to check if data exists

    Returns
    -------
    bool
        True if the bounds intersects a tile, False otherwise.

    Raises
    -------
    ValueError
        If the provided dem_type cannot be matched to either the Copernicus 30m global DEM or the REMA dem.
    """

    if isinstance(bounds, BoundingBox):
        bounds = bounds.bounds

    dem_type_match = dem_type.upper()
    if "COP" in dem_type_match and resolution in COP_VALID_RESOLUTIONS:
        dem_index_path = COP30_GPKG_PATH
        dem_type_formal = "Copernicus 30m global DEM"
        layer = None  # only one layer
    elif "REMA" in dem_type_match and resolution in REMA_VALID_RESOLUTIONS:
        dem_index_path = REMA_GPKG_PATH
        dem_type_formal = "REMA DEM"
        layer = f"REMA_Mosaic_Index_v2_{resolution}m"
    else:
        raise ValueError(
            f"DEM type `{dem_type}` and resolution {resolution} could not be matched to either the Copernicus 30m global DEM or a valid REMA DEM with resolutions of {REMA_VALID_RESOLUTIONS}"
        )

    logger.info(f"Checking if bounds intersect with tiles of the {dem_type_formal}")
    logger.info(f"Searching {dem_index_path}")

    gdf = gpd.read_file(dem_index_path, layer=layer)
    bounding_box = shapely.geometry.box(*bounds)

    if gdf.crs is not None:
        # ensure same crs
        bounding_box = (
            gpd.GeoSeries([bounding_box], crs="EPSG:4326").to_crs(gdf.crs).iloc[0]
        )
    else:
        logger.info('No crs found for index file. Assuming EPSG:4326"')
        bounding_box = gpd.GeoSeries([bounding_box], crs="EPSG:4326")
    # Find rows that intersect with the bounding box
    intersecting_tiles = gdf[gdf.intersects(bounding_box)]
    if len(intersecting_tiles) == 0:
        logger.info(f"No intersecting tiles found")
        return False
    else:
        logger.info(f"{len(intersecting_tiles)} intersecting tiles found")
        return True
