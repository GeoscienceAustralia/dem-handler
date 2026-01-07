from __future__ import annotations

"""
inspired by https://github.com/ACCESS-Cloud-Based-InSAR/dem-stitcher/blob/dev/src/dem_stitcher/geoid.py
"""

import os
from pathlib import Path
import logging
from typing import Literal

import numpy as np
import rasterio
import rasterio.transform
import shapely.geometry
from dem_handler.utils.raster import read_raster_with_bounds
from dem_handler.utils.rio_tools import reproject_arr_to_match_profile
from dem_handler.utils.spatial import transform_polygon


def read_geoid(
    geoid_path: str | Path, bounds: tuple, buffer_pixels: int = 0
) -> tuple[np.ndarray, dict]:
    """Read in the geoid for the bounds provided with a specified buffer.

    Parameters
    ----------
    geoid_path : str | Path
        Path to the GEOID file
    bounds : tuple
        the set of bounds (left, bottom, right, top)
    buffer_pixels : int, optional
        additional pixels to buffern around bounds, by default 0

    Returns
    -------
    tuple [np.darray, dict]
        geoid array and geoid rasterio profile

    Raises
    ------
    FileNotFoundError
        If ther GEOID file cannot be found
    """

    if not Path(geoid_path).exists():
        raise FileNotFoundError(f"Geoid file does not exist at path: {geoid_path}")

    geoid_arr, geoid_profile = read_raster_with_bounds(
        geoid_path, bounds, buffer_pixels=buffer_pixels
    )
    geoid_arr = geoid_arr.astype("float32")
    geoid_arr[geoid_profile["nodata"] == geoid_arr] = np.nan
    geoid_profile["nodata"] = np.nan

    return geoid_arr, geoid_profile


def apply_geoid(
    dem_array: np.ndarray,
    dem_profile: dict,
    geoid_path=str | Path,
    buffer_pixels: int = 2,
    dem_mask_buffer: int = 0,
    save_path: str | Path = "",
    mask_array: np.ndarray | None = None,
    method: Literal["add", "subtract"] = "add",
) -> np.ndarray:
    """Add or subtract geoid heights from a dem_array

    Parameters
    ----------
    dem_array : np.ndarray
        Array containing dem values
    dem_profile : dict
        Profile associated with dem_array
    geoid_path : str | Path , optional
        Path to the geoid file
    buffer_pixels : int, optional
        Additional pixel buffer for geoid, by default 2.
    dem_mask_buffer : int, optional
        An additional buffer for the mask that gets applied when
        reading the geoid. The mask is based on the dem bounds. Value
        is in geographic units. e.g. degrees for 4326 and metres for 3031.
    save_path : str | Path, optional
        Location to save dem, by default "".
    mask_array : np.ndarray | None, optional
        Boolean array with same shape as the dem. Used to apply the
        geoid to a specific part of the dem. by default None, and the geoid
        is applied to the entire dem
    method : Literal["add", "subtract"], optional
        Method to either add or subtract the geoid, by default "add".

    Returns
    -------
    np.ndarray
        dem array with the heights adjusted with the geoid values.
    """

    if method not in ["add", "subtract"]:
        raise ValueError('Apply geoid method must be "add" or "subtract".')

    dem_transform = dem_profile["transform"]
    dem_res = max(dem_transform.a, abs(dem_transform.e))
    dem_bounds = rasterio.transform.array_bounds(
        dem_profile["height"], dem_profile["width"], dem_transform
    )

    with rasterio.open(geoid_path, "r") as src:
        mask_dem_bounds = dem_bounds
        geoid_crs = src.crs.to_epsg()
        dem_crs = dem_profile["crs"].to_epsg()
        if geoid_crs != dem_crs:
            mask_dem_bounds = transform_polygon(
                shapely.geometry.box(*dem_bounds).buffer(dem_mask_buffer),
                dem_crs,
                geoid_crs,
            ).bounds
        geoid_array, geoid_transform = rasterio.mask.mask(
            src,
            [shapely.geometry.box(*mask_dem_bounds).buffer(dem_mask_buffer)],
            all_touched=True,
            crop=True,
            pad=True,
            pad_width=buffer_pixels,
        )

        geoid_profile = src.profile
        geoid_profile.update(
            {
                "height": geoid_array.shape[1],
                "width": geoid_array.shape[2],
                "transform": geoid_transform,
            }
        )

    # reproject the geoid to match the dem
    geoid_reprojected, geoid_reprojected_profile = reproject_arr_to_match_profile(
        geoid_array, geoid_profile, dem_profile, resampling="bilinear"
    )
    geoid_reprojected = np.squeeze(geoid_reprojected)

    geoid_reprojected_transform = geoid_reprojected_profile["transform"]

    geoid_res = max(geoid_reprojected_transform.a, abs(geoid_reprojected_transform.e))

    if geoid_res * buffer_pixels <= dem_res:
        buffer_recommendation = int(np.ceil(dem_res / geoid_res))
        warning = (
            "The dem resolution is larger than the geoid resolution and its buffer; "
            "Edges resampled with bilinear interpolation will be inconsistent so select larger buffer. "
            f"Recommended : `buffer_pixels = {buffer_recommendation}`"
        )
        logging.warning(warning)

    if mask_array is None:
        # no mask, apply geoid to whole dem
        if method == "add":
            dem_arr_offset = dem_array + geoid_reprojected
        elif method == "subtract":
            dem_arr_offset = dem_array - geoid_reprojected
    else:
        # apply geoid to a subset of the dem
        dem_arr_offset = dem_array.copy()
        if method == "add":
            dem_arr_offset[mask_array] += geoid_reprojected[mask_array]
        elif method == "subtract":
            dem_arr_offset[mask_array] -= geoid_reprojected[mask_array]

    if save_path:
        with rasterio.open(save_path, "w", **dem_profile) as dst:
            dst.write(dem_arr_offset, 1)

    return dem_arr_offset
