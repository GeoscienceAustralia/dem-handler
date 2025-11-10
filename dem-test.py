import os
from pathlib import Path
from dem_handler.dem.cop_glo30 import get_cop30_dem_for_bounds
from dem_handler.dem.rema import get_rema_dem_for_bounds
from dem_handler.utils.spatial import resize_bounds, BoundingBox
import logging

logging.basicConfig(level=logging.INFO)

bounds = (-177.884048, -78.176201, 178.838364, -75.697151)
bounds = (178.5, -78.1, -179.1, -77.6)
# bounds = (178.0, -78.5, -178, -77.2)
# bounds = (163.126465, -78.632896, 172.387283, -76.383476)
# bounds = (165.75, -77.07, 167.40, -76.53)
# bounds = (162.5, -70.4, 163.0, -70.0)  # data
# bounds = (
#     143.6423611111111143,
#     -17.7573611111111092,
#     146.5748611111111188,
#     -15.3943055555555546,
# )
print(bounds)
dem_type = "cop_glo30"
dem_type = "REMA_32"
dem_folder = f"/data/working/downloads/dem/{dem_type}"

# # The copernicus Global 30m DEM

# dem_name = dem_type + "_" + "_".join([str(round(x, 2)) for x in bounds])
# dem_name = "Copernicus_DSM_COG_10_antimeridian"
# # dem_name = "TMP"
# DEM_PATH = f"{dem_folder}/{dem_name}.tif"

# arr, profile, _ = get_cop30_dem_for_bounds(
#     bounds=bounds,
#     save_path=DEM_PATH,
#     ellipsoid_heights=False,
#     adjust_at_high_lat=False,
#     buffer_pixels=None,
#     # buffer_degrees=0.3,
#     cop30_folder_path=dem_folder,
#     geoid_tif_path=f"{dem_folder}/geoid.tif",
#     download_dem_tiles=True,
#     download_geoid=False,
# )


# rema dem

# get_rema_dem_for_bounds(
#     bounds=bounds,
#     bounds_src_crs=4326,
#     save_path=DEM_PATH,
#     resolution=32,
#     ellipsoid_heights=True,
#     download_geoid=True,
#     download_dir=Path(dem_folder),
#     # buffer_pixels=1,
# )

# from dem_handler.dem.rema import get_rema_dem_for_bounds, BBox
# from dem_handler.utils.spatial import resize_bounds, BoundingBox, transform_polygon

# bbox = BoundingBox(67.45, -72.55, 67.55, -72.45)
# bounds = resize_bounds(bbox, 10.0)
# dem_name = "rema_32m_four_tiles_ellipsoid_heights_fixed.tif"

# get_rema_dem_for_bounds(
#     bounds=bounds,
#     bounds_src_crs=4326,
#     save_path=dem_name,
#     resolution=32,
#     ellipsoid_heights=True,
#     download_geoid=True,
#     download_dir=".",
#     buffer_pixels=1,
# )

# from dem_handler.download.aws import read_rema_timeseries_vrt

# year = 2020
year = None
# # read_rema_timeseries_vrt(year, aws_access_key_id, aws_secret_access_key)

# # testing the rema timeseries over thwaites glacier
xmin = -1494005
ymin = -562_200
xmax = xmin + 2_000
ymax = ymin + 2_000

resolution = 10

bounds = (xmin, ymin, xmax, ymax)
bounds_str = "_".join([str(round(x, 2)) for x in bounds])

get_rema_dem_for_bounds(
    bounds=bounds,
    rema_year=year,
    bounds_src_crs=3031,
    save_path=f"{year}_{resolution}_{bounds_str}.tif",
    resolution=resolution,
    ellipsoid_heights=True,
    download_geoid=True,
    download_dir=".",
    buffer_pixels=1,
)


# bounds = (
#     -113.5402505966611,
#     -75.11750388746809,
#     -111.2636133622894,
#     -74.35496942539352,
# )
# bounds_str = "_".join([str(round(x, 2)) for x in bounds])

# get_rema_dem_for_bounds(
#     bounds=bounds,
#     bounds_src_crs=4326,
#     # save_path=f"{year}_{bounds_str}.tif",
#     save_path=f"{year}_full.tif",
#     resolution=32,
#     ellipsoid_heights=True,
#     download_geoid=True,
#     download_dir=".",
#     buffer_pixels=1,
#     timeseries=True,
#     year=year,
#     aws_access_key_id=aws_access_key_id,
#     aws_secret_access_key=aws_secret_access_key,
# )
