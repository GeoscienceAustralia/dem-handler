from dem_handler._version import __version__
from pathlib import Path

DATA_DIR = Path(__file__).parent / Path("data")
REMA_GPKG_PATH = DATA_DIR / Path("REMA_Mosaic_Index_v2.gpkg")
COP30_GPKG_PATH = DATA_DIR / Path("copdem_tindex_filename.gpkg")
REMA_VALID_RESOLUTIONS = [2, 10, 32]
