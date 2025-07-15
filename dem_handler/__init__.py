from dem_handler._version import __version__
from pathlib import Path
import typing

DATA_DIR = Path(__file__).parent / Path("data")
REMA_GPKG_PATH = DATA_DIR / Path("REMA_Mosaic_Index_v2.gpkg")
COP30_GPKG_PATH = DATA_DIR / Path("copdem_tindex_filename.gpkg")

REMAResolutions = typing.Literal[2, 10, 32]
COPResolutions = typing.Literal[30]
ValidDEMResolutions = typing.Literal[REMAResolutions, COPResolutions]

REMA_VALID_RESOLUTIONS = typing.get_args(REMAResolutions)
COP_VALID_RESOLUTIONS = typing.get_args(COPResolutions)
