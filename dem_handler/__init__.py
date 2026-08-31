import typing
from pathlib import Path

from dem_handler._version import __version__

DATA_DIR = Path(__file__).parent / Path("data")
REMA_GPKG_PATH = DATA_DIR / Path("REMA_Mosaic_Index_v2.gpkg")
COP30_GPKG_PATH = DATA_DIR / Path("copdem_tindex_filename.gpkg")
REMA_SERIES_CONFIG_PATH = DATA_DIR / Path("rema_series_config.json")

REMAResolutions = typing.Literal[2, 10, 30, 32]
COPResolutions = typing.Literal[30]
ValidDEMResolutions = typing.Literal[REMAResolutions, COPResolutions]

REMABoundCheckSkipResolutions = typing.Literal[30]

REMA_VALID_RESOLUTIONS = typing.get_args(REMAResolutions)
COP_VALID_RESOLUTIONS = typing.get_args(COPResolutions)
REMA_BOUND_CHECK_SKIP_RESOLUTIONS = typing.get_args(REMABoundCheckSkipResolutions)
