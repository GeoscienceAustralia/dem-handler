import typing
import warnings
from pathlib import Path

from dem_handler._version import __version__

DATA_DIR = Path(__file__).parent / Path("data")
REMA_GPKG_PATH = DATA_DIR / Path("REMA_Mosaic_Index_v2.gpkg")
COP30_GPKG_PATH = DATA_DIR / Path("copdem_tindex_filename.gpkg")

REMAResolutions = typing.Literal[2, 10, 32]
COPResolutions = typing.Literal[30]
ValidDEMResolutions = typing.Literal[REMAResolutions, COPResolutions]

REMA_VALID_RESOLUTIONS = typing.get_args(REMAResolutions)
COP_VALID_RESOLUTIONS = typing.get_args(COPResolutions)

# Suppress the GIL warning about the '_brotli' module specifically.
warnings.filterwarnings(
    "ignore",
    message="The global interpreter lock (GIL) has been enabled to load module '_brotli', which has not declared that it can run safely without the GIL",
    category=RuntimeWarning,
)  # This warning is triggered by the use of multiprocessing in the AWS utility functions. It can be safely ignored in this context.
