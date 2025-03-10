import os
import zipfile
from urllib.request import urlretrieve
import glob
from pathlib import Path


def get_rema_index_file(save_folder: Path) -> Path:
    """Retrieves REMA DEMs index file.
    Parameters
    ----------
    save_folder : Path
        Folder to save the downloaded file

    Returns
    -------
    Path
        Local path to the index file.
    """

    REMA_INDEX_URL = "https://data.pgc.umn.edu/elev/dem/setsm/REMA/indexes/REMA_Mosaic_Index_latest_gpkg.zip"
    rema_index_filename = os.path.basename(REMA_INDEX_URL)
    # download and store locally

    save_folder.mkdir(exist_ok=True)
    zip_save_path = str(save_folder / rema_index_filename)
    urlretrieve(REMA_INDEX_URL, zip_save_path)
    # unzip
    with zipfile.ZipFile(zip_save_path, "r") as zip_ref:
        zip_ref.extractall(save_folder)
    os.remove(zip_save_path)
    rema_index_path = glob.glob(f"{save_folder}/**")[0]
    return Path(rema_index_path)
