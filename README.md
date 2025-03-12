# dem-handler
Utility package for handling various Digital Elevation Models (DEMs). 
The package enables access to data stored in cloud as well as local copies of dem datasets. 

## Functionality

The core functionality of this package is to provide mosaicked DEMs for arbitrary bounds.
This is valuable for creating a mosaicked DEM that covers a scene. 
The package provides high level functions for [supported DEMs](#supported-dems), and 
low level functions that can be used to handle custom DEMs. 

The DEM mosaicking functions have the following features:
* When requesting a DEM for bounds that include the ocean, the mosaicked DEM will 
include the ocean, setting the value of non-land pixels to 0 (height above the geoid).
* If a geoid height model is provided, the height above the ellipsoid can be returned.
* The functions work for DEM tiles stored in the cloud or locally.
* The mosaicked DEM can be returned in memory, as well as saved to a file for reuse. 
* When a DEM has different resolutions, the mosaicked DEM will be returned with the highest resolution.
* If the DEM is requested over the antimeridian, the request will be split into Eastern 
and Western hemisphere components, then merged back together in an appropriate local coordinate reference system.

For more information on how the above functionality was implemented, 
see the [design documentation](docs/design.md).

## Supported DEMS
- Copernicus Global 30m (cop_glo30)
- REMA (2m,10m, ...)

## Usage
### Create mosaicked DEM for bounds from cloud files

```python
import os
from dem_handler.dem.cop_glo30 import get_cop30_dem_for_bounds
from dem_handler.dem.rema import get_rema_dem_for_bounds

import logging
logging.basicConfig(level=logging.INFO)

#Set the bounds and make a directory for the files to download

bounds = (72,-70, 73, -69)
save_dir = 'TMP'
os.makedirs(save_dir, exist_ok=True)

#The copernicus Global 30m DEM 

get_cop30_dem_for_bounds(
    bounds = bounds,
    save_path = f'{save_dir}/cop_glo30.tif',
    adjust_at_high_lat=False,
    cop30_folder_path = save_dir,
    ellipsoid_heights = False,
    download_dem_tiles = True
)

# The REMA DEM (32m)

get_rema_dem_for_bounds(
    bounds=bounds,
    save_path=f'{save_dir}/rema.tif',
    resolution=32,
    bounds_src_crs=4326,
)
```

## Developer Setup

```bash
git clone ...
conda create --file environment.yaml
conda activate dem-handler
pip install -e .
```

Test install

```bash
pytest
```

## Contact
...