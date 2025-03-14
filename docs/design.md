# Design

This package was designed to overcome issues that we encountered while using other python packages for DEM handling. 

## Components
* Location of DEM tiles (cloud or local).
* A [GDAL tile index file](https://gdal.org/en/stable/drivers/raster/gti.html) containing the file names and tile bounds for all DEM tiles.
* Location of geoid model files (cloud or local).

## Approach
The following workflow is applied for a given set of bounds:

* The tile index is searched for all tiles intersecting the given bounds
* A [GDAL virtual raster tile (VRT)](https://gdal.org/en/stable/drivers/raster/vrt.html) is built for the identified tiles, and set to have a total extent that matches the given bounds.
    * Any nodata pixels are given a value of 0.
    * If no tiles are found, a raster is created that matches the expected DEM resolution, filled with 0s.
* The VRT (or raster if no tiles were found) is then read for the requested bounds and can be used in-memory or written to a file.
* The geoid height model is read and subtracted if requested to return height above ellipsoid.

Values of 0 are used in place of nodata in the VRT to model the height above geoid of the ocean. 
It is assumed in our approach that all nodata values (within tiles, or outside tiles) are ocean. 

## DEM over ocean
It is common practice for tiled DEMs to only exist over land, with nodata values for any ocean areas within land tiles. 
While this makes sense for terrestrial applications, it is problematic for applications such as monitoring sea ice, where a DEM that covers the study area is required to produce analysis-ready synthetic aperture radar (SAR) data for scenes containing sea ice.

By building a VRT file, we are able to handle ocean areas.
The functions read from the created VRT, naturally filling the ocean with 0, representing the height above the geoid for the ocean. 

## DEM over the antimeridian
Analysis and data processing over the Arctic, Antarctic, and Pacific Islands can require DEMs that cross the antimeridian. 
For global DEMs provided in latitude and longitude coordinates (EPSG:4326), such as the Copernicus Global 30m DEM, there is a discontinuity at the antimeridian that causes challenges in creating seamless mosaicked DEMs. 

In our approach, if the requested bounds cross the antimeridian, we split the request into bounds for the Eastern Hemisphere and bounds for the Western Hemisphere. 
These bounds are each processed as normal, producing two mosaicked DEMs, one for the bounds in each Hemisphere.
The two DEMs are then transformed to an appropriate coordinate reference system that doesn't have a discontinuity namely:
* polar stereographic coordinate reference systems for the Arctic and Antarctic,
* Local UTM coordinate reference system otherwise.
The transformed DEMs from each hemisphere are then combined to produce on DEM in the appropriate coordinate reference system

## Ability to get ellipsoidal heights
Synthetic aperture radar processing requires height above the ellipsoid, rather than above the geoid. 

Our functions have an optional argument to provide a geoid model file, or download one. 
This can then be subtracted from the height above the geoid to provide the height above the ellipsoid.

## DEM tiles with different resolutions
To deal with distortion at high latitudes, the Copernicus Global 30m DEM [publishes DEM tiles at different resolutions depending on the latitude band](https://copernicus-dem-30m.s3.amazonaws.com/readme.html).
This is particularly challenging for Arctic and Antarctic work, where the DEM resolution changes in 10&deg; bands between 50&deg; and 80&deg; latitude, and in 5&deg; bands between 80&deg; and 90&deg; latitude.
By using a VRT, we are able to naturally handle these transitions, as the VRT is set to resample the DEM to the highest resolution if multiple resolutions are provided. 
This ensures that a DEM with uniform resolution is generated for any arbitrary bounds, even if it crosses a latitude boundary and requests DEM tiles at differing resolutions.

## Point vs Area DEMs
DEM tiles can be supplied in Point convention (the coordinate refers to the centre of a pixel) or Area convention (the coordinate refers to the top-left corner of the pixel). 
If not handled appropriately, Point convention DEMs can end up with half-pixel offsets after mosaicking and writing to Area convention. 

Our functions implement special handling for DEMs in Point convention (such as the Copernicus Global DEMs), and ensure the final mosaic is properly located after writing out in Area convention. 

## Supporting local processing
We have workflows that run on supercomputers with limited internet access, as well as in the cloud.
As such, we needed the same functionality to work whether the DEM tiles existed locally, or in the cloud. 

Our functions have an option to "download" the tiles, which will work for local solutions with internet access or cloud. 
Our functions can also take a local path to a folder that contains the DEM tiles, meaning no internet access is required if the tiles are already available on the system.
As mentioned in the [Approach](#approach) section, we use a tile index to find the correct tiles for given bounds.
The tile index contains only file names (no paths), so can be used to query tiles for either local or cloud processing.

Our high level functions for specific DEMs are already configured to pull from appropriate cloud repositories containing the authoritative data. 