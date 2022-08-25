from typing import Optional

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.io.img_tiles import Stamen
import geopy.distance

from utils import Ride


def ride_to_figure(ride: Ride, zoom_level: Optional[int] = None) -> plt.figure:
    lat = [x[0] for x in ride.coordinates]
    lon = [x[1] for x in ride.coordinates]

    dlat = max(lat) - min(lat)
    dlon = max(lon) - min(lon)

    min_lat = min(lat) - 0.1 * dlat
    max_lat = max(lat) + 0.1 * dlat
    min_lon = min(lon) - 0.1 * dlon
    max_lon = max(lon) + 0.1 * dlon

    diag_dist = geopy.distance.distance((min_lat, min_lon),
                                        (max_lat, max_lon)).km

    if zoom_level is None:
        zoom_level = int(-0.1 * (diag_dist - 3.4) + 16)
        zoom_level = max(1, zoom_level)
        zoom_level = min(20, zoom_level)

    tiler = cimgt.OSM()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=tiler.crs)
    ax.set_extent([min_lon, min_lat, max_lon, max_lat], crs=ccrs.PlateCarree())
    ax.add_image(tiler, zoom_level)
    ax.plot(lon, lat, 'r.', transform=ccrs.PlateCarree())
    ax.plot([lon[0]], [lat[0]], 'b.', transform=ccrs.PlateCarree())
    ax.plot([lon[-1]], [lat[-1]], 'k.', transform=ccrs.PlateCarree())
    ax.set_axis_off()

    fig.tight_layout()

    return fig