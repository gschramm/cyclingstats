from typing import Optional
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.io.img_tiles import Stamen
import geopy.distance

from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import FuncTickFormatter
from bokeh.layouts import gridplot
from bokeh.palettes import Plasma11
from bokeh.transform import linear_cmap
from bokeh.tile_providers import get_provider

from pyproj import Transformer
import xyzservices.providers as xyz

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

    asp = (max_lat - min_lat) / (max_lon - min_lon)

    diag_dist = geopy.distance.distance((min_lat, min_lon),
                                        (max_lat, max_lon)).km

    if zoom_level is None:
        zoom_level = int(-0.1 * (diag_dist - 3.4) + 16)
        zoom_level = max(1, zoom_level)
        zoom_level = min(20, zoom_level)

    tiler = cimgt.OSM()

    if asp > 1:
        fig_width = 7
        fig_height = asp * fig_width
    else:
        fig_height = 7
        fig_width = fig_height / asp

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(1, 1, 1, projection=tiler.crs)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.add_image(tiler, zoom_level)
    ax.plot(lon, lat, 'r.', transform=ccrs.PlateCarree())
    for i, txt in enumerate(np.arange(1, len(lon) + 1)):
        ax.text(lon[i],
                lat[i],
                str(txt),
                transform=ccrs.PlateCarree(),
                fontsize='xx-small')
    ax.set_axis_off()

    fig.tight_layout()

    return fig


def bokeh_cycling_stats(df: pd.DataFrame, output_html_file: str):
    # add a columns to the data frame that we need for stats and plotting
    df['grad'] = df['ascent'] / df['distance']
    # date string for tooltips
    df['datetime'] = [x[0] for x in df.datetimes]
    df['date'] = [x.date().strftime("%y-%m-%d") for x in df.datetime]

    output_file(output_html_file,
                title='cycling stats ' +
                df.iloc[-1, :].datetime.strftime('%Y-%m-%d'))

    #--- create weekly, monthly and yearly stats
    weekly_stats = df.resample('W', on='datetime').sum()
    monthly_stats = df.resample('M', on='datetime').sum()
    yearly_stats = df.resample('Y', on='datetime').sum()

    weekly_stats['cat'] = [(x - timedelta(days=x.weekday())).strftime('%y-%V')
                           for x in weekly_stats.index]
    monthly_stats['cat'] = [x.strftime('%y-%m') for x in monthly_stats.index]
    yearly_stats['cat'] = [str(x.year) for x in yearly_stats.index]

    formatter = FuncTickFormatter(code="""
    if (index % 4 == 0)
    {
        return tick;
    }
    else
    {
        return "";
    }
    """)

    p00 = figure(title="weekly distance [km]",
                 x_range=weekly_stats['cat'],
                 tooltips=[('week', "@{cat}"),
                           ('distance [km]', "@{distance}")])
    p00.vbar(x='cat',
             top='distance',
             width=0.7,
             source=weekly_stats,
             line_width=0)
    p00.xaxis.formatter = formatter
    p00.xaxis.major_label_orientation = np.pi / 2

    p10 = figure(title="monthly distance [km]",
                 x_range=monthly_stats['cat'],
                 tooltips=[('month', "@{cat}"),
                           ('distance [km]', "@{distance}")])
    p10.vbar(x='cat',
             top='distance',
             width=0.7,
             source=monthly_stats,
             line_width=0)
    p10.xaxis.major_label_orientation = np.pi / 2

    p20 = figure(title="yearly distance [km]",
                 x_range=yearly_stats['cat'],
                 tooltips=[('year', "@{cat}"),
                           ('distance [km]', "@{distance}")])
    p20.vbar(x='cat',
             top='distance',
             width=0.7,
             source=yearly_stats,
             line_width=0)

    p01 = figure(title="weekly ascent [km]",
                 x_range=weekly_stats['cat'],
                 tooltips=[('week', "@{cat}"), ('ascent [km]', "@{ascent}")])
    p01.vbar(x='cat',
             top='ascent',
             width=0.7,
             source=weekly_stats,
             fill_color='darkorange',
             line_width=0)
    p01.xaxis.formatter = formatter
    p01.xaxis.major_label_orientation = np.pi / 2

    p11 = figure(title="monthly ascent [km]",
                 x_range=monthly_stats['cat'],
                 tooltips=[('month', "@{cat}"), ('ascent [km]', "@{ascent}")])
    p11.xaxis.major_label_orientation = np.pi / 2

    p11.vbar(x='cat',
             top='ascent',
             width=0.7,
             source=monthly_stats,
             fill_color='darkorange',
             line_width=0)

    p21 = figure(title="yearly ascent",
                 x_range=yearly_stats['cat'],
                 tooltips=[('year', "@{cat}"), ('ascent [km]', "@{ascent}")])
    p21.vbar(x='cat',
             top='ascent',
             width=0.7,
             source=yearly_stats,
             fill_color='darkorange',
             line_width=0)

    # plot histograms about rides
    dist_histo = np.histogram(
        df["distance"],
        bins=np.arange(np.ceil(df['distance'].max() / 10) + 2) * 10 - 5)
    p30 = figure(title='ride distance [km] histogram')
    p30.quad(top=dist_histo[0],
             bottom=0,
             left=dist_histo[1][:-1],
             right=dist_histo[1][1:],
             fill_color="darkseagreen",
             line_width=0)

    mt_histo = np.histogram(
        df["moving_time"],
        bins=np.arange(np.ceil(df['moving_time'].max() / 20) + 2) * 20 - 10)
    p31 = figure(title='ride moving time [min] histogram')
    p31.quad(top=mt_histo[0],
             bottom=0,
             left=mt_histo[1][:-1],
             right=mt_histo[1][1:],
             fill_color="darkseagreen",
             line_width=0)

    p40 = figure(title='ride avg speed [km/h] vs (ascent / distance)',
                 tooltips=[('distance [km]', "@{distance}"),
                           ('ascent [km]', "@{ascent}"),
                           ('moving time [min]', "@{moving_time}"),
                           ('avg speed [km/h]', "@{avg_speed}"),
                           ('date', "@{date}")])
    p40.scatter('grad',
                'avg_speed',
                source=df,
                size=6,
                color=linear_cmap(field_name='ascent',
                                  palette=Plasma11,
                                  low=df['ascent'].min(),
                                  high=1.1 * df['ascent'].max()))

    tile_provider = get_provider(xyz.OpenStreetMap.Mapnik)

    lat = np.array([x[0][0] for x in df.coordinates])
    lon = np.array([x[0][1] for x in df.coordinates])

    # convert first lat/lon into mercator coordinates
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
    mlon, mlat = transformer.transform(lat, lon)

    source = ColumnDataSource(data=dict(lat=mlat, lon=mlon))

    p41 = figure(x_axis_type="mercator", y_axis_type="mercator", title='map')
    p41.add_tile(tile_provider)

    p41.circle(x="lon",
               y="lat",
               size=6,
               fill_color="red",
               fill_alpha=0.5,
               source=source)

    for fig in [p00, p01, p10, p11, p20, p21, p30, p31, p40, p41]:
        fig.toolbar.active_drag = None
        fig.toolbar.active_scroll = None
        fig.toolbar.active_tap = None

    # group all figures in a grid
    grid = gridplot(
        [[p00, p01], [p10, p11], [p20, p21], [p30, p31], [p40, p41]],
        merge_tools=False,
        width=600,
        height=250,
        sizing_mode='scale_both')
    show(grid)
