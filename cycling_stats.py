import configparser
from pathlib import Path
from datetime import datetime, timedelta

import pydantic
import numpy as np
import pandas as pd
import geopy.distance
import fitparse
from pyproj import Transformer

from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import FuncTickFormatter
from bokeh.layouts import gridplot
from bokeh.palettes import Plasma11
from bokeh.transform import linear_cmap
from bokeh.tile_providers import get_provider

import xyzservices.providers as xyz


class Ride(pydantic.BaseModel):
    name: str
    distance: float
    moving_time: float
    avg_speed: float
    max_speed: float
    ascent: float
    descent: float
    datetimes: list[datetime]
    coordinates: list[tuple[float, float]]
    major_version: int = 1
    minor_version: int = 0


def parse_fit_file(fname: str) -> Ride:
    fitfile = fitparse.FitFile(fname)

    time = []
    dist = []
    speed = []
    alt = []
    coords = []

    # factor to convert from semicircles to degrees
    semi_circ_to_deg = (180 / 2**31)

    # Get all data messages that are of type record
    for record in fitfile.get_messages('record'):
        field_names = [x.name for x in record.fields]

        if 'position_long' in field_names:
            time.append(record.get('timestamp').value)
            if 'enhanced_altitude' in field_names:
                alt_field_name = 'enhanced_altitude'
            else:
                alt_field_name = 'altitude'
            alt_unit = record.get(alt_field_name).units
            if alt_unit == 'm':
                alt.append(record.get(alt_field_name).value / 1000)
            else:
                raise NotImplementedError

            if 'enhanced_speed' in field_names:
                speed_field_name = 'enhanced_speed'
            else:
                speed_field_name = 'speed'
            speed_unit = record.get(speed_field_name).units
            if speed_unit == 'm/s':
                speed.append(record.get(speed_field_name).value * 3.6)
            else:
                raise NotImplementedError

            coords.append(
                (record.get('position_lat').value * semi_circ_to_deg,
                 record.get('position_long').value * semi_circ_to_deg))

    sm_kernel = np.ones(15) / 15

    alt = np.array(alt)
    alt_sm = np.convolve(alt, sm_kernel, mode='same')
    speed = np.array(speed)

    # calculate difference between time tags
    time = np.array(time)
    time_delta = np.array([x.seconds for x in (time[1:] - time[:-1])])

    dist = geopy.distance.geodesic(*coords).km

    # calculate ascent and descent
    alt_diff = alt_sm[1:] - alt_sm[:-1]
    ascent = alt_diff[alt_diff >= 0].sum()
    descent = alt_diff[alt_diff < 0].sum()

    # calculate moving time
    t_mov = time_delta[speed[:-1] > 4].sum() / 60
    avg_speed = dist / (t_mov / 60)

    # number of datapoints
    num_p = len(coords)
    # number of elements to store
    num_e = min(20, len(coords))

    sub_coords = coords[::(num_p // num_e)][:num_e]
    sub_times = time.tolist()[::(num_p // num_e)][:num_e]

    ride = Ride(name=Path(fname).stem.split('__')[1],
                distance=dist,
                moving_time=t_mov,
                avg_speed=avg_speed,
                max_speed=speed.max(),
                ascent=ascent,
                descent=descent,
                datetimes=sub_times,
                coordinates=sub_coords)

    return ride


class RideStats:

    def __init__(self, fnames: list[Path]) -> None:
        self.fnames = fnames
        self.rides = []

        for fname in self.fnames:
            # commuting file that was not renamed
            if not fname.stem.startswith('20'):
                fitfile = fitparse.FitFile(str(fname))
                for i, record in enumerate(fitfile.get_messages('record')):
                    if i == 0:
                        date = record.get('timestamp').value.strftime('%Y%m%d')
                        break

                new_filename = fname.parent / f'{date}__commute.FIT'

                i = 2
                while new_filename.exists():
                    new_filename = new_filename.parent / f'{new_filename.stem}_{i}{new_filename.suffix}'

                print(f'moving {fname} {new_filename}')
                fname.rename(new_filename)
                fname = new_filename

            preprocessed_fname = fname.with_suffix('.json')

            if not preprocessed_fname.exists():
                print(f'parsing {fname}')
                ride = parse_fit_file(str(fname))
                with open(preprocessed_fname, 'w') as f:
                    f.write(ride.json())
            else:
                print(f'loading {preprocessed_fname}')
                ride = Ride.parse_file(preprocessed_fname)

            self.rides.append(ride)

    @property
    def df(self) -> pd.DataFrame:
        df = pd.DataFrame([x.__dict__ for x in self.rides])
        return df


def bokeh_cycling_stats(df, output_html_file):
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
                 tooltips=[('name', "@{name}"),
                           ('distance [km]', "@{distance}"),
                           ('ascent [km]', "@{ascent}"),
                           ('moving time [min]', "@{moving_time}"),
                           ('avg speed [km/h]', "@{avg_speed}"),
                           ('date', "@{date}")])
    p40.scatter('grad',
                'avg_speed',
                source=df,
                size=8,
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


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    rs = RideStats(
        sorted(
            list(
                Path(config['fileio']['datapath']).rglob(
                    config['fileio']['pattern']))))

    bokeh_cycling_stats(rs.df, 'cycling_stats.html')