import configparser
import dataclasses
from pathlib import Path
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import geopy.distance
import fitparse
from pyproj import Transformer

from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Band, FuncTickFormatter
from bokeh.layouts import gridplot
from bokeh.palettes import Plasma11
from bokeh.transform import linear_cmap
from bokeh.tile_providers import get_provider

import xyzservices.providers as xyz


@dataclasses.dataclass
class Ride:
    name: str = ''
    year: int = 1970
    month: int = 1
    week: int = 1
    day: int = 1
    hour: int = 1
    minute: int = 0
    distance: float = 0
    moving_time: float = 0
    avg_speed: float = 0
    max_speed: float = 0
    ascent: float = 0
    descent: float = 0
    start_lat: float = 0
    start_lon: float = 0
    end_lat: float = 0
    end_lon: float = 0

    @property
    def __dict__(self):
        return dataclasses.asdict(self)

    def to_json(self, fname: Path):
        with open(fname, 'w') as f:
            json.dump(self.__dict__, f)

    def from_json(self, fname: Path):
        with open(fname, 'r') as f:
            rdict = json.load(f)
        self.name = rdict['name']
        self.year = rdict['year']
        self.month = rdict['month']
        self.week = rdict['week']
        self.day = rdict['day']
        self.hour = rdict['hour']
        self.minute = rdict['minute']
        self.distance = rdict['distance']
        self.moving_time = rdict['moving_time']
        self.avg_speed = rdict['avg_speed']
        self.max_speed = rdict['max_speed']
        self.ascent = rdict['ascent']
        self.descent = rdict['descent']
        self.start_lat = rdict['start_lat']
        self.start_lon = rdict['start_lon']
        self.end_lat = rdict['end_lat']
        self.end_lon = rdict['end_lon']


class FitParser:

    def __init__(self) -> None:
        # transformer to convert from lat/lon to mercator coordinates needed for plots
        self.transfomer = Transformer.from_crs("EPSG:4326", "EPSG:3857")

    def parse_fit_file(self, fname: str) -> Ride:
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

        # calculate distances between the coordinate points
        dist_delta = np.zeros(len(coords) - 1)
        for i in range(len(coords) - 1):
            dist_delta[i] = geopy.distance.distance(coords[i],
                                                    coords[i + 1]).km

        dist = dist_delta.sum()

        # calculate ascent and descent
        alt_diff = alt_sm[1:] - alt_sm[:-1]
        ascent = alt_diff[alt_diff >= 0].sum()
        descent = alt_diff[alt_diff < 0].sum()

        # calculate moving time
        t_mov = time_delta[speed[:-1] > 4].sum() / 60
        avg_speed = dist / (t_mov / 60)

        # convert first lat/lon into mercator coordinates
        start_m_lat, start_m_lon = self.transfomer.transform(
            coords[0][0], coords[0][1])
        end_m_lat, end_m_lon = self.transfomer.transform(
            coords[-1][0], coords[-1][1])

        ride = Ride(
            Path(fname).stem.split('__')[1], time[0].year, time[0].month,
            time[0].isocalendar().week,
            time[0].day, time[0].hour, time[0].minute, dist, t_mov, avg_speed,
            speed.max(), ascent, descent, start_m_lat, start_m_lon, end_m_lat,
            end_m_lon)

        return ride


class RideStats:

    def __init__(self, fnames: list[Path]):
        self.fnames = fnames
        self.rides = []

        parser = FitParser()

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

            print(fname)
            preprocessed_fname = fname.with_suffix('.json')

            if not preprocessed_fname.exists():
                ride = parser.parse_fit_file(str(fname))
                ride.to_json(preprocessed_fname)
            else:
                ride = Ride()
                ride.from_json(preprocessed_fname)

            self.rides.append(ride)

    @property
    def df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rides)
        df['datetime'] = pd.to_datetime(
            df[['year', 'month', 'day', 'hour', 'minute']])
        df['grad'] = df['ascent'] / df['distance']
        # date string for tooltips
        df['date'] = [x.date().strftime("%y-%m-%d") for x in df.datetime]

        return df


def bokeh_cycling_stats(df, output_html_file):
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
    m1 = df.start_lon
    m2 = df.start_lat
    source = ColumnDataSource(data=dict(lat=m1, lon=m2))

    p41 = figure(x_range=(m2.min() - 0.05 * (m2.max() - m2.min()),
                          m2.max() + 0.05 * (m2.max() - m2.min())),
                 y_range=(m1.min() - 0.05 * (m1.max() - m1.min()),
                          m1.max() + 0.05 * (m1.max() - m1.min())),
                 x_axis_type="mercator",
                 y_axis_type="mercator",
                 title='map')
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