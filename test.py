import dataclasses
from pathlib import Path
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import geopy.distance
import fitparse

from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Band, FuncTickFormatter
from bokeh.layouts import gridplot
from bokeh.palettes import Plasma11
from bokeh.transform import linear_cmap


def linreg(x, y):
    """ linear regression with estimate of 95% conf interval
      https://tomholderness.wordpress.com/2013/01/10/confidence_intervals/
  """

    xx = x.copy()
    yy = y.copy()

    for i in range(2):
        # fit a curve to the data using a least squares 1st order polynomial fit
        z = np.polyfit(xx, yy, 1)
        p = np.poly1d(z)
        fit = p(xx)

        # predict y values of origional data using the fit
        p_y = z[0] * xx + z[1]

        # calculate the y-error (residuals)
        y_err = yy - p_y

        # remove outliers
        inds = np.where(
            np.abs(y_err) < np.abs(y_err).mean() + 3 * np.abs(y_err).std())
        xx = x[inds]
        yy = y[inds]

    # predict y values of origional data using the fit
    p_y = z[0] * x + z[1]
    y_err = y - p_y

    # now calculate confidence intervals for new test x-series
    mean_x = np.mean(x[inds])  # mean of x
    n = len(x)  # number of samples in origional fit
    t = 2.31  # appropriate t value (where n=9, two tailed 95%)
    s_err = np.sum(np.power(y_err[inds],
                            2))  # sum of the squares of the residuals

    confs = t * np.sqrt((s_err / (n - 2)) * (1.0 / n + (np.power(
        (x - mean_x), 2) / ((np.sum(np.power(x, 2))) - n *
                            (np.power(mean_x, 2))))))

    # get lower and upper confidence limits based on predicted y and confidence intervals
    lower = p_y - 2 * abs(confs)
    upper = p_y + 2 * abs(confs)

    return p_y, lower, upper


@dataclasses.dataclass
class Ride:
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

    @property
    def __dict__(self):
        return dataclasses.asdict(self)

    def to_json(self, fname: Path):
        with open(fname, 'w') as f:
            json.dump(self.__dict__, f)

    def from_json(self, fname: Path):
        with open(fname, 'r') as f:
            rdict = json.load(f)
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


def parse_fit_file(fname: str) -> Ride:
    fitfile = fitparse.FitFile(fname)

    mes_fields = [x.name for x in fitfile.messages[0].fields]
    if 'manufacturer' in mes_fields:
        manufac = fitfile.messages[0].get('manufacturer').value
        # the ascent recorded by a phone with strave has to be corrected
        if manufac == 'strava':
            asc_fac = 0.6

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
            alt.append(record.get('enhanced_altitude').value)
            coords.append(
                (record.get('position_lat').value * semi_circ_to_deg,
                 record.get('position_long').value * semi_circ_to_deg))

            #speed.append(record.get('enhanced_speed').value)

    # calculate difference between time tags
    time = np.array(time)
    time_delta = np.array([x.seconds for x in (time[1:] - time[:-1])])

    # calculate distances between the coordinate points
    dist_delta = np.zeros(len(coords) - 1)
    for i in range(len(coords) - 1):
        dist_delta[i] = geopy.distance.distance(coords[i], coords[i + 1]).km

    dist = np.cumsum(dist_delta)

    speed = np.zeros(dist.shape)

    for i in np.arange(5, speed.shape[0] - 5):
        speed[i] = (geopy.distance.distance(coords[i + 5], coords[i - 5]).km /
                    ((time[i + 5] - time[i - 5]).seconds / 3600))

    alt = np.array(alt)

    # calculate ascend and descent
    if manufac == 'strava':
        alt_sm = np.convolve(alt, np.ones(33) / 33, mode='valid')
    else:
        alt_sm = np.convolve(alt, np.ones(7) / 7, mode='valid')
    alt_diff = alt_sm[1:] - alt_sm[:-1]
    ascent = alt_diff[alt_diff >= 0].sum()
    descent = alt_diff[alt_diff < 0].sum()

    # calculate moving time
    t_mov = time_delta[speed > 1].sum()

    avg_speed = dist[-1] / (t_mov / 3600)

    ride = Ride(time[0].year, time[0].month, time[0].isocalendar().week,
                time[0].day,
                time[0].hour, time[0].minute, dist[-1], t_mov / 60, avg_speed,
                speed.max(), ascent / 1000, descent / 10000)

    return ride


class RideStats:

    def __init__(self, fnames: list[Path]):
        self.fnames = fnames
        self.rides = []

        for fname in self.fnames:
            print(fname)
            preprocessed_fname = fname.with_suffix('.json')

            if not preprocessed_fname.exists():
                ride = parse_fit_file(str(fname))
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

        return df


def bokeh_cycling_stats(df, output_html_file):
    output_file(output_html_file,
                title='cycling stats ' +
                df.iloc[-1, :].datetime.strftime('%Y-%m-%d'))

    # calculate gradient for each ride
    df['grad'] = df['ascent'] / df['distance']

    # do regression
    xreg = df['grad'].values
    yreg = df['avg_speed'].values
    isort = np.argsort(xreg)
    xreg = xreg[isort]
    yreg = yreg[isort]

    reg, low, up = linreg(xreg, yreg)

    # date for tooltips
    df['date'] = [x.date().strftime("%y-%m-%d") for x in df.datetime]

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

    p11 = figure(title="weekly distance",
                 x_range=weekly_stats['cat'],
                 tooltips=[('week', "@{cat}"), ('distance', "@{distance}")])
    p11.vbar(x='cat',
             top='distance',
             width=0.7,
             source=weekly_stats,
             line_width=0)
    p11.xaxis.formatter = formatter
    p11.xaxis.major_label_orientation = np.pi / 2

    p12 = figure(title="monthly distance [km]",
                 x_range=monthly_stats['cat'],
                 tooltips=[('month', "@{cat}"), ('distance', "@{distance}")])
    p12.vbar(x='cat',
             top='distance',
             width=0.7,
             source=monthly_stats,
             line_width=0)
    p12.xaxis.major_label_orientation = np.pi / 2

    p13 = figure(title="yearly distance",
                 x_range=yearly_stats['cat'],
                 tooltips=[('year', "@{cat}"), ('distance', "@{distance}")])
    p13.vbar(x='cat',
             top='distance',
             width=0.7,
             source=yearly_stats,
             line_width=0)

    p21 = figure(title="weekly ascent",
                 x_range=weekly_stats['cat'],
                 tooltips=[('week', "@{cat}"), ('ascent', "@{ascent}")])
    p21.vbar(x='cat',
             top='ascent',
             width=0.7,
             source=weekly_stats,
             fill_color='darkorange',
             line_width=0)
    p21.xaxis.formatter = formatter
    p21.xaxis.major_label_orientation = np.pi / 2

    p22 = figure(title="monthly ascent",
                 x_range=monthly_stats['cat'],
                 tooltips=[('month', "@{cat}"), ('ascent', "@{ascent}")])
    p22.xaxis.major_label_orientation = np.pi / 2

    p22.vbar(x='cat',
             top='ascent',
             width=0.7,
             source=monthly_stats,
             fill_color='darkorange',
             line_width=0)

    p23 = figure(title="yearly ascent",
                 x_range=yearly_stats['cat'],
                 tooltips=[('year', "@{cat}"), ('ascent', "@{ascent}")])
    p23.vbar(x='cat',
             top='ascent',
             width=0.7,
             source=yearly_stats,
             fill_color='darkorange',
             line_width=0)

    # plot histograms about rides
    dist_histo = np.histogram(
        df["distance"],
        bins=np.arange(np.ceil(df['distance'].max() / 10) + 2) * 10 - 5)
    p31 = figure(title='ride distance [km] histogram')
    p31.quad(top=dist_histo[0],
             bottom=0,
             left=dist_histo[1][:-1],
             right=dist_histo[1][1:],
             fill_color="darkseagreen",
             line_width=0)

    mt_histo = np.histogram(
        df["moving_time"],
        bins=np.arange(np.ceil(df['moving_time'].max() / 20) + 2) * 20 - 10)
    p32 = figure(title='ride moving time [min] histogram')
    p32.quad(top=mt_histo[0],
             bottom=0,
             left=mt_histo[1][:-1],
             right=mt_histo[1][1:],
             fill_color="darkseagreen",
             line_width=0)

    p33 = figure(title='ride avg speed [km/h] vs (ascent / distance)',
                 tooltips=[('distance', "@{distance}"),
                           ('ascent', "@{ascent}"),
                           ('moving time', "@{moving_time}"),
                           ('avg speed', "@{avg_speed}"), ('date', "@{date}")])
    p33.line(xreg, low, color='gray')
    p33.line(xreg, up, color='gray')
    p33.scatter('grad',
                'avg_speed]',
                source=df,
                size=8,
                color=linear_cmap(field_name='distance',
                                  palette=Plasma11,
                                  low=df['distance'].min(),
                                  high=1.1 * df['distance'].max()))

    for fig in [p11, p21, p12, p22, p13, p23, p31, p32, p33]:
        fig.toolbar.active_drag = None
        fig.toolbar.active_scroll = None
        fig.toolbar.active_tap = None

    # group all figures in a grid
    grid = gridplot(
        [[p11, p21], [p12, p22], [p13, p23], [p31, p32], [p33, None]],
        merge_tools=False,
        width=600,
        height=250,
        sizing_mode='scale_both')
    show(grid)


if __name__ == '__main__':
    rs = RideStats(
        sorted(list(
            Path('/home/georg/Nextcloud/cycling/data').rglob('*.FIT'))))

    bokeh_cycling_stats(rs.df, 'test.html')