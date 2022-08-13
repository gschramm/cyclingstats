import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as py
import configparser
import hashlib
import pathlib
from glob import glob
from datetime import date, timedelta

import geopy.distance
import fitparse

from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Band
from bokeh.layouts import gridplot
from bokeh.palettes import Plasma11
from bokeh.transform import linear_cmap

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------


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


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------


def bokeh_cycling_stats(df, output_html_file):
    output_file(output_html_file,
                title='cycling stats ' + df.datetime[-1].strftime('%Y-%m-%d'))

    # calculate gradient for each ride
    df['grad'] = df['ascent [km]'] / df['distance [km]']

    # do regression
    xreg = df['grad'].values
    yreg = df['avg speed [km/h]'].values
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

    p11 = figure(title="weekly distance [km]",
                 x_range=weekly_stats['cat'],
                 tooltips=[('week', "@{cat}"),
                           ('distance [km]', "@{distance [km]}")])
    p11.vbar(x='cat',
             top='distance [km]',
             width=0.7,
             source=weekly_stats,
             line_width=0)
    p11.xaxis.major_label_orientation = np.pi / 2

    p12 = figure(title="monthly distance [km]",
                 x_range=monthly_stats['cat'],
                 tooltips=[('month', "@{cat}"),
                           ('distance [km]', "@{distance [km]}")])
    p12.vbar(x='cat',
             top='distance [km]',
             width=0.7,
             source=monthly_stats,
             line_width=0)

    p13 = figure(title="yearly distance [km]",
                 x_range=yearly_stats['cat'],
                 tooltips=[('year', "@{cat}"),
                           ('distance [km]', "@{distance [km]}")])
    p13.vbar(x='cat',
             top='distance [km]',
             width=0.7,
             source=yearly_stats,
             line_width=0)

    p21 = figure(title="weekly ascent [km]",
                 x_range=weekly_stats['cat'],
                 tooltips=[('week', "@{cat}"),
                           ('ascent [km]', "@{ascent [km]}")])
    p21.vbar(x='cat',
             top='ascent [km]',
             width=0.7,
             source=weekly_stats,
             fill_color='darkorange',
             line_width=0)
    p21.xaxis.major_label_orientation = np.pi / 2

    p22 = figure(title="monthly ascent [km]",
                 x_range=monthly_stats['cat'],
                 tooltips=[('month', "@{cat}"),
                           ('ascent [km]', "@{ascent [km]}")])
    p22.vbar(x='cat',
             top='ascent [km]',
             width=0.7,
             source=monthly_stats,
             fill_color='darkorange',
             line_width=0)

    p23 = figure(title="yearly ascent [km]",
                 x_range=yearly_stats['cat'],
                 tooltips=[('year', "@{cat}"),
                           ('ascent [km]', "@{ascent [km]}")])
    p23.vbar(x='cat',
             top='ascent [km]',
             width=0.7,
             source=yearly_stats,
             fill_color='darkorange',
             line_width=0)

    # plot histograms about rides
    dist_histo = np.histogram(
        df["distance [km]"],
        bins=np.arange(np.ceil(df['distance [km]'].max() / 10) + 2) * 10 - 5)
    p31 = figure(title='ride distance [km] histogram')
    p31.quad(top=dist_histo[0],
             bottom=0,
             left=dist_histo[1][:-1],
             right=dist_histo[1][1:],
             fill_color="darkseagreen",
             line_width=0)

    mt_histo = np.histogram(
        df["moving time [min]"],
        bins=np.arange(np.ceil(df['moving time [min]'].max() / 20) + 2) * 20 -
        10)
    p32 = figure(title='ride moving time [min] histogram')
    p32.quad(top=mt_histo[0],
             bottom=0,
             left=mt_histo[1][:-1],
             right=mt_histo[1][1:],
             fill_color="darkseagreen",
             line_width=0)

    p33 = figure(title='ride avg speed [km/h] vs (ascent / distance)',
                 tooltips=[('distance [km]', "@{distance [km]}"),
                           ('ascent [km]', "@{ascent [km]}"),
                           ('moving time [min]', "@{moving time [min]}"),
                           ('avg speed [km/h]', "@{avg speed [km/h]}"),
                           ('date', "@{date}")])
    #p33.line(xreg, reg)
    p33.line(xreg, low, color='gray')
    p33.line(xreg, up, color='gray')
    p33.scatter('grad',
                'avg speed [km/h]',
                source=df,
                size=8,
                color=linear_cmap(field_name='distance [km]',
                                  palette=Plasma11,
                                  low=df['distance [km]'].min(),
                                  high=1.1 * df['distance [km]'].max()))

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


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
def plot_cycling_stats(df):
    # summary stats
    weekly_stats = df.groupby('week')[['distance [km]', 'ascent [km]']].sum()
    monthly_stats = df.groupby('month')[['distance [km]', 'ascent [km]']].sum()
    yearly_stats = df.groupby('year')[['distance [km]', 'ascent [km]']].sum()

    #----------------------------------------------------------------------------------------
    # plots

    fig = py.figure(figsize=(18, 4))
    gs = matplotlib.gridspec.GridSpec(1,
                                      3,
                                      width_ratios=[
                                          weekly_stats.shape[0],
                                          monthly_stats.shape[0],
                                          yearly_stats.shape[0]
                                      ])
    ax = np.array([py.subplot(x) for x in gs])

    weekly_stats.plot(kind='bar',
                      ax=ax[0],
                      rot=25,
                      secondary_y='ascent [km]',
                      legend=True,
                      width=0.75)
    monthly_stats.plot(kind='bar',
                       ax=ax[1],
                       rot=25,
                       secondary_y='ascent [km]',
                       legend=False,
                       width=0.75)
    yearly_stats.plot(kind='bar',
                      ax=ax[2],
                      rot=25,
                      secondary_y='ascent [km]',
                      legend=False,
                      width=0.75)

    for axx in ax.flatten():
        axx.set_axisbelow(True)
        axx.grid(ls=':')

    fig.tight_layout()

    return fig


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------


def parse_fit_files(data_path, df_file, salt):
    fnames = sorted(
        glob(os.path.join(data_path, '20??', '*.fit')) +
        glob(os.path.join(data_path, '20??', '*.FIT')))

    if os.path.exists(df_file):
        df = pd.read_csv(df_file, index_col=0, parse_dates=['datetime'])
    else:
        df = pd.DataFrame(columns=[
            'datetime', 'distance [km]', 'moving time [min]',
            'avg speed [km/h]', 'max speed [km/h]', 'ascent [km]',
            'descent [km]'
        ])

    # factor to convert from semicircles to degrees
    scf = (180 / 2**31)

    for fname in fnames:
        print(fname)
        index = hashlib.sha256(
            pathlib.Path(fname).read_bytes() +
            salt.encode('utf-8')).hexdigest()[:8]

        if not index in df.index:
            fitfile = fitparse.FitFile(fname)

            asc_fac = 1.
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

            has_speed = True

            # Get all data messages that are of type record
            for record in fitfile.get_messages('record'):
                field_names = [x.name for x in record.fields]

                if 'position_long' in field_names:
                    time.append(record.get('timestamp').value)
                    alt.append(record.get('enhanced_altitude').value)
                    coords.append((record.get('position_lat').value * scf,
                                   record.get('position_long').value * scf))

                    #speed.append(record.get('enhanced_speed').value)

            # calculate difference between time tags
            time = np.array(time)
            time_delta = np.array([x.seconds for x in (time[1:] - time[:-1])])

            # calculate distances between the coordinate points
            dist_delta = np.zeros(len(coords) - 1)
            for i in range(len(coords) - 1):
                dist_delta[i] = geopy.distance.distance(
                    coords[i], coords[i + 1]).km

            dist = np.cumsum(dist_delta)

            speed = np.zeros(dist.shape)

            for i in np.arange(5, speed.shape[0] - 5):
                speed[i] = (
                    geopy.distance.distance(coords[i + 5], coords[i - 5]).km /
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

            # fill data frame
            df.loc[index] = pd.Series({
                'datetime': time[0],
                'distance [km]': dist[-1],
                'moving time [min]': t_mov / 60,
                'avg speed [km/h]': avg_speed,
                'max speed [km/h]': speed.max(),
                'ascent [km]': asc_fac * ascent / 1000.,
                'descent [km]': asc_fac * descent / 1000.
            })

    # add week / month / year column
    df['week'] = df.datetime.apply(
        lambda x: (x - timedelta(days=x.weekday())).strftime('%y-%V'))
    df['month'] = df.datetime.apply(lambda x: x.strftime('%y-%m'))
    df['year'] = df.datetime.apply(lambda x: x.year)

    # save data frame
    df.to_csv(df_file, float_format='%.3f')

    return df


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    datapath = config['default']['datapath']
    df_file = config['default']['df_file']
    salt = config['default']['salt']
    df = parse_fit_files(datapath, df_file, salt)

    bokeh_cycling_stats(df, 'cycling_stats.html')
