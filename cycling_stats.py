import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as py
import configparser
from glob import glob

import geopy.distance
import fitparse

from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import gridplot

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def bokeh_cycling_stats(df, output_html_file):
  output_file(output_html_file)
  
  weekly_stats  = df.groupby('week')[['distance [km]', 'ascent [km]']].sum()
  monthly_stats = df.groupby('month')[['distance [km]', 'ascent [km]']].sum()
  yearly_stats  = df.groupby('year')[['distance [km]', 'ascent [km]']].sum()
  
  weekly_stats['cat']  = [str(x) for x in weekly_stats.index.values]
  monthly_stats['cat'] = [str(x) for x in monthly_stats.index.values]
  
  p11 = figure(title ="weekly distance [km]", x_range = weekly_stats['cat'],
              tooltips = [('week', "@{week}"),('distance [km]', "@{distance [km]}")])
  p11.vbar(x = 'cat', top = 'distance [km]', width = 0.7, source = weekly_stats)
  
  p12 = figure(title ="monthly distance [km]", x_range = monthly_stats['cat'],
              tooltips = [('month', "@{month}"),('distance [km]', "@{distance [km]}")])
  p12.vbar(x = 'cat', top = 'distance [km]', width = 0.7, source = monthly_stats)
  
  p21 = figure(title ="weekly ascent [km]", x_range = weekly_stats['cat'],
              tooltips = [('week', "@{week}"),('distance [km]', "@{ascent [km]}")])
  p21.vbar(x = 'cat', top = 'ascent [km]', width = 0.7, source = weekly_stats, 
           fill_color = 'gray')
  
  p22 = figure(title ="monthly ascent [km]", x_range = monthly_stats['cat'],
              tooltips = [('month', "@{month}"),('distance [km]', "@{ascent [km]}")])
  p22.vbar(x = 'cat', top = 'ascent [km]', width = 0.7, source = monthly_stats, 
           fill_color = 'gray')
  
  grid = gridplot([[p11, p21], [p12, p22]], plot_width = 800, plot_height = 400)

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
def plot_cycling_stats(df):
  # summary stats
  weekly_stats  = df.groupby('week')[['distance [km]', 'ascent [km]']].sum()
  monthly_stats = df.groupby('month')[['distance [km]', 'ascent [km]']].sum()
  yearly_stats  = df.groupby('year')[['distance [km]', 'ascent [km]']].sum()
  
  #----------------------------------------------------------------------------------------
  # plots
  
  fig = py.figure(figsize = (18,4))
  gs  = matplotlib.gridspec.GridSpec(1, 3, 
         width_ratios=[weekly_stats.shape[0],monthly_stats.shape[0],yearly_stats.shape[0]]) 
  ax  = np.array([py.subplot(x) for x in gs])
  
  weekly_stats.plot(kind = 'bar', ax = ax[0], rot = 25, secondary_y = 'ascent [km]', 
                    legend = True, width = 0.75)
  monthly_stats.plot(kind = 'bar', ax = ax[1], rot = 25, secondary_y = 'ascent [km]', 
                     legend = False, width = 0.75)
  yearly_stats.plot(kind = 'bar', ax = ax[2], rot = 25, secondary_y = 'ascent [km]', 
                    legend = False, width = 0.75)
  
  for axx in ax.flatten():
    axx.set_axisbelow(True)
    axx.grid(ls = ':')
  
  fig.tight_layout()

  return fig
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def parse_fit_files(data_path,df_file):
  fnames = sorted(glob(os.path.join (data_path,'20??','*.fit')) + 
                  glob(os.path.join (data_path,'20??','*.FIT')))
  
  if os.path.exists(df_file):
    df = pd.read_csv(df_file, index_col = 0, parse_dates=['datetime'])
  else:
    df = pd.DataFrame(columns = ['datetime','distance [km]', 'moving time [min]', 'avg speed [km/h]', 
                                 'max speed [km/h]', 'ascent [km]', 'descent [km]'])
  
  # factor to convert from semicircles to degrees
  scf = (180 / 2**31)
  
  for fname in fnames:
    print(fname)
    index = os.path.splitext(os.path.basename(fname))[0]
  
    if not index in df.index:
      fitfile = fitparse.FitFile(fname)
   
      asc_fac = 1.
      mes_fields = [x.name for x in fitfile.messages[0].fields]
      if 'manufacturer' in mes_fields:
        manufac = fitfile.messages[0].get('manufacturer').value
        # the ascent recorded by a phone with strave has to be corrected
        if manufac == 'strava':
          asc_fac = 0.6 
  
      time   = []
      dist   = []
      speed  = []
      alt    = []
      coords = []
   
      has_speed = True
  
      # Get all data messages that are of type record
      for record in fitfile.get_messages('record'):
        field_names = [x.name for x in record.fields]
  
        if 'position_long' in field_names:
          time.append(record.get('timestamp').value)
          alt.append(record.get('enhanced_altitude').value)
          coords.append((record.get('position_lat').value * scf, record.get('position_long').value * scf))
  
          #speed.append(record.get('enhanced_speed').value)
  
      # calculate difference between time tags
      time  = np.array(time)
      time_delta = np.array([x.seconds for x in (time[1:] - time[:-1])])
  
      # calculate distances between the coordinate points
      dist_delta = np.zeros(len(coords) - 1)
      for i in range(len(coords) - 1):
        dist_delta[i] = geopy.distance.distance(coords[i],coords[i+1]).km
  
      dist  = np.cumsum(dist_delta)
  
      speed = np.zeros(dist.shape)
  
      for i in np.arange(5, speed.shape[0] - 5):
        speed[i] = (geopy.distance.distance(coords[i+5],coords[i-5]).km / 
                    ((time[i+5] - time[i-5]).seconds / 3600))
  
      alt   = np.array(alt)
  
      # calculate ascend and descent
      alt_sm   = np.convolve(alt,np.ones(33)/33, mode = 'valid')
      alt_diff = alt_sm[1:] - alt_sm[:-1]
      ascent   = alt_diff[alt_diff>=0].sum()
      descent  = alt_diff[alt_diff<0].sum()
  
      # calculate moving time 
      t_mov = time_delta[speed > 1].sum()
  
      avg_speed = dist[-1] / (t_mov/3600)
  
      # fill data frame
      df.loc[index] = pd.Series({'datetime':time[0], 'distance [km]':dist[-1], 
                                 'moving time [min]':t_mov/60, 
                                 'avg speed [km/h]':avg_speed, 'max speed [km/h]': speed.max(),
                                 'ascent [km]':asc_fac*ascent/1000., 
                                 'descent [km]': asc_fac*descent / 1000.})
 
  # add week / month / year column
  df['week']  = df.datetime.apply(lambda x: x.strftime('%y-%W'))
  df['month'] = df.datetime.apply(lambda x: x.strftime('%y-%m'))
  df['year']  = df.datetime.apply(lambda x: x.year)

  # save data frame
  df.to_csv(df_file)

  return df
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

if __name__ == '__main__':
  config = configparser.ConfigParser()
  config.read('config.ini')
  datapath = config['default']['datapath']
  df_file = os.path.join(datapath,config['default']['df_file'])
  df = parse_fit_files(datapath, df_file)

  bokeh_cycling_stats(df, 'stats.html')

  fig = plot_cycling_stats(df)
  fig.savefig(os.path.splitext(df_file)[0] + '.pdf')
  py.show()
