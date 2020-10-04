import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.stats as stats
import utils

def subset_traffic_data(path,boro_sel,link_id_path='linkIds.csv'):
    """takes a subset of the NYC traffic speed sensor network, by 
    borough andby average speed

    Args:
        path (str): path to the archived traffic speed data.s
        boro_sel (str): the boroughs that we want to include in our 
        analysis.
        link_id_path (str): path to csv containing 

    Returns:
        df (pandas.DataFrame): dataframe containing NYC traffic speed 
        sensor data for various NYC road segments with an average speed 
        > 20 mph (highways)

    """

    # each sensor has a corresponding linkId which indicates what road segment it monitors
    # select linkIds that are in the boroughs you're interested in
    df_link = pd.read_csv('linkIds.csv')
    link_ids = df_link[df_link['borough'].isin(boro_sel)]['link_id'].unique()

    # load the monthly traffic data files, dropping misformatted rows
    df = utils.load_csv(path)
    df = df[df['linkId'].isin(link_ids)]

    df = df.dropna()
    df['DataAsOf'] = pd.to_datetime(df['DataAsOf'])

    # get the average speed for links, drop if the average speed is less than 20 mph
    mean = df[['Speed','linkId']].groupby('linkId').mean()
    gt_20 = mean[mean['Speed']>20].index.astype(int)

    df = df[df['linkId'].isin(gt_20)]

    return df

def downsample_sensors(df,freq):
    """downsample each sensor's timeseries as the average speed over 15 
    minute intervals

    Args:
        df (pandas.DataFrame): dataframe containing all sensor data at 
        all locations

    Returns:
        df_rs (pandas.DataFrame): resampled dataframe with linkId as 
        the column and time as the index

    """ 

    df_list = []
    for group in df.groupby('linkId'):
        df_tem = group[1].set_index('DataAsOf')
        df_i = df_tem.rename(columns = {'Speed':df_tem['linkId'][0]})[df_tem['linkId'][0]]
        df_i = df_i.resample(freq).mean()

        df_list.append(df_i)

    df_rs = pd.concat(df_list, axis=1, ignore_index=False)

    return df_rs

def nyc_median_speed(df_rs,set_na=True,n_sensors=152):
    """create an NYC median traffic speed index, setting times where 
    >25% of traffic speed sensors aren't reporting data as np.NaN

    Args:
        df_rs (pandas.DataFrame): dataframe containing all sensor data 
        at all locations
        set_na (bool): Timesteps set to np.NaN if True 
        and >25% of sensors are not reporting
        n_sensors (bool): set to the total number of traffic speed 
        sensors in the NYC sensor network

    Returns:
        speed_ts (pandas.Series): timeseries of NYC median traffic 
        speed. Some timesteps set to np.NaN if set_na=True and >25% of 
        sensors are not reporting
        cond_toss (pandas.Series,bool): True where there are >25% of 
        traffic speed sensors not reporting

    """ 

    sensor_outage = df_rs.isna().sum(axis=1)
    tol = 153-(153*0.75)

    cond_toss = sensor_outage>tol

    speed_ts = df_rs.median(axis=1)

    if set_na:
        speed_ts[cond_toss] = np.nan

    return cond_toss,speed_ts

def clean_traffic(year = 2019,
    path = '*',
    boro_sel = ['Manhattan','Staten Island','Queens','Bronx','Brooklyn'],
    window = '2H',freq = '15min',
    ):
    """create a clean NYC median traffic speed timeseries"""

    path = f'{path}{year}.csv'

    df = subset_traffic_data(path,boro_sel)
    
    cond_toss, df_rs = downsample_sensors(df)

    cond_toss,speed_ts = nyc_median_speed(df_rs)

    return speed_ts

def download_weather_data(year):
    """download hourly historical weather from the Central Park ASOS
    station"""

    baseurl='https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?'
    request = ('station=NYC'
        '&data=tmpf&data=dwpf&data=p01i&data=tempf&data=sped'
        f'&year1={year}&month1=1&day1=1&year2={year}&month2=12&day2=31&tz=Etc%2FUTC'
        '&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2')

    df_weather = pd.read_csv(f'{baseurl}{request}',na_values=['M'])
    print(df_weather.isna().sum())
    
    return df_weather

def clean_weather(year,window,freq):
    
    df_weather = download_weather_data(year)
    
    # small amounts of precip are marked at T (trace)
    df_weather['p01i'][df_weather['p01i'] == 'T'] = 0.001
    df_weather['p01i'] = df_weather['p01i'].astype(float)

    # set datetime index and make sure only these columns are in the data ['time','tmpf','dwpf','p01i']
    df_weather['time'] = pd.to_datetime(df_weather['valid'])
    df_weather = df_weather[['time','p01i','tmpf']]
    df_weather['time'] = df_weather['time'] - pd.Timedelta('30min')
    df_weather = df_weather.set_index('time')

    #get weather "forecast" by shifting time index
    df_weather_pred = df_weather
    df_weather_pred = df_weather_pred.set_index(df_weather.index - pd.Timedelta(window))
    df_weather_pred = df_weather_pred.rename({'p01i': 'p01i_pred', 'tmpf': 'tmpf_pred'}, axis='columns')

    #drop na values
    df_weather = df_weather.dropna()
    df_weather_pred = df_weather_pred.dropna()
    
    df_weather = df_weather.resample(freq).pad()
    df_weather_pred = df_weather_pred.resample(freq).pad()
    
    df_weather['tmpf_pred'] = df_weather_pred['tmpf_pred']
    df_weather['p01i_pred'] = df_weather_pred['p01i_pred']
    
    return df_weather

def merge_data(df_weather,speed_ts):
    """merge the median NYC traffic speed data with weather data. Here
    we also fill gaps <1H and mark where larger gaps remain.

    Args:
        df_weather (pandas.DataFrame): dataframe containing predicted 
        and observed weather
        speed_ts (pandas.Series): timeseries of NYC median traffic 
        speed.

    Returns:
        df_merge (int): dataframe containing all features

    """    
    
    df_merge = df_weather
    df_merge['speed'] = speed_ts
    df_merge = df_merge[df_merge.index.isin(speed_ts.index)]
    
    dhour = (df_merge.index.hour.values+df_merge.index.minute.values/60)

    #create a feature for hour of day, normalize using sin
    df_merge['hour'] = np.sin((dhour/24)*2*np.pi)
    
    #interpolate gaps that are <1 hour long
    df_merge = df_merge.interpolate(method='linear',limit=4)
    
    # create a feature indicating where gaps remain
    df_merge['gaps'] = df_merge['speed'].isna().rolling(4).sum()
    df_merge['gaps'] = (df_merge['gaps'] > 0).astype(int)
    
    return df_merge
    
def build_am_pm_range(day,freq,am_start = 4,am_end = 12,pm_start = 12,pm_end = 20):
    """create ranges of times for the morning and afternoon subsets of 
    df_merge

    Args:
        df_merge (pandas.DataFrame): dataframe containing all features
        am_start (int): hour that you start your morning
        am_end (int): hour that you end your morning
        pm_start (int): hour that you start your afternoon
        pm_end (int): hour that you end your afternoon

    Returns:
        df_morning (pandas.DataFrame): morning subset of df_merge
        df_afternoon (pandas.DataFrame): afternoon subset of df_merge

    """
    
    am_start = pd.to_datetime(day.index.date[0]) + pd.Timedelta(f'{am_start}H')
    am_end = pd.to_datetime(day.index.date[0]) + pd.Timedelta(f'{am_end}H')
    
    pm_start = pd.to_datetime(day.index.date[0]) + pd.Timedelta(f'{pm_start}H')
    pm_end = pd.to_datetime(day.index.date[0]) + pd.Timedelta(f'{pm_end}H')
    
    am_range = pd.date_range(start=am_start,end=am_end, freq=freq)
    pm_range = pd.date_range(start=pm_start,end=pm_end, freq=freq)
    
    return am_range,pm_range

def split_into_segments(df_merge,freq,save=False):
    """split the timeseries into morning and afternoon segments that
    contain no gaps. Add a feature including the first difference of 
    traffic speed.

    Args:
        df_merge (pandas.DataFrame): dataframe containing all features

    Returns:
        df_morning (pandas.DataFrame): morning subset of df_merge
        df_afternoon (pandas.DataFrame): afternoon subset of df_merge

    """ 

    count = 0
    morning_list = []
    afternoon_list = []

    # go through series, day by day, splitting into am and pm segments
    for group in df_merge.groupby(df_merge.index.date):

        day = group[1]
        day['speed_diff'] = day['speed'].diff()
        am_range,pm_range = build_am_pm_range(day)

        morning = day[day.index.isin(am_range)]
        afternoon = day[day.index.isin(pm_range)]
        # check if segment if the full length and that there are no gaps
        if (len(am_range) == len(morning)) & (morning['gaps'].max() < 1):
            morning_list.append(morning)
        if (len(pm_range) == len(afternoon)) & (afternoon['gaps'].max() < 1):
            afternoon_list.append(afternoon)
            
    df_morning = pd.concat(morning_list, axis=0, ignore_index=False)
    df_afternoon = pd.concat(afternoon_list, axis=0, ignore_index=False)
    
    if save:
        morning_df.to_csv(f'morning_df_{year}.csv')
        print('saved morning_df.csv')
        afternoon_df.to_csv(f'afternoon_df_{year}.csv')
        print('saved afternoon_df.csv')
        
    return df_morning,df_afternoon


