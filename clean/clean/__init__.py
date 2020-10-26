import datetime
import glob
import urllib


import pytz
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.stats as stats

def utc_to_local(utc_dt,local_tz):
    
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return local_tz.normalize(local_dt)

def four_hours_ago():

    local_tz = pytz.timezone('US/Eastern')
    current_time_utc = datetime.datetime.now()

    current_time_et = utc_to_local(current_time_utc,local_tz)

    return current_time_et - datetime.timedelta(hours=4)

def load_speed_from_csv(path):
    
    '''minor cleaning and concatenates csv files'''
    
    all_files = glob.glob(path)

    li = []

    for filename in tqdm(all_files):
        df = pd.read_csv(filename, index_col=None, header=0)
        df.columns = ['Id', 'Speed', 'TravelTime', 'Status', 'DataAsOf', 'linkId']
        if df['Speed'].dtype != np.float64:
            cond = ((df['Speed'].str.contains('Speed')) | (df['Speed'].str.contains('Bronx')))
            df[cond]=np.nan
        df['Speed'] = df['Speed'].astype(float)
        df['DataAsOf'] = df['DataAsOf'].astype(str)
        df = df[['Speed','DataAsOf','linkId']]
        li.append(df)


    return pd.concat(li, axis=0, ignore_index=True)

def load_speed_from_api():

    #SQL query
    query = 'SELECT%20LINK_ID,SPEED,DATA_AS_OF%20WHERE%20DATA_AS_OF%20<%20%272020-01-01%27%20'
    # query = ('SELECT LINK_ID,SPEED,DATA_AS_OF '
    #     'WHERE DATA_AS_OF > \'2020-09-01\'')
    # print(query)
    # query = urllib.parse.quote_plus(query)
    query = '$query={query}'

    base_url = 'https://data.cityofnewyork.us/resource/i4gi-tjb9.csv?'
    
    return pd.read_csv('https://data.cityofnewyork.us/resource/i4gi-tjb9.csv?$query=SELECT%20LINK_ID,SPEED,DATA_AS_OF%20WHERE%20DATA_AS_OF%20%3E%20%272020-01-22T03:59:00.000%27%20%20LIMIT%2010000')

def subset_speed_data(df,boro_sel,link_id_path='./forecast/linkIds.csv'):
    """takes a subset of the NYC traffic speed sensor network, by 
    borough and by average speed

    Args:
        path (str): path to the archived traffic speed data
        boro_sel (str): the boroughs that we want to include in our 
        analysis.
        link_id_path (str): path to csv containing road segment IDs 

    Returns:
        df (pandas.DataFrame): dataframe containing NYC traffic speed 
        sensor data for various NYC road segments with an average speed 
        > 20 mph (highways)

    """

    # each sensor has a corresponding linkId which indicates what road segment it monitors
    # select linkIds that are in the boroughs you're interested in
    df_link = pd.read_csv(link_id_path)
    link_ids = df_link[df_link['borough'].isin(boro_sel)]['link_id'].unique()
    
    df = df[df['linkId'].isin(link_ids)]

    print('dropping na values, will take a couple minutes')
    df = df.dropna()
    df['DataAsOf'] = pd.to_datetime(df['DataAsOf'])

    # get the average speed for links, drop if the average speed is less than 20 mph
    mean = df[['Speed','linkId']].groupby('linkId').mean()
    gt_20 = mean[mean['Speed']>20].index.astype(int)

    df = df[df['linkId'].isin(gt_20)]
    print('done')

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

def nyc_median_speed(df_rs,set_na=True,n_sensors=153,frac=0.75):
    """create an NYC median traffic speed index, setting times where 
    >25% of traffic speed sensors aren't reporting data as np.NaN

    Args:
        df_rs (pandas.DataFrame): dataframe containing all sensor data 
        at all locations
        set_na (bool): Timesteps set to np.NaN if True 
        and >25% of sensors are not reporting
        n_sensors (int): set to the total number of traffic speed 
        sensors in the NYC sensor network
        frac (float): fraction of sensors needed to keep a data point

    Returns:
        median_speed (pandas.Series): timeseries of NYC median traffic 
        speed. Some timesteps set to np.NaN if set_na=True and >25% of 
        sensors are not reporting
        cond_toss (pandas.Series,bool): True where there are >25% of 
        traffic speed sensors not reporting

    """ 

    sensor_outage = df_rs.isna().sum(axis=1)
    tol = n_sensors-(n_sensors*0.75) # tolerance

    cond_toss = sensor_outage>tol

    median_speed = df_rs.median(axis=1)

    if set_na:
        median_speed[cond_toss] = np.nan

    return cond_toss,median_speed

def clean_median_speed(year = 2019,
    path = '*',
    boro_sel = ['Manhattan','Staten Island','Queens','Bronx','Brooklyn'],
    window = '2H',freq = '15min',
    ):
    """create a clean NYC median traffic speed timeseries"""

    path = f'../nyc_speed_data/{path}{year}.csv'

    df = subset_speed_data(path,boro_sel)
    
    df_rs = downsample_sensors(df,freq)

    cond_toss,median_speed = nyc_median_speed(df_rs)

    return median_speed

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
    """Upsample hourly weather data to 15min intervals (padding), 
    and feature for frozen precip"""
    
    df_weather = download_weather_data(year)
    
    # small amounts of precip are marked at T (trace)
    df_weather['p01i'][df_weather['p01i'] == 'T'] = 0.001
    df_weather['p01i'] = df_weather['p01i'].astype(float)

    # set datetime index and make sure only these columns are in the 
    # data ['time','tmpf','p01i']
    df_weather['time'] = pd.to_datetime(df_weather['valid'])
    df_weather = df_weather[['time','p01i','tmpf']]
    df_weather['time'] = df_weather['time'] - pd.Timedelta('30min')
    df_weather = df_weather.set_index('time')

    # create one timeseries for preciptiation when air temp < 32F and 
    # another when air temp > 32F
    df_weather['frz_prec'] = df_weather.pop('p01i')
    df_weather['liq_prec'] = df_weather['frz_prec']

    # when these conditions are true, set to zero
    cond_frz = ((df_weather['tmpf']>32) & (df_weather['frz_prec'].notna()))
    cond_liq = ((df_weather['tmpf']<32) & (df_weather['liq_prec'].notna()))

    df_weather['frz_prec'][cond_frz] = 0
    df_weather['liq_prec'][cond_liq] = 0

    #get weather "forecast" by shifting time index
    df_weather_pred = df_weather
    df_weather_pred = df_weather_pred.set_index(df_weather.index - pd.Timedelta(window))

    #drop na values
    df_weather = df_weather.dropna()
    df_weather_pred = df_weather_pred.dropna()
    
    # duplicate values to match the 15min sampling frequency of
    df_weather = df_weather.resample(freq).pad()
    df_weather_pred = df_weather_pred.resample(freq).pad()
    
    df_weather['frz_prec_pred'] = df_weather_pred['frz_prec']
    df_weather['liq_prec_pred'] = df_weather_pred['liq_prec']
    
    return df_weather

def merge_data(df_weather,median_speed):
    """merge the median NYC traffic speed data with weather data. Here
    gaps <1H are fill and larger gaps are identified.

    Args:
        df_weather (pandas.DataFrame): dataframe containing predicted 
        and observed weather
        median_speed (pandas.Series): timeseries of NYC median traffic 
        speed.

    Returns:
        df_merge (int): dataframe containing all features

    """    
    
    df_merge = df_weather
    df_merge['speed'] = median_speed
    df_merge = df_merge[df_merge.index.isin(median_speed.index)]
    
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

def split_into_segments(df_merge,freq,year,save=False):
    """split the timeseries into morning and afternoon segments that
    contain no gaps. Add a feature including the first difference of 
    traffic speed.

    Args:
        df_merge (pandas.DataFrame): dataframe containing all features

    Returns:
        df_morning (pandas.DataFrame): morning subset of df_merge
        df_afternoon (pandas.DataFrame): afternoon subset of df_merge

    """ 

    morning_list = []
    afternoon_list = []

    # go through series, day by day, splitting into am and pm segments
    for group in df_merge.groupby(df_merge.index.date):

        day = group[1]
        day['speed_diff'] = day['speed'].diff()
        am_range,pm_range = build_am_pm_range(day,freq)

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
        df_morning.to_csv(f'morning_df_{year}.csv')
        print('saved morning_df.csv')
        df_afternoon.to_csv(f'afternoon_df_{year}.csv')
        print('saved afternoon_df.csv')
        
    return df_morning,df_afternoon


