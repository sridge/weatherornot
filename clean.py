import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.stats as stats
import utils


def subset_data(path,boro_sel,link_id_path='linkIds.csv'):
    """Takes a subset of the NYC traffic speed sensor network, by 
    borough andby average speed

    Args:
        path (str): path to the archived traffic speed data.
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

def resample_sensors(df):
    """resample each sensor as the average over 15 minute intervals

    Args:
        df (pandas.DataFrame): dataframe containing all sensor data at all locations

    Returns:
        df_rs (pandas.DataFrame): resampled dataframe with linkId as the 
        column and time as the index

    """ 

    df_list = []
    for group in df.groupby('linkId'):
        df_tem = group[1].set_index('DataAsOf')
        df_i = df_tem.rename(columns = {'Speed':df_tem['linkId'][0]})[df_tem['linkId'][0]]
        df_i = df_i.resample(freq).mean()

        df_list.append(df_i)

    df_rs = pd.concat(df_list, axis=1, ignore_index=False)

    return df_rs

def check_outages(df_rs,set_na=True):
    """create an NYC median traffic speed index, setting times where
     >25% of traffic speed sensors aren't reporting data as np.NaN

    Args:
        df_rs (pandas.DataFrame): dataframe containing all sensor data 
        at all locations
        set_na (bool): Timesteps set to np.NaN if True and >25% of
        sensors are not reporting

    Returns:
        speed_ts (pandas.Series): timeseries of NYC median traffic 
        speed. Some timesteps set to np.NaN if set_na=True and >25% of
        sensors are not reporting

    """ 

    sensor_outage = df_rs.isna().sum(axis=1)
    tol = 153-(153*0.75)

    cond_toss = sensor_outage>tol

    speed_ts = df_rs.median(axis=1)

    if set_na:
        speed_ts[cond_toss] = np.nan

    return cond_toss,speed_ts

def clean(year = 2019,
    path = f'../nyc_speed_data/*{year}.csv',
    boro_sel = ['Manhattan','Staten Island','Queens','Bronx','Brooklyn'],
    window = '2H',freq = '15min',
    ):

    df = subset_data(path,boro_sel)
    
    df_rs = sample_sensors(df)

    cond_toss,speed_ts = check_outages(df_rs)



    return df  


