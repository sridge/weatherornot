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

def clean(year = 2019,
    path = f'../nyc_speed_data/*{year}.csv',
    boro_sel = ['Manhattan','Staten Island','Queens','Bronx','Brooklyn'],
    window = '2H',freq = '15min',
    ):

    df = subset_data(path,boro_sel)
    
    # resample each sensor as the average over 15 minute intervals
    # combine into a dataframe with linkId as the column and time as the index 
    df_list = []
    for group in df.groupby('linkId'):
        df_tem = group[1].set_index('DataAsOf')
        df_i = df_tem.rename(columns = {'Speed':df_tem['linkId'][0]})[df_tem['linkId'][0]]
        df_i = df_i.resample(freq).mean()

        df_list.append(df_i)

    df_rs = pd.concat(df_list, axis=1, ignore_index=False)

    return df  


