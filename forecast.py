import requests
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import pytz
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

def array_from_json(json_out,out_var):

    arr_out = np.ones(len(json_out),dtype='object')
    for ind in range(0,len(arr_out)):
        arr_out[ind] = json_out[ind][out_var]['value']
    return arr_out

# def forecast_date_range(delta = 24, hour_i_edt = 8, hour_f_edt = 2, hour_i_est = 9, hour_f_est = 3):

#     tomorrow = datetime.now() + timedelta(hours=24)
#     day_after = tomorrow + timedelta(hours=delta)

#     day = tomorrow.day
#     day_day_after = day_after.day
#     month = tomorrow.month
#     month_day_after = day_after.month
#     year = tomorrow.year
#     year_day_after = day_after.year


#     if is_dst():
#         hour_i = hour_i_edt
#         hour_f = hour_f_edt
#     else:
#         hour_i = hour_i_est
#         hour_f = hour_f_est

#     start_time = f"{year}-{month:02}-{day:02}T{hour_i:02}:00:00Z"
#     end_time = f"{year_day_after}-{month_day_after:02}-{day_day_after:02}T{hour_f:02}:00:00Z"

#     return start_time,end_time

def is_dst():
    """Determine whether or not Daylight Savings Time (DST)
    is currently in effect

    from: https://gist.github.com/dpapathanasiou/09bd2885813038d7d3eb
    """

    x = datetime(datetime.now().year, 1, 1, 0, 0, 0, tzinfo=pytz.timezone('US/Eastern')) # Jan 1 of this year
    y = datetime.now(pytz.timezone('US/Eastern'))

    # if DST is in effect, their offsets will be different
    return not (y.utcoffset() == x.utcoffset())

def get_weather_forecast(start_time, end_time, freq='15min', window='2H'):

    base_url = "https://api.climacell.co/v3/weather/forecast/hourly"

    # start_time,end_time = forecast_date_range()

    querystring = {"lat":"40.7812","lon":"-73.9665",
        "unit_system":"us",
        "start_time":start_time,
        "end_time":end_time,
        "fields":"precipitation,temp",
        "apikey":"XjiQRaTyQ1wANUsJNgF4PFSGLfs0VJ03"}

    response = requests.request("GET", base_url, params=querystring)

    return response.json()

def synthetic_weather_forecast(df_weather,window):

    df_weather_pred = df_weather
    df_weather_pred = df_weather_pred.set_index(df_weather.index - pd.Timedelta(window))

    return df_weather_pred

def get_historical_weather(start_time,end_time,freq='15min',window='2h'):
    
    base_url = "https://api.climacell.co/v3/weather/historical/climacell"

    querystring = {"lat":"40.7812","lon":"-73.9665",
        "timestep":"60",
        "unit_system":"us",
        "start_time":start_time,
        "end_time":end_time,
        "fields":"precipitation,temp",
        "apikey":"XjiQRaTyQ1wANUsJNgF4PFSGLfs0VJ03"}

    response = requests.request("GET", url, params=querystring)

    return response.json()

def climacell_json_to_df(json_out,freq):

    df_weather = pd.DataFrame()
    out_vars = ['temp','precipitation','observation_time']

    for out_var in out_vars:

        df_weather[out_var] = array_from_json(json_out,out_var)
    
    df_weather['observation_time'] = pd.to_datetime(df_weather['observation_time']) - pd.Timedelta('30min')
    df_weather = df_weather.set_index('observation_time')

    df_weather = df_weather.resample(freq).pad()

    return df_weather



# def format_data_in_24h(df_merge,df_weather,df_weather_pred):

#     freq = '15min'
#     delta = 8

#     hour_i_edt = 8
#     hour_f_edt = 16
#     hour_i_est = hour_i_edt+1
#     hour_f_est = hour_i_edt+1

#     start_time,end_time = forecast_date_range(delta=delta,
#         hour_i_edt=hour_i_edt,
#         hour_f_edt=hour_f_edt,
#         hour_i_est=hour_i_est,
#         hour_f_est=hour_f_est)

#     forecast_time = pd.date_range(start=start_time,end=end_time,freq=freq)
#     data_in = df_merge.set_index(forecast_time)

#     df_weather_tem = df_weather[df_weather.index.isin(data_in.index)]
#     df_weather_pred_tem = df_weather_pred[df_weather_pred.index.isin(data_in.index)]

#     data_in['p01i'] = df_weather_tem['precipitation']**(1/3)
#     data_in['p01i_pred'] = df_weather_pred_tem['precipitation']**(1/3)

#     return data_in

def format_data_in_2h(ampm,median_speed,df_weather,df_weather_pred):

    data_in = pd.DataFrame()
    data_in['speed'] = median_speed
    dhour = (data_in.index.hour.values+data_in.index.minute.values/60)

    df_weather_tem = df_weather[df_weather.index.isin(data_in.index)]
    df_weather_pred_tem = df_weather_pred[df_weather_pred.index.isin(data_in.index)]

    #create a feature for hour of day, normalize using sin
    data_in['hour'] = np.sin((dhour/24)*2*np.pi)

    data_in['speed_diff'] = data_in['speed'].diff()

    data_in['lt_32'] = (df_weather_tem['temp'] < 32).astype(int)

    # apply the same normalization applied to the training data
    mn = pd.read_csv(f'./forecast/mean_{ampm}.csv',index_col=0).T
    std = pd.read_csv(f'./forecast/std_{ampm}.csv',index_col=0).T

    col_list = list(data_in.columns)

    data_in = (data_in-mn[col_list])/std[col_list]

    # weather is normalized differently
    data_in['p01i'] = df_weather_tem['precipitation']**(1/3)
    data_in['p01i_pred'] = df_weather_pred_tem['precipitation']**(1/3)

    # order columns like the training dataset
    reorder_colums = ['p01i','tmpf','p01i_pred','speed',
        'hour','speed_diff','lt_32']
    
    data_in = data_in[reorder_colums]

    return data_in

def speed_forecast_2h(model_name,data_in):

    model = tf.keras.models.load_model(model_name)
    return model.predict(data_in.values[np.newaxis,])[0,:,1]

def speed_forecast(data_in):

    data_in = data_in.drop(['tmpf','tmpf_pred'],axis=1)
    data_in = data_in.reset_index(drop=True)
    speed_forecast = data_in['speed'].copy()
    
    model_name = 'weather_morning'
    model = tf.keras.models.load_model(model_name)
    
    for ind in tqdm(range(0,24,8)):

        pred = model.predict(data_in[ind:ind+8].values[np.newaxis,].astype(np.float32))[0,:,1]
        data_in[ind+8:ind+16]['speed'] = pred
        plt.plot(pred)

    speed_forecast = data_in['speed']
    return speed_forecast