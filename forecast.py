import requests
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import pytz
import tensorflow as tf

def array_from_json(json_out,out_var):
    
    arr_out = np.ones(len(json_out),dtype='object')
    for ind in range(0,len(arr_out)):
        arr_out[ind] = json_out[ind][out_var]['value']
    return arr_out

def forecast_date_range(delta = 24, hour_i_edt = 8, hour_f_edt = 2, hour_i_est = 9, hour_f_est = 3):

    tomorrow = datetime.now() + timedelta(hours=24)
    day_after = tomorrow + timedelta(hours=delta)

    day = tomorrow.day
    day_day_after = day_after.day
    month = tomorrow.month
    month_day_after = day_after.month
    year = tomorrow.year
    year_day_after = day_after.year


    if is_dst():
        hour_i = hour_i_edt
        hour_f = hour_f_edt
    else:
        hour_i = hour_i_est
        hour_f = hour_f_est

    start_time = f"{year}-{month:02}-{day:02}T{hour_i:02}:00:00Z"
    end_time = f"{year_day_after}-{month_day_after:02}-{day_day_after:02}T{hour_f:02}:00:00Z"

    return start_time,end_time

def is_dst():
    """Determine whether or not Daylight Savings Time (DST)
    is currently in effect

    from: https://gist.github.com/dpapathanasiou/09bd2885813038d7d3eb
    """

    x = datetime(datetime.now().year, 1, 1, 0, 0, 0, tzinfo=pytz.timezone('US/Eastern')) # Jan 1 of this year
    y = datetime.now(pytz.timezone('US/Eastern'))

    # if DST is in effect, their offsets will be different
    return not (y.utcoffset() == x.utcoffset())

def get_weather_forecast(freq='15min',window='2H'):

    base_url = "https://api.climacell.co/v3/weather/forecast/hourly"

    start_time,end_time = forecast_date_range()

    querystring = {"lat":"40.7812","lon":"-73.9665",
        "unit_system":"us",
        "start_time":start_time,
        "end_time":end_time,
        "fields":"precipitation,temp",
        "apikey":"XjiQRaTyQ1wANUsJNgF4PFSGLfs0VJ03"}

    response = requests.request("GET", base_url, params=querystring)
    json_out = response.json()

    df_weather = pd.DataFrame()
    out_vars = ['temp','precipitation','observation_time']

    for out_var in out_vars:

        df_weather[out_var] = array_from_json(json_out,out_var)
    
    df_weather['observation_time'] = pd.to_datetime(df_weather['observation_time']) - pd.Timedelta('30min')
    df_weather = df_weather.set_index('observation_time')

    df_weather = df_weather.resample(freq).pad()

    df_weather_pred = df_weather
    df_weather_pred = df_weather_pred.set_index(df_weather.index - pd.Timedelta(window))

    return df_weather,df_weather_pred


def generate_data_in(df_merge,df_weather,df_weather_pred):

    freq = '15min'
    delta = 8

    hour_i_edt = 8
    hour_f_edt = 16
    hour_i_est = hour_i_edt+1
    hour_f_est = hour_i_edt+1

    start_time,end_time = forecast_date_range(delta=delta,
        hour_i_edt=hour_i_edt,
        hour_f_edt=hour_f_edt,
        hour_i_est=hour_i_est,
        hour_f_est=hour_f_est)


    forecast_time = pd.date_range(start=start_time,end=end_time,freq=freq)
    data_in = df_merge.set_index(forecast_time)

    df_weather_tem = df_weather[df_weather.index.isin(data_in.index)]
    df_weather_pred_tem = df_weather_pred[df_weather_pred.index.isin(data_in.index)]

    data_in['p01i'] = df_weather_tem['precipitation']**(1/3)
    data_in['p01i_pred'] = df_weather_pred_tem['precipitation']**(1/3)

    return data_in

def forecast_2h(model_name,data_in):

    model = tf.keras.models.load_model(model_name)
    return model.predict(data_in)

def forecast_24h(data_in):

    for ind in range(8,72,8):

        if ind < 32:
            ampm = 'am'
            pred = forecast_2h(model_name=f'weather_{ampm}', data_in = data_in[ind-8:ind])

        if ind > 32:
            ampm = 'pm'
            pred = forecast_2h(model_name=f'weather_{ampm}', data_in = data_in[ind-8:ind])