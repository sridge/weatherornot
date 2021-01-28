from email.message import EmailMessage
import os
import smtplib
import ssl

import requests
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import pytz
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

import clean

def array_from_json(json_out,out_var):

    arr_out = np.ones(len(json_out),dtype='object')
    for ind in range(0,len(arr_out)):
        arr_out[ind] = json_out[ind][out_var]['value']
    return arr_out

def is_dst():
    
    """Determine whether or not Daylight Savings Time (DST)
    is currently in effect

    from: https://gist.github.com/dpapathanasiou/09bd2885813038d7d3eb
    """

    x = datetime(datetime.now().year, 1, 1, 0, 0, 0, tzinfo=pytz.timezone('US/Eastern')) # Jan 1 of this year
    y = datetime.now(pytz.timezone('US/Eastern'))

    # if DST is in effect, their offsets will be different
    return not (y.utcoffset() == x.utcoffset())

def time_nearest_15min():

    now = datetime.utcnow()

    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    minute = minute - minute%15

    return f'{year}-{month:02}-{day:02}T{hour:02}:{minute:02}:00Z'

def get_weather_forecast(start_time, end_time, freq='15min', window='2H'):

    base_url = "https://api.climacell.co/v3/weather/forecast/hourly"

    # start_time,end_time = forecast_date_range()

    querystring = {"lat":"40.7812","lon":"-73.9665",
        "unit_system":"us",
        "start_time":start_time,
        "end_time":end_time,
        "fields":"precipitation,temp",
        "apikey":os.environ['apikey']}

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
        "apikey":os.environ['apikey']}

    response = requests.request("GET", base_url, params=querystring)

    return response.json()

def get_weather_data(freq,end_time):
    # end_time = median_speed.index[-1].isoformat()

    start_time = (pd.to_datetime(end_time)-pd.Timedelta('2H')).isoformat()
    json_out = get_historical_weather(start_time=start_time,end_time=end_time)
    df_weather = climacell_json_to_df(json_out,freq)

    start_time = end_time
    end_time = (pd.to_datetime(start_time)+pd.Timedelta('2H')).isoformat()
    json_out = get_weather_forecast(start_time=start_time,end_time=end_time)
    df_weather_pred = climacell_json_to_df(json_out,freq)

    return df_weather,df_weather_pred

def climacell_json_to_df(json_out,freq):

    df_weather = pd.DataFrame()
    out_vars = ['temp','precipitation','observation_time']

    for out_var in out_vars:

        df_weather[out_var] = array_from_json(json_out,out_var)
    
    df_weather['observation_time'] = pd.to_datetime(df_weather['observation_time']) - pd.Timedelta('30min')
    df_weather = df_weather.set_index('observation_time')

    df_weather = df_weather.resample(freq).pad()

    return df_weather

def normalize(ampm,data_in):
    
    # apply the same normalization applied to the training data
    mn = pd.read_csv(f'./forecast/mean_{ampm}.csv',index_col=0).T
    std = pd.read_csv(f'./forecast/std_{ampm}.csv',index_col=0).T
    # some columns have a different normalization
    col_list = list(data_in.columns)
    # use array broadcasting
    data_in = data_in.sub(mn[col_list].iloc[0], axis='columns')
    data_in = data_in.div(std[col_list].iloc[0], axis='columns')
    
    return data_in
    
def denormalize_speed(ampm,df_forecast):
    
    # apply the same normalization applied to the training data
    mn = pd.read_csv(f'./forecast/mean_{ampm}.csv',index_col=0).T
    std = pd.read_csv(f'./forecast/std_{ampm}.csv',index_col=0).T
    # use array broadcasting
    df_forecast = df_forecast.mul(std['speed'].iloc[0], axis='columns')
    df_forecast = df_forecast.add(mn['speed'].iloc[0], axis='columns')
    
    return df_forecast

def format_data_in_2h(median_speed,df_weather,df_weather_pred):

    data_in = pd.DataFrame()
    data_in['speed'] = median_speed
    
    df_weather = df_weather_pred.set_index(df_weather.index)  
    df_weather_tem = df_weather[df_weather.index.isin(data_in.index)]
    df_weather_pred_tem = df_weather_pred[df_weather_pred.index.isin(data_in.index)]

    # create a feature for hour of day, normalize using sin
    eastern_index = data_in.index.tz_convert('US/Eastern')
    dhour = (eastern_index.hour.values+eastern_index.minute.values/60)
    
    if dhour.max() > 13:
        ampm = 'morning'
    else:
        ampm = 'afternoon'
    
    data_in['hour'] = np.sin((dhour/24)*2*np.pi)

    data_in['speed_diff'] = data_in['speed'].diff()

    data_in['lt_32'] = (df_weather_tem['temp'] < 32).astype(int)

    # apply the same normalization applied to the training data
    mn = pd.read_csv(f'./forecast/mean_{ampm}.csv',index_col=0).T
    std = pd.read_csv(f'./forecast/std_{ampm}.csv',index_col=0).T
    # some columns have a different normalization
    col_list = list(data_in.columns)
    # use array broadcasting
    data_in = data_in.sub(mn[col_list].iloc[0], axis='columns')
    data_in = data_in.div(std[col_list].iloc[0], axis='columns')

    # weather is normalized differently
    data_in['p01i'] = df_weather_tem['precipitation']**(1/3)
    data_in['p01i_pred'] = df_weather_pred_tem['precipitation']**(1/3)

    # order columns like the training dataset
    reorder_colums = ['p01i','p01i_pred','speed',
        'hour','speed_diff','lt_32']
    
    data_in = data_in[reorder_colums]

    return ampm,data_in

def save_forecast_image(median_speed,df_forecast):
    wdir = os.environ(['wdir'])
    os.chdir(wdir)

    fig = plt.figure(dpi=300)
    plt.title('Live Traffic Forecast')
    median_speed.tz_convert('US/Eastern').plot(label='Past Traffic')
    df_forecast['weather'].tz_convert('US/Eastern').plot(marker='.',linestyle='None',label='Forecasted Weather Impact')
    df_forecast['no_weather'].tz_convert('US/Eastern').plot(marker='.',linestyle='None',label='Forecast Without Weather Impact')
    plt.legend()
    plt.grid()
    plt.ylabel('speed (MPH)')
    plt.xlabel('time')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('bottom')
    plt.savefig('./static/forecast.png')

def speed_forecast_2h(boro_sel = ['Manhattan','Staten Island','Queens','Bronx','Brooklyn'],freq = '15min'):
    wdir = os.environ(['wdir'])
    os.chdir(wdir)
    # currently pulling from archive because the system isn't reporting live speed anymore
    df = clean.load_speed_from_api()
    df = df.rename(columns={'SPEED':'Speed','LINK_ID':'linkId','DATA_AS_OF':'DataAsOf'})
    df = clean.subset_speed_data(df,boro_sel,link_id_path='./forecast/linkIds.csv')
    df_rs = clean.downsample_sensors(df,freq)

    _,median_speed = clean.nyc_median_speed(df_rs)
    median_speed = median_speed.interpolate(method='linear')
    median_speed.index = median_speed.index.tz_localize(tz='US/Eastern')
    median_speed.index = median_speed.index.tz_convert('UTC')

    df_weather,df_weather_pred = get_weather_data(freq,end_time=time_nearest_15min())
    
    # ---------------------------
    # this block is matching the time of the archived speed timeseries with the live weather timeseries
    speed_times = pd.Series(median_speed.index.values.astype(str))
    forecast_times = pd.Series(df_weather.index[1:].values.astype(str))
    speed_times = speed_times.str.slice(11,19)
    forecast_times = forecast_times.str.slice(11,19)
    df_speed = pd.DataFrame()
    df_speed['median_speed'] = median_speed
    df_speed['speed_time'] = speed_times.values
    df_speed = df_speed[df_speed['speed_time'].isin(forecast_times)]
    median_speed = df_speed['median_speed']
    median_speed.index = df_weather.index[1:]
    # ---------------------------

    ampm,data_in = format_data_in_2h(median_speed,df_weather,df_weather_pred)
    data_in = data_in.astype('float32')
    data_in = data_in.fillna(method='bfill')

    model_name=f'./forecast/weather_{ampm}'
    model = tf.keras.models.load_model(f'{model_name}')
    weather = model.predict(data_in.values[np.newaxis,])[0,:,1]

    weather = pd.Series(weather)
    weather.index = data_in.index + pd.Timedelta('2H')

    model_name=f'./forecast/no_weather_{ampm}'
    model = tf.keras.models.load_model(f'{model_name}')
    data_in = data_in.drop(['p01i','p01i_pred','lt_32'],axis=1)
    no_weather = model.predict(data_in.values[np.newaxis,])[0,:,1]

    no_weather = pd.Series(no_weather)
    no_weather.index = data_in.index + pd.Timedelta('2H')

    df_forecast = pd.DataFrame()
    df_forecast['weather'] = weather
    df_forecast['no_weather'] = no_weather
    df_forecast = denormalize_speed(ampm,df_forecast)
    
    save_forecast_image(median_speed,df_forecast)
    
    return df_forecast

def alert_hours(df_forecast,thresh=0):
    
    df_forecast['weather_anom'] = df_forecast['weather'] - df_forecast['no_weather']
    
    hours = []
    
    for group in df_forecast.groupby(df_forecast.index.hour):
        cond = (group[1]['weather_anom'] < thresh)
        if group[1]['weather_anom'][cond].any() :
            hours.append(group[0])
            
    return hours

def send_alerts(hours,
    sender_email = 'smr1020@gmail.com',
    main_url = 'http://weatherornotapi.herokuapp.com',
    port = 465):

    wdir = os.environ(['wdir'])
    os.chdir(wdir)

    forecast_link = f'{main_url}/forecast'

    password = os.environ['password']

    # Create a secure SSL context
    context = ssl.create_default_context()

    df_users = pd.read_csv('./data/users.csv',index_col=0)

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:

        server.login(sender_email, password)

        for hour in hours:

            df_users_alert = df_users[(df_users['enter_nyc'] == hour) | (df_users['leave_nyc'] == hour)]
            
            if hour < 12:
                ampm='AM'
            if hour > 12:
                hour = hour-12
                ampm='PM'

            message = (f'Weather is forecast to slow your {hour}:00 {ampm} commute.' 
                f' Check {forecast_link} to plan your commute.')
            
            for user_id,receiver_email in df_users_alert['email'].iteritems():

                remove_from_list = f' Click this link to unsubscribe: {main_url}/main?$remove={user_id}'

                message = message + remove_from_list
                
                em = EmailMessage()
                em['Subject'] = 'Weather or Not Alert'
                em['From'] = sender_email
                em['To'] = receiver_email
                em.set_content(message)
                print(message)
                server.send_message(em)

def run_forecast_system():

    df_forecast = speed_forecast_2h()
    hours = alert_hours(df_forecast,thresh=0)
    send_alerts(hours)