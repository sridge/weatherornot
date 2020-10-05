from email.message import EmailMessage
import os
import smtplib

import ssl
import pandas as pd

# carriers = ['@txt.att.net','@vtext.com','@tmomail.net']
# {'leave_nyc':'07','get_to_nyc':'07'}

def send_alerts(
    alert_thresh = 3,
    sender_email = 'smr1020@gmail.com',
    main_url = 'http://weatherornot.herokuapp.com',
    port = 465):

    forecast_link = f'{main_url}/forecast'

    password = os.environ['password']

    # Create a secure SSL context
    context = ssl.create_default_context()

    df_users = pd.read_csv('users.csv',index_col=0)
    # df_forecast = pd.read_csv('forecast.csv')

    # hours = df_forecast['delta'][df_forecast['delta']>alert_thresh]['hour']
    hours = [3,12,17]

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
