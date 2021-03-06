from email.message import EmailMessage
import os
import smtplib
import ssl

import numpy as np
import pandas as pd
# from twilio.rest import Client

def add_user(number,carrier,enter_nyc,leave_nyc):

    df_users = pd.read_csv('./data/users.csv',index_col=0)
    email = (str(number)+carrier)

    if ~(df_users['email']==email).any():
        df = pd.DataFrame()
        n = 8
        uid = ''.join([str(np.random.randint(0, 9)) for _ in range(0, n)])
        while (df_users.index==uid).any():
            uid = ''.join([str(np.random.randint(0, 9)) for _ in range(0, n)])

        df['uid'] = [uid]
        df = df.set_index('uid')
        df['email'] = [email]
        df['enter_nyc'] = [enter_nyc]
        df['leave_nyc'] = [leave_nyc]

        df_users = df_users.append(df)
        df_users.to_csv('./data/users.csv')

        return 'Success! You\'re now enrolled in text alerts'

    else:
        return 'This number is already enrolled in text alerts'

def remove_user(uid):

    df_users = pd.read_csv('users.csv',index_column=0)
    df_users = df_users[df_users.index != uid]
    df_users.to_csv('./data/users.csv')

# def send_alerts_twilio(
#     hours,
#     main_url = 'http://weatherornot.herokuapp.com'):

#     forecast_link = f'{main_url}/forecast'

#     sender_number = os.environ['sender_number']
#     account_sid = os.environ['account_sid']
#     auth_token = os.environ['auth_token']

#     client = Client(account_sid, auth_token)

#     df_users = pd.read_csv('users_text.csv',index_col=0)

#     for hour in hours:

#         df_users_alert = df_users[(df_users['enter_nyc'] == hour) | (df_users['leave_nyc'] == hour)]
        
#         if hour < 12:
#             ampm='AM'
#         if hour > 12:
#             hour = hour-12
#             ampm='PM'

#         message = (f'Weather is forecast to slow your {hour}:00 {ampm} commute.' 
#             f' Check {forecast_link} to plan your commute.')
        
#         for user_id,receiver_number in df_users_alert['number'].iteritems():

#             remove_from_list = f' Click this link to unsubscribe: {main_url}/main?$remove={user_id}'

#             message = message + remove_from_list
            
#             message = client.messages.create(
#                                         body=message,
#                                         from_=sender_number,
#                                         to=receiver_number
#                                     )

#             print(message.sid)
