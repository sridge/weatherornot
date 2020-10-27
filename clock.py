from apscheduler.schedulers.blocking import BlockingScheduler
sched = BlockingScheduler()

import forecast
import os

@sched.scheduled_job('interval', minutes=3)
def timed_job():
    print('forecast starting')
    forecast.run_forecast_system()
    print('forecast finished')

sched.start()