from apscheduler.schedulers.blocking import BlockingScheduler
sched = BlockingScheduler()

import forecast

@sched.scheduled_job('interval', minutes=3)
def timed_job():
    print('-----------------------')
    print('forecast starting')
    print('-----------------------')
    forecast.run_forecast_system()
    print('-----------------------')
    print('forecast finished')
    print('-----------------------')

sched.start()