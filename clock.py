from apscheduler.schedulers.blocking import BlockingScheduler
import forecast

sched = BlockingScheduler()

@sched.scheduled_job('interval', minutes=3)
def timed_job():
    print('forecast starting')
    forecast.run_forecast_sytem()
    print('forecast finished')
    

@sched.scheduled_job('cron', day_of_week='mon-fri', hour=17)
def scheduled_job():
    print('This job is run every weekday at 5pm.')

sched.start()