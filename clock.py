from multiprocessing import Process

import forecast
from apscheduler.schedulers.blocking import BlockingScheduler
sched = BlockingScheduler()


@sched.scheduled_job('interval', minutes=5, id='forecast')
def timed_job():
    print('-----------------------')
    print('forecast starting')
    print('-----------------------')

    p = Process(target=forecast.run_forecast_system(), name='forecast_sys')
    p.start()
    p.join(timeout=100)
    p.terminate()

    print('-----------------------')
    print('forecast finished')
    print('-----------------------')


sched.start()
