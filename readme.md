# Weather or Not
## link: [seanridge.com/weatherornot](http://www.seanridge.com/weatherornot)
Weather or Not is a service that provides live forecast of the impact of weather on the next two hours of New York City traffic. Deployed on an AWS instance using the included dockerfile.

## Organization
#### forecast: 
Python module for the forecast system. Advanced Python Scheduler is used to run it in the background
#### clean: 
Python module for cleaning New York City's traffic data that is ingested by the forecast system
#### notebooks:
Jupyter notebooks used to train the model
#### src:
FastAPI code that serves the html signup portal
#### templates: 
Jinja templates
#### static: 
Contains forecast image hosted by FastAPI, updated by the forecast system (./forecast)
#### data: 
contains the users who are currently signed up for alerts

