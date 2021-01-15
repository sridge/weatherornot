# Weather or Not
Weather or Not is a service that provides live forecast of the impact of weather on New York City traffic. Currently hosted on Heroku (free) but also can be deployed on an AWS instance. NOTE: NYC's datafeed is currently unreliable so the data is stale

## Organization
#### forecast: 
Python module that runs the forecast system in the background
#### clean: 
Python module for cleaning New York City's traffic data that is ingested by the forecast system
#### notebooks:
Jupyter notebooks used to train the model
#### src:
FastAPI code that serves the html signup portal
#### templates: 
Jinja templates
#### static: 
forecast image updated by the forecast system
#### data: 
contains the users who are currently signed up for alerts

